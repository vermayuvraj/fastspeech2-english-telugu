"""Run both independent experiments sequentially; safely resume after interruptions.

This runner writes no external messages and does not publish or submit anything.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fs2.config import ROOT, load_config, sha256, write_json


def run(args, stage, command):
    write_json(ROOT / "STATUS.json", {"state": "running", "stage": stage, "pid": os.getpid(), "command": command, "updated_at_unix": time.time()})
    print("\n" + stage, flush=True)
    subprocess.run([sys.executable, *command], cwd=ROOT, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--languages", nargs="+", choices=["English", "Telugu"], default=["Telugu", "English"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--wait-for-access", action="store_true")
    args = parser.parse_args()
    work_dir = args.work_dir.resolve()
    lock = work_dir / "runner.lock"
    lock.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        raise SystemExit(f"A runner lock exists at {lock}. Check whether that runner is active before removing a stale lock.")
    os.write(descriptor, str(os.getpid()).encode())
    os.close(descriptor)
    try:
        for language in args.languages:
            config = load_config(ROOT / language / "config.yaml")
            directory = work_dir / language
            base = [f"{language}/preprocess.py", "--work-dir", str(work_dir), "--device", args.device]
            complete = directory / "data_summary.json"
            prepared = complete.is_file() and json.loads(complete.read_text(encoding="utf-8")).get("hours_requirement_met", False)
            if not prepared:
                if not (directory / "candidates.jsonl").is_file():
                    if language == "English" and args.wait_for_access:
                        from huggingface_hub import get_hf_file_metadata, get_token, hf_hub_url
                        from huggingface_hub.errors import GatedRepoError
                        while True:
                            try:
                                data = config["dataset"]
                                get_hf_file_metadata(hf_hub_url(data["repository"], data["files"][0], repo_type="dataset", revision=data["revision"]), token=get_token(), timeout=20)
                                break
                            except GatedRepoError:
                                write_json(ROOT / "STATUS.json", {"state": "waiting_for_svarah_access", "stage": language, "pid": os.getpid(), "updated_at_unix": time.time()})
                                print("Waiting for approved Svarah account access; retry in 60 seconds.", flush=True)
                                time.sleep(60)
                    run(args, f"{language}: download and export", base + ["--stage", "export"])
                if not (directory / "selection.json").is_file():
                    run(args, f"{language}: select disjoint subsets", base + ["--stage", "prepare"])
                run(args, f"{language}: extract aligned targets", base + ["--stage", "features"])
            for name in ["train.jsonl", "validation.jsonl", "data_summary.json", "selection.json", "dataset_provenance.json", "stats.json", "speakers.json", "export_rejections.jsonl", "feature_rejections.jsonl"]:
                source = directory / name
                if source.is_file():
                    destination = ROOT / "Results" / language / "data_manifests" / name
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    destination.write_bytes(source.read_bytes())
            summary_file = ROOT / "Results" / language / "training_summary.json"
            trained = summary_file.is_file() and json.loads(summary_file.read_text()).get("completed_steps", 0) >= config["training"]["min_steps"]
            if not trained:
                run(args, f"{language}: train FastSpeech 2", [f"{language}/train.py", "--work-dir", str(work_dir), "--device", args.device, "--resume"])
            inference_file = ROOT / "Results" / language / "inference_summary.json"
            checkpoint = ROOT / language / "checkpoints/best.pt"
            synthesized = inference_file.is_file() and json.loads(inference_file.read_text(encoding="utf-8")).get("checkpoint_sha256") == sha256(checkpoint)
            if not synthesized:
                run(args, f"{language}: synthesize and plot", [f"{language}/inference.py", "--work-dir", str(work_dir), "--device", args.device])
            evaluation_file = ROOT / "Results" / language / "objective_audio_evaluation.json"
            if not evaluation_file.is_file():
                run(args, f"{language}: measure generated audio", ["scripts/evaluate_audio.py", "--language", language, "--work-dir", str(work_dir), "--device", args.device])
            run(args, "Update technical report", ["scripts/build_report.py"])
        run(args, "Audit submission artifacts", ["scripts/audit_submission.py"])
        write_json(ROOT / "STATUS.json", {"state": "experiments_finished_review_required", "updated_at_unix": time.time(), "next_action": "Listen to every test sample, complete listening_review.json for each language, rebuild the report and run the submission audit."})
    except Exception as error:
        write_json(ROOT / "STATUS.json", {"state": "failed", "error_type": type(error).__name__, "message": str(error), "updated_at_unix": time.time()})
        raise
    finally:
        lock.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
