"""Check assignment assets and provenance. Missing evidence always fails the audit."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fs2.config import ROOT, load_config, sha256, write_json
from fs2.split import assert_disjoint


def audit():
    checks = []
    def check(name, passed, detail=""):
        checks.append({"check": name, "passed": bool(passed), "detail": detail})
    checkpoints = []
    for language in ["English", "Telugu"]:
        config = load_config(ROOT / language / "config.yaml")
        result = ROOT / "Results" / language
        for filename in ["preprocess.py", "train.py", "inference.py", "config.yaml"]:
            check(f"{language}: {filename}", (ROOT / language / filename).is_file())
        checkpoint = ROOT / language / "checkpoints/best.pt"
        check(f"{language}: trained checkpoint", checkpoint.is_file())
        if checkpoint.is_file():
            checkpoints.append(sha256(checkpoint))
            check(f"{language}: GitHub file size", checkpoint.stat().st_size < 100_000_000, "Use a public release asset or LFS if larger")
        for name in ["training_loss.png", "validation_loss.png", "losses.jsonl", "training_summary.json", "inference_summary.json"]:
            check(f"{language}: {name}", (result / name).is_file())
        summary_path = result / "data_manifests/data_summary.json"
        if summary_path.is_file():
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            check(f"{language}: 5 h / 1 h data", summary.get("hours_requirement_met") and not summary.get("smoke"), str(summary.get("duration_seconds")))
            check(f"{language}: configuration provenance", summary.get("config_sha256") == config["_config_sha256"])
            from fs2.config import read_jsonl
            try:
                assert_disjoint(read_jsonl(result / "data_manifests/train.jsonl"), read_jsonl(result / "data_manifests/validation.jsonl"))
                check(f"{language}: disjoint IDs/audio/prompts", True)
            except (ValueError, FileNotFoundError) as error:
                check(f"{language}: disjoint IDs/audio/prompts", False, str(error))
        else:
            check(f"{language}: data provenance", False)
        training = result / "training_summary.json"
        if training.is_file():
            summary = json.loads(training.read_text(encoding="utf-8"))
            check(f"{language}: full training run", summary.get("completed_steps", 0) >= config["training"]["min_steps"] and not summary.get("smoke"), str(summary.get("completed_steps")))
            check(f"{language}: checkpoint hash", checkpoint.is_file() and summary.get("checkpoint_sha256") == sha256(checkpoint))
        rows = json.loads((ROOT / "test_sentences" / f"{language}.json").read_text(encoding="utf-8"))
        for row in rows:
            for suffix in [f"test_generated_audios/{row['id']}.wav", f"prediction_traces/{row['id']}.json", *[f"variance_adaptor_plots/{key}/{row['id']}.png" for key in ["duration", "pitch", "energy"]]]:
                check(f"{language}/{suffix}", (result / suffix).is_file())
            trace = result / "prediction_traces" / f"{row['id']}.json"
            if trace.is_file():
                value = json.loads(trace.read_text(encoding="utf-8"))
                check(f"{row['id']}: exact prompt and real run", value.get("raw_text") == row["text"] and not value.get("smoke"))
        reviews_path = result / "listening_review.json"
        reviews = json.loads(reviews_path.read_text(encoding="utf-8")) if reviews_path.is_file() else {}
        for row in rows:
            review = reviews.get(row["id"], {})
            check(f"{row['id']}: listening observations", all(str(review.get(k, "")).strip() and str(review[k]).strip().lower() != "pending" for k in ["pronunciation", "overall_quality", "specific_observations"]))
    check("Independent checkpoint files", len(checkpoints) == 2 and len(set(checkpoints)) == 2)
    report = ROOT / "Report/Technical_Report.pdf"
    check("Technical report PDF", report.is_file())
    review_path = ROOT / "Report/review_status.json"
    if review_path.is_file():
        review = json.loads(review_path.read_text(encoding="utf-8"))
        if review.get("report_review_required"):
            check("Current report approved by user", report.is_file() and review.get("approved_report_sha256") == sha256(report), "Await explicit user review of the current PDF")
            check("Google Form submission authorized by user", review.get("google_form_submission_authorized", False))
    value = {"ready_for_submission": all(row["passed"] for row in checks), "checks_passed": sum(row["passed"] for row in checks), "checks_total": len(checks), "checks": checks, "public_repository_verification": "pending", "form_submission": "pending"}
    write_json(ROOT / "Report/submission_audit.json", value)
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()
    value = audit()
    print(f"{value['checks_passed']}/{value['checks_total']} checks passed. Ready for submission: {value['ready_for_submission']}")
    for row in value["checks"]:
        if not row["passed"]:
            print(f"MISSING: {row['check']} {row['detail']}")
    if args.strict and not value["ready_for_submission"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
