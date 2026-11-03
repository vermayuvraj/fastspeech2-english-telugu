"""Independent native-script CTC decoding to cross-check Whisper diagnostics."""

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
from math import gcd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fs2.config import ROOT, sha256, write_json
from fs2.evaluation import score_recognition
from automatic_assessment import env_setup, inventory, progress, stored


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, required=True)
    args = parser.parse_args()
    args.work_dir = args.work_dir.resolve()
    env_setup(args.work_dir)
    os.environ["HF_HUB_OFFLINE"] = "1"
    import huggingface_hub.constants

    huggingface_hub.constants.HF_HUB_OFFLINE = True
    from huggingface_hub import snapshot_download
    import torch
    from transformers import AutoProcessor, Wav2Vec2ForCTC

    revisions = json.loads((ROOT / "docs/automatic_evaluation_models.json").read_text())
    repo = "facebook/mms-1b-all"
    directory = Path(
        snapshot_download(
            repo, revision=revisions[repo], cache_dir=args.work_dir / "cache/huggingface", local_files_only=True
        )
    )
    progress(args.work_dir, "Loading independent MMS cross-check")
    processor = AutoProcessor.from_pretrained(directory, local_files_only=True)
    model = Wav2Vec2ForCTC.from_pretrained(
        directory, local_files_only=True, use_safetensors=True, torch_dtype=torch.float16, attn_implementation="sdpa"
    )
    checkpoint_hash = sha256(directory / "model.safetensors")
    rows = inventory(args.work_dir)
    for language, code in [("English", "eng"), ("Telugu", "tel")]:
        processor.tokenizer.set_target_lang(code)
        model.load_adapter(code, local_files_only=True, use_safetensors=True)
        model = model.to("cuda", dtype=torch.float16).eval()
        selected = [s for s in rows if s["language"] == language]
        result = stored(language, "automatic_asr_mms.json")
        for i, row in enumerate(selected):
            digest = sha256(row["path"])
            old = next((s for s in result["samples"] if s["id"] == row["id"] and s["split"] == row["split"]), None)
            if old and old["wav_sha256"] == digest and result.get("model_revision") == revisions[repo]:
                continue
            progress(args.work_dir, "MMS cross-check " + row["id"], i, len(selected))
            start = time.monotonic()
            y, sr = sf.read(row["path"], dtype="float32")
            divisor = gcd(sr, 16000)
            y = resample_poly(y, 16000 // divisor, sr // divisor).astype(np.float32)
            inputs = processor(y, sampling_rate=16000, return_tensors="pt")
            inputs = {
                k: v.to("cuda", dtype=torch.float16) if v.is_floating_point() else v.to("cuda")
                for k, v in inputs.items()
            }
            with torch.inference_mode():
                logits = model(**inputs).logits
            if not torch.isfinite(logits).all():
                raise ValueError("The MMS cross-check produced nonfinite logits")
            transcript = processor.batch_decode(logits.argmax(-1))[0]
            value = {
                "id": row["id"],
                "split": row["split"],
                "wav_sha256": digest,
                "reference_expanded": row["reference"],
                "transcript": transcript,
                **score_recognition(row["reference"], transcript, language),
                "seconds": time.monotonic() - start,
            }
            result["samples"] = [s for s in result["samples"] if (s["id"], s["split"]) != (row["id"], row["split"])] + [
                value
            ]
            result.update(
                model=repo,
                model_revision=revisions[repo],
                checkpoint_sha256=checkpoint_hash,
                adapter=code,
                adapter_sha256=sha256(directory / f"adapter.{code}.safetensors"),
                implementation="transformers",
                implementation_version=importlib.metadata.version("transformers"),
                scoring_version=3,
                decoding="Greedy CTC, native-script language adapter; no external language model or reference hint.",
                interpretation="An independent automatic transcript diagnostic. It is not phoneme accuracy or human listening. Report both recognizers and their held-out controls.",
            )
            write_json(ROOT / "Results" / language / "automatic_asr_mms.json", result)
            print(row["id"], "CER", round(value["cer_without_spaces"], 3), repr(transcript), flush=True)
    progress(args.work_dir, "Independent MMS cross-check complete", 30, 30)


if __name__ == "__main__":
    main()
