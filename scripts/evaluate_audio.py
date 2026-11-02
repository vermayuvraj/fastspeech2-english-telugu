"""Measure generated waveforms and transcript alignment against held-out recordings.

CTC confidence is a diagnostic proxy, not a listening score or a calibrated
pronunciation accuracy. Ground-truth recordings provide a same-language baseline.
"""

import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fs2.alignment import ForcedAligner
from fs2.audio import acoustic_features
from fs2.config import ROOT, load_config, read_jsonl, sha256, write_json
from fs2.frontend import normalize


def measure(path, text, config, aligner):
    y, sr = sf.read(path, dtype="float32")
    if sr != config["audio"]["sample_rate"] or y.ndim != 1:
        raise ValueError("Expected mono audio at the configured sampling rate")
    row = {
        "sha256": sha256(path),
        "duration_seconds": len(y) / sr,
        "peak": float(np.abs(y).max()),
        "rms": float(np.sqrt(np.mean(y**2))),
        "clipped_fraction": float((np.abs(y) >= 0.999).mean()),
    }
    frames = len(y) // config["audio"]["hop_length"]
    try:
        features = acoustic_features(y, config["audio"])
        voiced = features["voiced"]
        f0 = np.exp(features["log_f0"])[voiced]
        row.update(
            {
                "measured_voiced_fraction": float(voiced.mean()),
                "measured_voiced_f0_median_hz": float(np.median(f0)) if len(f0) else None,
                "measured_voiced_f0_10_90_hz": [float(v) for v in np.percentile(f0, [10, 90])] if len(f0) else None,
            }
        )
    except ValueError as error:
        row["pitch_extraction_error"] = str(error)
    try:
        front = normalize(text, config["language"])
        _, _, confidence = aligner(y, sr, front["words"], frames, config["audio"]["hop_length"])
        row["transcript_ctc_alignment_confidence"] = confidence
    except (ValueError, RuntimeError) as error:
        row["alignment_error"] = str(error)
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--language", choices=["English", "Telugu"], required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    os.environ["TORCH_HOME"] = str(args.work_dir.resolve() / "cache/torch")
    import torch

    torch.set_num_threads(4)
    config = load_config(ROOT / args.language / "config.yaml")
    aligner = ForcedAligner(args.device)
    directory = args.work_dir.resolve() / args.language
    baseline = []
    for row in sorted(read_jsonl(directory / "validation.jsonl"), key=lambda r: r["id"])[:10]:
        value = measure(directory / row["audio_path"], row["raw_text"], config, aligner)
        baseline.append({"id": row["id"], **value})
        print("Baseline", row["id"], value.get("transcript_ctc_alignment_confidence"), flush=True)
    generated = []
    prompts = json.loads((ROOT / "test_sentences" / f"{args.language}.json").read_text(encoding="utf-8"))
    result_dir = ROOT / "Results" / args.language
    for row in prompts:
        value = measure(result_dir / "test_generated_audios" / f"{row['id']}.wav", row["text"], config, aligner)
        generated.append({"id": row["id"], **value})
        print("Generated", row["id"], value.get("transcript_ctc_alignment_confidence"), flush=True)
    write_json(
        result_dir / "objective_audio_evaluation.json",
        {
            "language": args.language,
            "baseline_selection": "first 10 held-out utterances ordered by ID; not used for training or checkpoint selection",
            "interpretation": "CTC confidence is an uncalibrated transcript-compatibility diagnostic, not human pronunciation accuracy, MOS, WER or CER",
            "baseline": baseline,
            "generated": generated,
        },
    )


if __name__ == "__main__":
    main()
