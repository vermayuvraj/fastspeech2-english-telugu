"""Run reproducible automatic assessment; this never fabricates listening ratings."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import random
import sys
import time

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fs2.audio import acoustic_features
from fs2.config import ROOT, load_config, read_jsonl, sha256, write_json
from fs2.evaluation import score_recognition, token_spans, weighted_mean
from fs2.frontend import normalize

_DLL_HANDLES = []


def progress(work, stage, done=0, total=30):
    value = {
        "stage": stage,
        "completed": done,
        "total": total,
        "pid": os.getpid(),
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    write_json(work / "automatic_evaluation_progress.json", value)
    status_path = ROOT / "STATUS.json"
    status = json.loads(status_path.read_text(encoding="utf-8")) if status_path.exists() else {}
    status.update(
        state="automatic_evaluation_running",
        stage=f"{stage} ({done}/{total})",
        pid=os.getpid(),
        google_form_submission_authorized=False,
    )
    write_json(status_path, status)
    print(value["stage"], done, "/", total, flush=True)


def inventory(work):
    rows = []
    # Generated samples first: pilot results become available before controls.
    for language in ["English", "Telugu"]:
        result = ROOT / "Results" / language
        samples = json.loads((result / "inference_summary.json").read_text(encoding="utf-8"))["samples"]
        for row in samples:
            rows.append(
                {
                    "language": language,
                    "split": "generated",
                    "id": row["id"],
                    "path": result / "test_generated_audios" / f"{row['id']}.wav",
                    "reference": row["expanded_text"],
                }
            )
    for language in ["English", "Telugu"]:
        directory = work / language
        for row in sorted(read_jsonl(directory / "validation.jsonl"), key=lambda r: r["id"])[:10]:
            rows.append(
                {
                    "language": language,
                    "split": "held_out",
                    "id": row["id"],
                    "path": directory / row["audio_path"],
                    "reference": normalize(row["raw_text"], language)["expanded_text"],
                }
            )
    return rows


def stored(language, name):
    path = ROOT / "Results" / language / name
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"samples": []}


def env_setup(work):
    os.environ["USE_TF"] = "0"
    os.environ["USE_FLAX"] = "0"
    os.environ["HF_HOME"] = str(work / "cache/huggingface_home")
    os.environ["HF_HUB_CACHE"] = str(work / "cache/huggingface")
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(work / "cache/huggingface")
    os.environ["TORCH_HOME"] = str(work / "cache/torch")
    os.environ["UTMOSV2_CHACHE"] = str(work / "cache/utmosv2")
    os.environ["HF_HUB_DISABLE_XET"] = "1"
    # CTranslate2 uses the CUDA/cuDNN DLLs shipped with the installed PyTorch.
    import torch

    torch.set_num_threads(4)
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    lib = Path(torch.__file__).parent / "lib"
    if os.name == "nt":
        os.environ["PATH"] = str(lib) + os.pathsep + os.environ["PATH"]
        _DLL_HANDLES.append(os.add_dll_directory(str(lib)))


def asr(args, rows, revisions):
    from faster_whisper import WhisperModel
    from huggingface_hub import snapshot_download

    repo = "Systran/faster-whisper-large-v3"
    directory = snapshot_download(
        repo, revision=revisions[repo], cache_dir=args.work_dir / "cache/huggingface", local_files_only=True
    )
    model = WhisperModel(
        directory,
        device=args.device,
        compute_type="int8_float16" if args.device == "cuda" else "int8",
        cpu_threads=4,
        num_workers=1,
    )
    for i, row in enumerate(rows):
        result = stored(row["language"], "automatic_asr.json")
        digest = sha256(row["path"])
        old = next((s for s in result["samples"] if s["id"] == row["id"] and s["split"] == row["split"]), None)
        if (
            old
            and old["wav_sha256"] == digest
            and result.get("model_revision") == revisions[repo]
            and old.get("reference_expanded") == row["reference"]
            and result.get("scoring_version") == 3
            and not args.force
        ):
            continue
        progress(args.work_dir, "Independent ASR " + row["id"], i, len(rows))
        start = time.monotonic()
        segments, info = model.transcribe(
            str(row["path"]),
            language="en" if row["language"] == "English" else "te",
            task="transcribe",
            beam_size=5,
            temperature=0,
            condition_on_previous_text=False,
            vad_filter=False,
            word_timestamps=True,
            initial_prompt=None,
        )
        segments = list(segments)
        hypothesis = " ".join(s.text.strip() for s in segments).strip()
        item = {
            "id": row["id"],
            "split": row["split"],
            "wav_sha256": digest,
            "reference_expanded": row["reference"],
            "transcript": hypothesis,
            **score_recognition(row["reference"], hypothesis, row["language"]),
            "language": info.language,
            "seconds": time.monotonic() - start,
            "segments": [
                {
                    "start": s.start,
                    "end": s.end,
                    "text": s.text,
                    "avg_logprob": s.avg_logprob,
                    "no_speech_prob": s.no_speech_prob,
                    "words": [
                        {"word": w.word, "start": w.start, "end": w.end, "probability": w.probability}
                        for w in (s.words or [])
                    ],
                }
                for s in segments
            ],
        }
        result["samples"] = [s for s in result["samples"] if (s["id"], s["split"]) != (row["id"], row["split"])] + [
            item
        ]
        result.update(
            model=repo,
            model_revision=revisions[repo],
            implementation="faster-whisper",
            implementation_version=importlib.metadata.version("faster-whisper"),
            decoding={
                "beam_size": 5,
                "temperature": 0,
                "language_forced": True,
                "reference_prompt_supplied": False,
                "vad_filter": False,
            },
            scoring_version=3,
            scoring="Apply the same deterministic number/time expansion to reference and hypothesis, then NFC/case/punctuation normalization. CER counts Unicode codepoints and ignores whitespace. WER can penalize supplied internal word splits or equivalent number readings. Unexpanded-hypothesis scores are also retained.",
            interpretation="ASR disagreement is an automatic intelligibility diagnostic, not a phoneme accuracy score or proof of a synthesis error.",
        )
        write_json(ROOT / "Results" / row["language"] / "automatic_asr.json", result)
        print(
            row["id"],
            "WER",
            round(item["wer"], 3),
            "CER",
            round(item["cer_without_spaces"], 3),
            repr(hypothesis),
            flush=True,
        )


def quality(args, rows, revisions):
    # All quality backbones are prepared first. Offline loading also avoids
    # Transformers starting a non-daemon automatic safetensors-conversion job.
    os.environ["HF_HUB_OFFLINE"] = "1"
    import huggingface_hub.constants

    huggingface_hub.constants.HF_HUB_OFFLINE = True
    import torch
    import utmosv2
    from huggingface_hub import hf_hub_download

    repo = "sarulab-speech/UTMOSv2"
    checkpoint = hf_hub_download(
        repo,
        "fold0_s42_best_model.pth",
        revision=revisions[repo],
        cache_dir=args.work_dir / "cache/huggingface",
        local_files_only=True,
    )
    progress(args.work_dir, "Loading UTMOSv2 quality predictor")
    model = utmosv2.create_model(pretrained=True, checkpoint_path=checkpoint, device=args.device)
    checkpoint_hash = sha256(Path(checkpoint))
    for i, row in enumerate(rows):
        result = stored(row["language"], "automatic_quality.json")
        digest = sha256(row["path"])
        old = next((s for s in result["samples"] if s["id"] == row["id"] and s["split"] == row["split"]), None)
        if (
            old
            and old["wav_sha256"] == digest
            and result.get("checkpoint_sha256") == checkpoint_hash
            and not args.force
        ):
            continue
        progress(args.work_dir, "Predicted quality " + row["id"], i, len(rows))
        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)
        wave, sr = sf.read(row["path"], dtype="float32")
        start = time.monotonic()
        predictions = []
        # Repeated random crops are recorded separately; their spread is not a listener confidence interval.
        for _ in range(3):
            value = model.predict(
                data=wave,
                sr=sr,
                device=args.device,
                batch_size=1,
                num_workers=0,
                num_repetitions=1,
                remove_silent_section=False,
                verbose=False,
            )
            predictions.append(float(np.asarray(value).reshape(-1)[0]))
        item = {
            "id": row["id"],
            "split": row["split"],
            "wav_sha256": digest,
            "predicted_mos": float(np.mean(predictions)),
            "crop_predictions": predictions,
            "crop_standard_deviation": float(np.std(predictions)),
            "seconds": time.monotonic() - start,
        }
        if not np.isfinite(item["predicted_mos"]):
            raise ValueError("Nonfinite quality prediction")
        result["samples"] = [s for s in result["samples"] if (s["id"], s["split"]) != (row["id"], row["split"])] + [
            item
        ]
        result.update(
            model=repo,
            model_revision=revisions[repo],
            checkpoint_sha256=checkpoint_hash,
            code_revision="cc2700db57bb83ee13dc31ebe1b868c254e15d09",
            configuration="fusion_stage3 fold0 seed42",
            backbone_revisions={
                k: v
                for k, v in revisions.items()
                if k in {"facebook/wav2vec2-base", "timm/tf_efficientnetv2_s.in21k_ft_in1k"}
            },
            predict_dataset="sarulab",
            repeats=3,
            remove_silent_section=False,
            interpretation="Raw predicted naturalness MOS, not human MOS. Predictions are unbounded and are not clipped to the nominal 1-5 listener scale. Crop variation is not listener uncertainty. Telugu calibration is unverified; compare only cautiously with same-language held-out controls.",
        )
        write_json(ROOT / "Results" / row["language"] / "automatic_quality.json", result)
        print(row["id"], "predicted MOS", round(item["predicted_mos"], 3), flush=True)


def low_energy_intervals(wave, sr, relative_db=-30):
    hop = max(1, round(sr * 0.01))
    frame = max(hop, round(sr * 0.02))
    starts = np.arange(0, max(1, len(wave) - frame + 1), hop)
    rms = np.asarray([np.sqrt(np.mean(wave[s : s + frame] ** 2)) for s in starts])
    low = rms < max(float(rms.max()) * 10 ** (relative_db / 20), 1e-7)
    padded = np.pad(low.astype(int), (1, 1))
    begin, end = np.where(np.diff(padded) == 1)[0], np.where(np.diff(padded) == -1)[0]
    return [
        {"start_seconds": float(a * hop / sr), "duration_ms": float((b - a) * hop / sr * 1000)}
        for a, b in zip(begin, end)
        if (b - a) * hop / sr >= 0.08
    ]


def acoustic(args):
    for language in ["English", "Telugu"]:
        result_dir = ROOT / "Results" / language
        config = load_config(ROOT / language / "config.yaml")
        summaries = []
        for row in json.loads((result_dir / "inference_summary.json").read_text(encoding="utf-8"))["samples"]:
            progress(args.work_dir, "Duration pitch energy " + row["id"], len(summaries), 5)
            path = result_dir / "prediction_traces" / (row["id"] + ".npz")
            trace = np.load(path)
            tokens = trace["tokens"].tolist()
            durations, pitch, energy = trace["duration_frames"], trace["continuous_f0_hz"], trace["stft_energy"]
            sr, hop = config["audio"]["sample_rate"], config["audio"]["hop_length"]
            words, boundaries = token_spans(tokens, durations, pitch, energy, hop, sr)
            wave_path = result_dir / "test_generated_audios" / (row["id"] + ".wav")
            wave, _ = sf.read(wave_path, dtype="float32")
            features = acoustic_features(wave, config["audio"])
            valid = durations > 0
            voiced_f0 = np.exp(features["log_f0"])[features["voiced"]]
            item = {
                "id": row["id"],
                "wav_sha256": sha256(wave_path),
                "trace_sha256": sha256(path),
                "duration_seconds": len(wave) / sr,
                "words": words,
                "boundaries": boundaries,
                "zero_duration_tokens": [{"index": int(i), "token": tokens[i]} for i in np.flatnonzero(~valid)],
                "predicted_f0_range_hz": [float(pitch[valid].min()), float(pitch[valid].max())],
                "predicted_energy_range": [float(energy[valid].min()), float(energy[valid].max())],
                "predicted_duration_cv": float(np.std(durations[valid]) / np.mean(durations[valid])),
                "measured_voiced_fraction": float(features["voiced"].mean()),
                "measured_voiced_f0_median_hz": float(np.median(voiced_f0)),
                "measured_voiced_f0_10_90_hz": [float(v) for v in np.percentile(voiced_f0, [10, 90])],
                "low_energy_intervals_80ms": low_energy_intervals(wave, sr),
                "low_energy_intervals_minus40db_80ms": low_energy_intervals(wave, sr, -40),
                "peak": float(np.abs(wave).max()),
                "clipped_fraction": float(np.mean(np.abs(wave) >= 0.999)),
            }
            if row["id"].startswith("E2"):
                focus = next(w for w in words if w["word"] == "conduct")
                ids = focus["token_indices"]
                item["conduct_grapheme_regions"] = []
                for label, part in [("con", ids[:3]), ("duct", ids[3:])]:
                    item["conduct_grapheme_regions"].append(
                        {
                            "label": label,
                            "duration_ms": float(durations[part].sum() * hop / sr * 1000),
                            "mean_predicted_f0_hz": weighted_mean(pitch[part], durations[part]),
                            "mean_predicted_energy": weighted_mean(energy[part], durations[part]),
                        }
                    )
            # Save frame-level measured contours to support independent inspection.
            out = result_dir / "automatic_acoustic_traces"
            out.mkdir(exist_ok=True)
            np.savez_compressed(
                out / f"{row['id']}.npz",
                voiced=features["voiced"],
                measured_continuous_f0_hz=np.exp(features["log_f0"]),
                measured_stft_energy=np.exp(features["log_energy"]),
            )
            summaries.append(item)
        write_json(
            result_dir / "automatic_acoustic_analysis.json",
            {
                "language": language,
                "human_listening_performed": False,
                "boundary_interpretation": "Word spans and con/duct regions follow model grapheme durations, not validated phonetic segmentation. F0 is interpolated through unvoiced intervals. Low-energy intervals use 20ms RMS windows, 10ms hop, threshold -30dB relative to maximum window RMS, minimum 80ms. A -40dB sensitivity check is also retained; these thresholds do not prove linguistic silence.",
                "samples": summaries,
            },
        )


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["asr", "quality", "acoustics"])
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    args.work_dir = args.work_dir.resolve()
    env_setup(args.work_dir)
    if args.stage == "acoustics":
        acoustic(args)
    else:
        revisions = json.loads((args.work_dir / "evaluation_models.json").read_text(encoding="utf-8"))
        rows = inventory(args.work_dir)
        if args.limit:
            rows = rows[: args.limit]
        {"asr": asr, "quality": quality}[args.stage](args, rows, revisions)
    progress(args.work_dir, args.stage + " stage complete", 1, 1)


if __name__ == "__main__":
    main()
