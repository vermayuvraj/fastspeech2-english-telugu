"""Pinned downloads, WAV export, real forced alignment and train-only statistics."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import time
from pathlib import Path

import numpy as np
import soundfile as sf

from .config import ROOT, load_config, read_jsonl, sha256, workspace, write_json, write_jsonl
from .frontend import normalize
from .split import assert_disjoint, select_splits


def download(config: dict, work_dir: Path) -> list[Path]:
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import GatedRepoError
    data = config["dataset"]
    paths = []
    for filename in data["files"]:
        print(f"Download {data['repository']} / {filename}", flush=True)
        from huggingface_hub.errors import HfHubHTTPError, LocalEntryNotFoundError
        from requests import RequestException
        try:
            cached = hf_hub_download(repo_id=data["repository"], filename=filename, repo_type="dataset", revision=data["revision"], cache_dir=work_dir / "cache" / "huggingface", local_files_only=True)
            paths.append(Path(cached))
            continue
        except LocalEntryNotFoundError:
            pass
        for attempt in range(5):
            try:
                path = hf_hub_download(repo_id=data["repository"], filename=filename, repo_type="dataset", revision=data["revision"], cache_dir=work_dir / "cache" / "huggingface")
                break
            except GatedRepoError as error:
                raise RuntimeError("Svarah access is required. Accept the dataset conditions on Hugging Face, then run python scripts/sign_in_huggingface.py. Never put tokens in the repository.") from error
            except (RequestException, LocalEntryNotFoundError, HfHubHTTPError) as error:
                response = getattr(error, "response", None)
                if attempt == 4 or (response is not None and response.status_code in [401, 403, 404]):
                    raise
                print(f"Transient download error ({type(error).__name__}); resumable retry {attempt + 1}/4", flush=True)
                time.sleep(2 ** (attempt + 1))
        paths.append(Path(path))
    return paths


def column(row: dict, configured: str | None, candidates: list[str], kind: str):
    if configured:
        if configured not in row:
            raise ValueError(f"Configured {kind} column {configured!r} missing; columns: {list(row)}")
        return row[configured]
    for key in candidates:
        if key in row and row[key] is not None:
            return row[key]
    raise ValueError(f"Cannot determine {kind}; available columns: {list(row)}. Set dataset.{kind}_field explicitly.")


def speaker_for_row(row: dict, config: dict) -> str:
    fields = config["dataset"].get("speaker_profile_fields")
    if fields:
        if any(field not in row or row[field] is None for field in fields):
            raise ValueError("Missing configured speaker-profile column")
        values = [str(row[field]).strip() for field in fields]
        digest = hashlib.sha256(json.dumps(values, ensure_ascii=False).encode("utf-8")).hexdigest()
        return "profile_" + digest[:16]
    return str(column(row, config["dataset"].get("speaker_field"), ["speaker_id", "speaker", "client_id"], "speaker"))


def export_audio(config: dict, paths: list[Path], out: Path, limit: int | None = None) -> list[dict]:
    import pyarrow.parquet as pq
    from scipy.signal import resample_poly
    from math import gcd
    target_sr = config["audio"]["sample_rate"]
    rows, rejects = [], []
    audio_dir = out / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    sequence = 0
    for path in paths:
        parquet = pq.ParquetFile(path)
        print(f"Read {path.name}: {parquet.schema.names}", flush=True)
        for batch in parquet.iter_batches(batch_size=8):
            for raw in batch.to_pylist():
                uid = f"{config['language'][:2].lower()}_{sequence:06d}"
                sequence += 1
                try:
                    text = str(column(raw, config["dataset"].get("text_field"), ["text", "transcript", "transcription", "sentence"], "text"))
                    speaker = speaker_for_row(raw, config)
                    audio = raw[config["dataset"].get("audio_field", "audio")]
                    if audio.get("bytes") is not None:
                        y, sr = sf.read(io.BytesIO(audio["bytes"]), dtype="float32", always_2d=True)
                    elif audio.get("path") and Path(audio["path"]).is_file():
                        y, sr = sf.read(audio["path"], dtype="float32", always_2d=True)
                    else:
                        raise ValueError("Audio has neither embedded bytes nor an existing local path")
                    y = y.mean(axis=1)
                    duration = len(y) / sr
                    if not config["selection"]["min_seconds"] <= duration <= config["selection"]["max_seconds"]:
                        raise ValueError(f"duration_filter:{duration:.3f}")
                    if not np.isfinite(y).all() or np.max(np.abs(y)) < 1e-5:
                        raise ValueError("nonfinite_or_silent_audio")
                    front = normalize(text, config["language"])
                    if len(front["token_ids"]) > config["selection"]["max_tokens"]:
                        raise ValueError("token_length_filter")
                    digest = hashlib.sha256(str(sr).encode() + np.asarray(y, dtype="<f4").tobytes()).hexdigest()
                    if sr != target_sr:
                        factor = gcd(sr, target_sr)
                        y = resample_poly(y, target_sr // factor, sr // factor).astype(np.float32)
                    # Preserve gain; only an actually clipped input is rescaled.
                    peak = float(np.max(np.abs(y)))
                    if peak > 1:
                        y = y / peak * 0.99
                    wav = audio_dir / f"{uid}.wav"
                    sf.write(wav, y, target_sr, subtype="PCM_16")
                    rows.append({"id": uid, "speaker": speaker, "raw_text": text, "normalized_text": front["normalized_text"], "duration_seconds": len(y) / target_sr, "audio_sha256": digest, "wav_sha256": sha256(wav), "audio_path": f"audio/{uid}.wav", "source_file": path.name, "source_row": sequence - 1})
                except (ValueError, RuntimeError, KeyError) as error:
                    # Schema problems must not silently exclude the entire corpus.
                    if "column" in str(error) or "available columns" in str(error):
                        raise
                    rejects.append({"id": uid, "reason": str(error)})
                if sequence % 100 == 0:
                    print(f"Exported {len(rows)} / read {sequence}", flush=True)
                if limit and len(rows) >= limit:
                    break
            if limit and len(rows) >= limit:
                break
        if limit and len(rows) >= limit:
            break
    write_jsonl(out / "candidates.jsonl", rows)
    write_jsonl(out / "export_rejections.jsonl", rejects)
    write_json(out / "dataset_provenance.json", {"repository": config["dataset"]["repository"], "revision": config["dataset"]["revision"], "files": [{"filename": p.name, "sha256": sha256(p), "bytes": p.stat().st_size} for p in paths], "read_rows": sequence, "accepted_rows": len(rows), "smoke_limit": limit, "config_sha256": config["_config_sha256"]})
    return rows


def prepare(config: dict, out: Path, smoke: bool = False) -> None:
    rows = read_jsonl(out / "candidates.jsonl")
    train_h = config["selection"]["train_hours"]
    val_h = config["selection"]["validation_hours"]
    if smoke:
        # A real-data pipeline test, always isolated from the submission Results.
        train_h, val_h = 0.01, 0.003
    reserve = 1.15
    train, validation, summary = select_splits(rows, train_h * reserve, val_h * reserve, config["seed"])
    summary["target_hours"] = {"train": train_h, "validation": val_h}
    summary["smoke"] = smoke
    write_jsonl(out / "train_candidates.jsonl", train)
    write_jsonl(out / "validation_candidates.jsonl", validation)
    write_json(out / "selection.json", summary)


def feature_extraction(config: dict, out: Path, device: str) -> None:
    from .alignment import ForcedAligner
    from .audio import acoustic_features, token_average
    selection = json.loads((out / "selection.json").read_text(encoding="utf-8"))
    aligner = ForcedAligner(device)
    feature_dir = out / "features"
    feature_dir.mkdir(exist_ok=True)
    result, rejects = {}, []
    for split in ["train", "validation"]:
        accepted, seconds = [], 0.0
        target = selection["target_hours"][split] * 3600
        for row in read_jsonl(out / f"{split}_candidates.jsonl"):
            if seconds >= target:
                break
            feature_file = feature_dir / f"{row['id']}.npz"
            metadata_file = feature_file.with_suffix(".json")
            try:
                if feature_file.is_file() and metadata_file.is_file():
                    metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
                    if metadata["config_sha256"] != config["_config_sha256"] or metadata["wav_sha256"] != row["wav_sha256"]:
                        raise ValueError("Stale feature cache: use a new work directory or remove this utterance's cached features")
                else:
                    y, sr = sf.read(out / row["audio_path"], dtype="float32")
                    front = normalize(row["raw_text"], config["language"])
                    values = acoustic_features(y, config["audio"])
                    tokens, durations, score = aligner(y, sr, front["words"], len(values["mel"]), config["audio"]["hop_length"])
                    if tokens != front["tokens"]:
                        raise ValueError("Frontend/alignment token mismatch")
                    if score < config["alignment"]["min_score"]:
                        raise ValueError(f"alignment_score:{score:.4f}")
                    feature_file_temp = feature_file.with_suffix(".tmp.npz")
                    np.savez_compressed(feature_file_temp, tokens=np.asarray(front["token_ids"], dtype=np.int64), durations=durations, mel=values["mel"], pitch=token_average(values["log_f0"], durations), energy=token_average(values["log_energy"], durations), frame_f0=np.exp(values["log_f0"]), frame_voiced=values["voiced"])
                    os.replace(feature_file_temp, feature_file)
                    metadata = {"config_sha256": config["_config_sha256"], "wav_sha256": row["wav_sha256"], "alignment_score": score, "frames": len(values["mel"]), "frontend": front}
                    write_json(metadata_file, metadata)
                accepted.append({**row, "feature_path": f"features/{row['id']}.npz", "alignment_score": metadata["alignment_score"], "frames": metadata["frames"]})
                seconds += row["duration_seconds"]
            except (ValueError, RuntimeError) as error:
                rejects.append({"id": row["id"], "split": split, "reason": str(error)})
                print(f"Rejected {row['id']}: {error}", flush=True)
            if len(accepted) and len(accepted) % 25 == 0:
                print(f"{split}: {len(accepted)} utterances; {seconds / 3600:.3f} / {target / 3600:.3f} h", flush=True)
        result[split] = accepted
        write_jsonl(out / f"{split}.jsonl", accepted)
    assert_disjoint(result["train"], result["validation"])
    write_jsonl(out / "feature_rejections.jsonl", rejects)
    train_speakers = sorted({r["speaker"] for r in result["train"]})
    if {r["speaker"] for r in result["validation"]} - set(train_speakers):
        raise RuntimeError("Alignment filtering removed every training example for a validation speaker")
    write_json(out / "speakers.json", {speaker: i for i, speaker in enumerate(train_speakers)})
    stats = {}
    for name in ["pitch", "energy"]:
        arrays = []
        for row in result["train"]:
            with np.load(out / row["feature_path"], allow_pickle=False) as values:
                arrays.append(values[name][values["durations"] > 0].astype(np.float64))
        if not arrays:
            raise RuntimeError("No training features were accepted")
        values = np.concatenate(arrays)
        mean, std = float(values.mean()), max(float(values.std()), 1e-6)
        # Reference variance bins expect normalized extrema followed by mean/std.
        stats[name] = [(float(values.min()) - mean) / std, (float(values.max()) - mean) / std, mean, std]
    stats["fit_split"] = "train"
    stats["pitch_representation"] = "token mean of interpolated log F0 (natural log Hz)"
    stats["energy_representation"] = "token mean of log STFT L2 magnitude"
    write_json(out / "stats.json", stats)
    seconds_by_speaker = {speaker: sum(r["duration_seconds"] for r in result["train"] if r["speaker"] == speaker) for speaker in train_speakers}
    summary = {**selection, "duration_seconds": {s: sum(r["duration_seconds"] for r in rows) for s, rows in result.items()}, "utterances": {s: len(rows) for s, rows in result.items()}, "speakers": len(train_speakers), "training_seconds_by_speaker": seconds_by_speaker, "rejected_alignments": len(rejects), "config_sha256": config["_config_sha256"]}
    summary["hours_requirement_met"] = all(summary["duration_seconds"][s] >= selection["target_hours"][s] * 3600 for s in result) and not selection["smoke"]
    write_json(out / "data_summary.json", summary)
    if not summary["hours_requirement_met"] and not selection["smoke"]:
        raise RuntimeError("Insufficient accepted hours. Inspect data_summary.json and rejections; expand candidate selection before training.")


def main(default_language: str | None = None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / default_language / "config.yaml" if default_language else None, required=not bool(default_language))
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--stage", choices=["download", "export", "prepare", "features", "all"], default="all")
    parser.add_argument("--local-parquet-dir", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    if default_language and config["language"] != default_language:
        parser.error("Configuration language does not match this entry point")
    out = workspace(config, args.work_dir)
    os.environ["TORCH_HOME"] = str(args.work_dir.resolve() / "cache" / "torch")
    if args.smoke and "smoke" not in str(args.work_dir).lower():
        parser.error("Use a work directory containing 'smoke' to isolate experimental artifacts")
    if args.stage in {"download", "export", "all"}:
        if args.local_parquet_dir:
            paths = sorted(args.local_parquet_dir.glob("*.parquet"))
            if not paths:
                parser.error("No parquet files found")
        else:
            paths = download(config, args.work_dir.resolve())
        if args.stage in {"export", "all"}:
            export_audio(config, paths, out, limit=40 if args.smoke else None)
    if args.stage in {"prepare", "all"}:
        prepare(config, out, args.smoke)
    if args.stage in {"features", "all"}:
        feature_extraction(config, out, args.device)


if __name__ == "__main__":
    main()
