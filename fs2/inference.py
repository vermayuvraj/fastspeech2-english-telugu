"""Synthesize the exact test prompts and export numerical prediction traces."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from .config import ROOT, sha256, write_json
from .frontend import SYMBOLS, normalize
from .model import create_model
from .vocoder import Vocoder


def plot_variance(tokens, durations, pitch_hz, energy, sample_id, output, sr, hop):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    n = len(tokens)
    indices = np.arange(n)
    step = max(1, n // 65)
    for key, values, label, title in [
        ("duration", durations * hop / sr * 1000, "Predicted duration (ms)", "Duration per romanized grapheme"),
        ("pitch", pitch_hz, "Continuous F0 (Hz)", "Predicted continuous pitch; not measured waveform F0"),
        ("energy", energy, "STFT magnitude L2 (relative units)", "Predicted energy per romanized grapheme"),
    ]:
        fig, ax = plt.subplots(figsize=(max(10, min(18, n * 0.13)), 4.8), constrained_layout=True)
        if key == "duration":
            ax.bar(indices, values, color="#245f8d", width=0.85)
        else:
            valid = durations > 0
            ax.plot(indices, np.where(valid, values, np.nan), color="#245f8d", marker=".", markersize=3)
        ax.set_xticks(indices[::step], [t if t != "|" else "[space]" for t in tokens[::step]], rotation=70, fontsize=8)
        ax.set(xlabel="Model token index (romanization; full labels in prediction CSV)", ylabel=label, title=f"{sample_id} | {title}")
        ax.grid(axis="y", alpha=0.2)
        target = output / "variance_adaptor_plots" / key
        target.mkdir(parents=True, exist_ok=True)
        fig.savefig(target / f"{sample_id}.png", dpi=160)
        plt.close(fig)


def focus_word(front, durations, pitch_hz, energy, word):
    if word not in front["words"]:
        return None
    wi = front["words"].index(word)
    start = 1 + sum(len(w) + 1 for w in front["words"][:wi])
    stop = start + len(word)
    weights = durations[start:stop]
    if weights.sum() == 0:
        return {"word": word, "duration_frames": 0}
    return {"word": word, "token_start": start, "token_end_exclusive": stop, "duration_frames": int(weights.sum()), "mean_continuous_f0_hz": float(np.average(pitch_hz[start:stop], weights=weights)), "mean_stft_energy": float(np.average(energy[start:stop], weights=weights))}


@torch.inference_mode()
def synthesize(checkpoint: Path, sentences: Path, output: Path, cache_dir: Path, device: str, speaker: str | None = None, pitch_control=1.0, energy_control=1.0, duration_control=1.0):
    export = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if export["symbols"] != SYMBOLS:
        raise ValueError("Checkpoint/frontend symbol inventory mismatch")
    config, stats, speakers = export["config"], export["stats"], export["speakers"]
    if export.get("smoke") and "smoke" not in str(output).lower():
        raise ValueError("Diagnostic checkpoints may only write to an output path containing 'smoke'")
    torch.set_num_threads(config["training"]["cpu_threads"])
    model = create_model(config, stats, speakers).to(device).eval()
    model.load_state_dict(export["model"], strict=True)
    exposure = export["data_summary"].get("training_seconds_by_speaker", {})
    speaker = speaker if speaker is not None else max(exposure, key=exposure.get) if exposure else sorted(speakers)[0]
    if speaker not in speakers:
        raise ValueError(f"Unknown speaker {speaker!r}; available: {sorted(speakers)}")
    vocoder = Vocoder(config, cache_dir, device)
    sr, hop = config["audio"]["sample_rate"], config["audio"]["hop_length"]
    rows = json.loads(sentences.read_text(encoding="utf-8"))
    output.mkdir(parents=True, exist_ok=True)
    audio_dir, trace_dir = output / "test_generated_audios", output / "prediction_traces"
    audio_dir.mkdir(exist_ok=True)
    trace_dir.mkdir(exist_ok=True)
    metadata = []
    for row in rows:
        front = normalize(row["text"], config["language"])
        token_ids = torch.tensor([front["token_ids"]], device=device)
        prediction = model(torch.tensor([speakers[speaker]], device=device), token_ids, torch.tensor([len(front["tokens"])], device=device), len(front["tokens"]), p_control=pitch_control, e_control=energy_control, d_control=duration_control)
        duration = prediction[5][0].cpu().numpy().astype(np.int64)
        pitch_z, energy_z = prediction[2][0].cpu().numpy(), prediction[3][0].cpu().numpy()
        pitch = np.exp(pitch_z * stats["pitch"][3] + stats["pitch"][2])
        energy = np.exp(energy_z * stats["energy"][3] + stats["energy"][2])
        mel = prediction[1][0, :int(prediction[9][0])].T.unsqueeze(0)
        if not torch.isfinite(mel).all() or not np.isfinite(pitch).all() or not np.isfinite(energy).all():
            raise RuntimeError(f"Non-finite predictions for {row['id']}")
        audio = vocoder(mel)
        if len(audio) != int(duration.sum()) * hop:
            raise RuntimeError("Vocoder length does not agree with predicted duration")
        path = audio_dir / f"{row['id']}.wav"
        sf.write(path, audio, sr, subtype="PCM_16")
        np.savez_compressed(trace_dir / f"{row['id']}.npz", tokens=np.asarray(front["tokens"]), duration_frames=duration, pitch_normalized=pitch_z, continuous_f0_hz=pitch, energy_normalized=energy_z, stft_energy=energy, log_mel=mel[0].cpu().numpy())
        with (trace_dir / f"{row['id']}.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(["index", "token", "duration_frames", "duration_ms", "continuous_f0_hz", "stft_energy"])
            for i, token in enumerate(front["tokens"]):
                writer.writerow([i, token, int(duration[i]), duration[i] * hop / sr * 1000, float(pitch[i]), float(energy[i])])
        plot_variance(front["tokens"], duration, pitch, energy, row["id"], output, sr, hop)
        valid = duration > 0
        item = {**row, **front, "speaker": speaker, "checkpoint_step": export["step"], "wav_sha256": sha256(path), "duration_seconds": len(audio) / sr, "sample_rate": sr, "samples": len(audio), "rms": float(np.sqrt(np.mean(audio ** 2))), "peak": float(np.abs(audio).max()), "clipped_fraction": float((np.abs(audio) >= 0.999).mean()), "continuous_f0_range_hz": [float(pitch[valid].min()), float(pitch[valid].max())], "energy_range": [float(energy[valid].min()), float(energy[valid].max())], "zero_duration_tokens": int((duration == 0).sum()), "listening_review": "pending", "smoke": export.get("smoke", False)}
        focus = "conduct" if row["id"].startswith("E2") else normalize("పట్టు", "Telugu")["words"][0] if row["id"].startswith("T2") else None
        item["focus_word_measurements"] = focus_word(front, duration, pitch, energy, focus) if focus else None
        write_json(trace_dir / f"{row['id']}.json", item)
        metadata.append(item)
        print(f"{row['id']}: {item['duration_seconds']:.2f} s, speaker={speaker}, zero-duration tokens={item['zero_duration_tokens']}", flush=True)
    write_json(output / "inference_summary.json", {"language": config["language"], "checkpoint_sha256": sha256(checkpoint), "checkpoint_step": export["step"], "vocoder": vocoder.provenance, "controls": {"pitch_normalized_space": pitch_control, "energy_normalized_space": energy_control, "duration": duration_control}, "samples": metadata, "smoke": export.get("smoke", False)})
    return metadata


def main(default_language: str | None = None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / default_language / "checkpoints/best.pt" if default_language else None, required=not bool(default_language))
    parser.add_argument("--sentences", type=Path)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--speaker")
    parser.add_argument("--pitch-control", type=float, default=1.0)
    parser.add_argument("--energy-control", type=float, default=1.0)
    parser.add_argument("--duration-control", type=float, default=1.0)
    args = parser.parse_args()
    if min(args.pitch_control, args.energy_control, args.duration_control) <= 0:
        parser.error("Variance controls must be positive")
    export = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    language = export["language"]
    if default_language and language != default_language:
        parser.error("Checkpoint language mismatch")
    synthesize(args.checkpoint, args.sentences or ROOT / "test_sentences" / f"{language}.json", args.output or ROOT / "Results" / language, args.work_dir.resolve() / "cache/huggingface", args.device, args.speaker, args.pitch_control, args.energy_control, args.duration_control)


if __name__ == "__main__":
    main()
