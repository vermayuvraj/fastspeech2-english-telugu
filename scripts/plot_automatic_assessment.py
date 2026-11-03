"""Plot generated audio against the model's predicted timing and variance traces."""

import json
from pathlib import Path
import sys
import textwrap

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fs2.config import ROOT


def report_figures():
    output = ROOT / "Report/assets"
    output.mkdir(exist_ok=True)
    with plt.rc_context({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False}):
        fig, axes = plt.subplots(2, 2, figsize=(7, 3.8), layout="constrained")
        for i, language in enumerate(["English", "Telugu"]):
            rows = [json.loads(s) for s in (ROOT / "Results" / language / "losses.jsonl").read_text().splitlines() if s]
            for j, cutoff in enumerate([0, 2000]):
                ax = axes[i, j]
                for split, color, style in [("train", "#345b78", "-"), ("validation", "#b37741", "--")]:
                    selected = [s for s in rows if s["split"] == split and s["step"] >= cutoff]
                    ax.plot(
                        [s["step"] for s in selected],
                        [s["total"] for s in selected],
                        color=color,
                        linestyle=style,
                        linewidth=1,
                        label=split.capitalize(),
                    )
                ax.set_title(language + (" complete run" if j == 0 else " after step 2,000"), loc="left", fontsize=9)
                ax.set_xlabel("Optimizer step")
                ax.set_ylabel("Total loss")
                ax.tick_params(labelsize=7)
                ax.grid(axis="y", color="#e6e9ec", linewidth=0.5)
                ax.legend(fontsize=6.5, loc="upper right", framealpha=0.95)
        fig.savefig(output / "training_loss_comparison.png", dpi=220)
        plt.close(fig)
        figures = []
        for language in ["English", "Telugu"]:
            rows = json.loads((ROOT / "Results" / language / "inference_summary.json").read_text(encoding="utf-8"))[
                "samples"
            ]
            figures.extend(s for s in rows if s["focus_word_measurements"])
        fig, axes = plt.subplots(1, 3, figsize=(7, 2.0), layout="constrained")
        metrics = ["duration_frames", "mean_continuous_f0_hz", "mean_stft_energy"]
        labels = ["Target duration (ms)", "Mean target F0 (Hz)", "Mean target energy"]
        for ax, metric, label in zip(axes, metrics, labels):
            values = [
                s["focus_word_measurements"][metric] * (256 / 22050 * 1000 if metric == "duration_frames" else 1)
                for s in figures
            ]
            ax.bar([s["id"] for s in figures], values, color=["#345b78", "#345b78", "#47877e", "#47877e"], width=0.6)
            ax.set_title(label, fontsize=9, loc="left")
            ax.tick_params(labelsize=8)
            ax.grid(axis="y", color="#e6e9ec", linewidth=0.5)
            ax.set_axisbelow(True)
        fig.savefig(output / "target_variance_comparison.png", dpi=220)
        plt.close(fig)


def main():
    for language in ["English", "Telugu"]:
        directory = ROOT / "Results" / language
        data = json.loads((directory / "automatic_acoustic_analysis.json").read_text(encoding="utf-8"))
        output = directory / "automatic_assessment_plots"
        output.mkdir(exist_ok=True)
        for sample in data["samples"]:
            name = sample["id"]
            predicted = np.load(directory / "prediction_traces" / f"{name}.npz")
            measured = np.load(directory / "automatic_acoustic_traces" / f"{name}.npz")
            wave, sr = sf.read(directory / "test_generated_audios" / f"{name}.wav")
            duration = predicted["duration_frames"]
            hop = 256
            dt = hop / sr
            t = np.arange(len(measured["voiced"])) * dt + dt / 2
            model_f0 = np.repeat(predicted["continuous_f0_hz"], duration)
            model_energy = np.repeat(predicted["stft_energy"], duration)
            voiced_f0 = np.where(measured["voiced"], measured["measured_continuous_f0_hz"], np.nan)
            assert len(t) == len(model_f0) == len(model_energy)
            with plt.rc_context({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False}):
                fig, axes = plt.subplots(3, 1, figsize=(12, 6.5), sharex=True, gridspec_kw={"height_ratios": [1, 1, 1]})
                fig.suptitle(
                    f"{name}  |  Generated waveform and automatic acoustic measurements", x=0.09, ha="left", fontsize=13
                )
                axes[0].plot(np.arange(len(wave))[::22] / sr, wave[::22], color="#345b78", linewidth=0.5)
                axes[0].set_ylabel("Amplitude")
                for i, word in enumerate(sample["words"], 1):
                    for ax in axes:
                        ax.axvline(word["start_seconds"], color="#b5bdc5", linewidth=0.45, linestyle=":")
                    middle = (word["start_seconds"] + word["end_seconds"]) / 2
                    axes[0].text(
                        middle, 0.92, str(i), transform=axes[0].get_xaxis_transform(), ha="center", va="top", fontsize=8
                    )
                axes[1].plot(t, model_f0, label="Model continuous F0", color="#286f9a", linewidth=1)
                axes[1].plot(t, voiced_f0, label="WORLD F0 in voiced frames", color="#ad5f32", linewidth=0.9)
                axes[1].set_ylabel("Pitch (Hz)")
                axes[1].legend(loc="upper right", fontsize=8, framealpha=0.95)
                axes[2].plot(
                    t,
                    measured["measured_stft_energy"],
                    label="Measured frame energy",
                    color="#ad5f32",
                    linewidth=0.7,
                    alpha=0.8,
                )
                axes[2].plot(t, model_energy, label="Model token energy", color="#286f9a", linewidth=1)
                axes[2].set_yscale("log")
                axes[2].set_ylabel("STFT energy\n(relative units)")
                axes[2].set_xlabel("Time (seconds)")
                axes[2].legend(loc="upper right", fontsize=8, framealpha=0.95)
                for ax in axes:
                    ax.grid(axis="y", color="#e6e9ec", linewidth=0.5)
                    ax.set_xlim(0, len(wave) / sr)
                legend = "  ".join(f"{i} {w['word']}" for i, w in enumerate(sample["words"], 1))
                footer = (
                    "Model-timed word starts are dotted; they are not independently validated phonetic boundaries.\n"
                    + "\n".join(textwrap.wrap(legend, width=158))
                )
                fig.text(0.09, 0.02, footer, fontsize=7.5, ha="left", va="bottom")
                fig.subplots_adjust(
                    left=0.09, right=0.98, top=0.9, bottom=0.17 if len(legend) < 300 else 0.2, hspace=0.16
                )
                fig.savefig(output / f"{name}.png", dpi=190, bbox_inches="tight", pad_inches=0.12)
                plt.close(fig)
            print(name, "overlay saved", flush=True)
    report_figures()


if __name__ == "__main__":
    main()
