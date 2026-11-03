"""Refresh result navigation only from completed automatic assessment files."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    for language in ["English", "Telugu"]:
        directory = ROOT / "Results" / language
        review = read(directory / "automatic_review.json")
        data = read(directory / "data_manifests/data_summary.json")
        training = read(directory / "training_summary.json")
        inference = read(directory / "inference_summary.json")
        assert len(review["samples"]) == 5 and review["controls_mms"]["count"] == 10
        lines = [
            f"# {language} experiment",
            "",
            f"Trained on {data['duration_seconds']['train'] / 3600:.4f} hours and validated on {data['duration_seconds']['validation'] / 3600:.4f} hours. Training ended at {training['completed_steps']:,} steps; synthesis uses step {inference['checkpoint_step']:,}.",
            "",
            review["author_statement"],
            "",
            "## Audio and plots",
            "",
            "| Sample | Audio | Duration | Pitch | Energy | Waveform comparison |",
            "|---|---|---|---|---|---|",
        ]
        for sample in review["samples"]:
            name = sample["id"]
            lines.append(
                f"| {name} | [WAV](test_generated_audios/{name}.wav) | [Plot](variance_adaptor_plots/duration/{name}.png) | [Plot](variance_adaptor_plots/pitch/{name}.png) | [Plot](variance_adaptor_plots/energy/{name}.png) | [Overlay](automatic_assessment_plots/{name}.png) |"
            )
        lines.extend(
            [
                "",
                "## Automatic measurements",
                "",
                "| Sample | Whisper CER % | MMS CER % | Raw UTMOS prediction |",
                "|---|---:|---:|---:|",
            ]
        )
        for s in review["samples"]:
            lines.append(
                f"| {s['id']} | {100 * s['cer_without_spaces']:.1f} | {100 * s['mms_cer_without_spaces']:.1f} | {s['predicted_mos']:.2f} |"
            )
        lines.extend(
            [
                "",
                "CER measures transcript disagreement, not pronunciation accuracy. UTMOS is an unbounded quality prediction, not human MOS; predictions below 1 are not listener ratings. See the same-language controls and full limitations before interpreting the numbers.",
                "",
                "- [Per-sentence findings and both ASR transcripts](AUTOMATIC_ASSESSMENT.md)",
                "- [Whisper measurements](automatic_asr.json)",
                "- [Independent MMS measurements](automatic_asr_mms.json)",
                "- [Quality predictions and crop variation](automatic_quality.json)",
                "- [Acoustic measurements](automatic_acoustic_analysis.json)",
                "- [Assessment evidence and control summaries](automatic_review.json)",
                "- [Protocol and reproduction](../../docs/AUTOMATIC_ASSESSMENT.md)",
                "",
                "Human listening notes in `listening_review.json` remain unfilled and are not claimed as completed. The earlier forced-alignment diagnostic remains available separately in `objective_audio_evaluation.json`.",
                "",
            ]
        )
        (directory / "README.md").write_text("\n".join(lines), encoding="utf-8")
    print("English and Telugu result indexes updated.")


if __name__ == "__main__":
    main()
