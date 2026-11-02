"""Build the report from its editable text and recorded experiment results."""

from datetime import datetime
import json
from pathlib import Path
import re
from xml.sax.saxutils import escape
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import Image, Paragraph, Spacer, Table, TableStyle

ROOT = Path(__file__).resolve().parents[1]
WIDTH = A4[0] - 104
STYLES = {
    "body": ParagraphStyle(
        "body", fontName="Times-Roman", fontSize=10.5, leading=14, alignment=TA_JUSTIFY, spaceAfter=7
    ),
    "title": ParagraphStyle(
        "title", fontName="Times-Bold", fontSize=17, leading=21, alignment=TA_CENTER, spaceAfter=12
    ),
    "author": ParagraphStyle(
        "author", fontName="Times-Roman", fontSize=11, leading=14, alignment=TA_CENTER, spaceAfter=14
    ),
    "heading": ParagraphStyle(
        "heading", fontName="Times-Bold", fontSize=12, leading=15, spaceBefore=11, spaceAfter=6, keepWithNext=True
    ),
    "cell": ParagraphStyle("cell", fontName="Times-Roman", fontSize=9, leading=11.5),
    "reference": ParagraphStyle("reference", fontName="Times-Roman", fontSize=9, leading=11.5, spaceAfter=5),
    "caption": ParagraphStyle(
        "caption", fontName="Times-Roman", fontSize=9, leading=12, alignment=TA_CENTER, spaceAfter=5, keepWithNext=True
    ),
    "code": ParagraphStyle("code", fontName="Courier", fontSize=7.3, leading=11, spaceAfter=7),
}


def read(path, default):
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else default


def results():
    paragraphs = []
    v = read(ROOT / "Report/verification.json", {})
    if v.get("all_passed"):
        pytest_output = " ".join(check.get("stdout", "") for check in v.get("checks", []))
        count_match = re.search(r"(\d+) passed", pytest_output)
        test_count = count_match[1] if count_match else "recorded"
        paragraphs.append(
            f"The saved verification run passed all {test_count} tests, static checks and Python compilation. Tests cover text processing, data splits, duplicate removal, duration targets and model operations. A full forward and backward pass also completed on the local RTX 4050 GPU. These checks verify execution, not speech quality."
        )
    observed = False
    for language in ["English", "Telugu"]:
        path = ROOT / "Results" / language / "losses.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines() if line] if path.is_file() else []
        validation = [r for r in rows if r["split"] == "validation"]
        if validation:
            observed = True
            best = min(validation, key=lambda r: r["total"])
            paragraphs.append(
                f"{language}: validation loss fell from {validation[0]['total']:.4f} at the first check to {validation[-1]['total']:.4f} at the final check. Its lowest value was {best['total']:.4f} at step {best['step']:,}."
            )
            summary = read(ROOT / "Results" / language / "training_summary.json", {})
            data = read(ROOT / "Results" / language / "data_manifests/data_summary.json", {})
            selected_step = read(ROOT / "Results" / language / "inference_summary.json", {}).get(
                "checkpoint_step", best["step"]
            )
            if data:
                hours = data["duration_seconds"]
                paragraphs.append(
                    f"The accepted {language} subset contains {hours['train'] / 3600:.4f} training hours and {hours['validation'] / 3600:.4f} validation hours. Training ended at step {summary.get('completed_steps'):,}"
                    + (
                        " after 12 validation checks without enough improvement."
                        if summary.get("stopped_early")
                        else "."
                    )
                    + f" Test audio uses the checkpoint from step {selected_step:,}. A saved checkpoint is replaced only when validation loss improves by at least 0.002."
                )
            paragraphs.append(f"{{{{CURVES:{language}}}}}")
        reviews = read(ROOT / "Results" / language / "listening_review.json", {})
        for sample in read(ROOT / "Results" / language / "inference_summary.json", {}).get("samples", []):
            observed = True
            low, high = sample["continuous_f0_range_hz"]
            text = f"{sample['id']}: generated duration {sample['duration_seconds']:.2f} s; predicted continuous F0 range {low:.0f}-{high:.0f} Hz; {sample['zero_duration_tokens']} zero-duration tokens."
            review = reviews.get(sample["id"], {})
            if all(
                str(review.get(k, "")).strip().lower() not in {"", "pending"}
                for k in ["pronunciation", "overall_quality", "specific_observations"]
            ):
                text += " " + " ".join(
                    str(review[k]) for k in ["pronunciation", "overall_quality", "specific_observations"]
                )
            paragraphs.append(text)
        audio = read(ROOT / "Results" / language / "objective_audio_evaluation.json", {})
        if audio:
            baseline = [
                r["transcript_ctc_alignment_confidence"]
                for r in audio["baseline"]
                if "transcript_ctc_alignment_confidence" in r
            ]
            paragraphs.append(
                f"The mean MMS alignment confidence for ten held-out {language} recordings was {sum(baseline) / len(baseline):.3f}. The generated samples were checked with the same aligner. This uncalibrated score measures transcript alignment; it is not pronunciation accuracy or a listener rating."
            )
            lines = ["| Sample | Waveform measurements |"]
            for row in audio["generated"]:
                lines.append(
                    f"| {row['id']} | Duration {row['duration_seconds']:.2f} s; voiced F0 median {row.get('measured_voiced_f0_median_hz', 0):.1f} Hz; alignment confidence {row.get('transcript_ctc_alignment_confidence', 0):.3f}; clipped samples {row['clipped_fraction']:.1%} |"
                )
            paragraphs.append("\n".join(lines))
            least = min(audio["generated"], key=lambda r: r.get("transcript_ctc_alignment_confidence", 0))
            paragraphs.append(
                f"{least['id']} had the lowest alignment confidence among the generated {language} samples. It needs a closer listening check; the score alone does not identify a pronunciation error."
            )
    if not observed:
        paragraphs.append(
            "No full training curves, final acoustic checkpoints or assignment-sentence synthesis results are available in the current experiment record. Convergence, pronunciation accuracy and comparative speech quality therefore remain unevaluated. The software checks establish that the tested components execute correctly; they do not establish the performance of a trained TTS system."
        )
    return "\n\n".join(paragraphs)


def footer(canvas, doc):
    canvas.setFont("Times-Roman", 9)
    canvas.drawCentredString(A4[0] / 2, 29, str(doc.page))


def main():
    complete = all(
        (ROOT / "Results" / lang / "training_summary.json").is_file()
        and (ROOT / "Results" / lang / "inference_summary.json").is_file()
        for lang in ["English", "Telugu"]
    )
    source = (ROOT / "Report/report_text.md").read_text(encoding="utf-8")
    available_languages = [
        lang
        for lang in ["English", "Telugu"]
        if (ROOT / "Results" / lang / "training_summary.json").is_file()
        and (ROOT / "Results" / lang / "inference_summary.json").is_file()
    ]
    status = (
        "Available training and synthesis measurements are presented below."
        if complete
        else f"The implementation has been checked. Training and test synthesis are complete for {', '.join(available_languages)}; the remaining language experiment is incomplete."
        if available_languages
        else "The preprocessing, training and inference routines have been implemented and checked. Full corpus training and test synthesis have not yet been completed; this report therefore covers the implementation and its verification."
    )
    source = source.replace("{{EXPERIMENT_STATUS}}", status).replace("{{RESULTS}}", results())
    story = []
    for block in source.strip().split("\n\n"):
        if block.startswith("{{CURVES:"):
            language = block[len("{{CURVES:") : -2]
            paths = [ROOT / "Results" / language / name for name in ["training_loss.png", "validation_loss.png"]]
            if all(path.is_file() for path in paths):
                story.append(Paragraph(f"{language}: training and validation objectives.", STYLES["caption"]))
                story.append(
                    Table(
                        [[Image(str(path), width=WIDTH / 2 - 4, height=(WIDTH / 2 - 4) / 1.875) for path in paths]],
                        colWidths=[WIDTH / 2, WIDTH / 2],
                    )
                )
            continue
        if block.startswith("|"):
            rows = [[cell.strip() for cell in line.strip().strip("|").split("|")] for line in block.splitlines()]
            cells = [
                [Paragraph(("<b>" + escape(c) + "</b>") if i == 0 else escape(c), STYLES["cell"]) for c in row]
                for i, row in enumerate(rows)
            ]
            table = Table(cells, colWidths=[WIDTH * 0.34, WIDTH * 0.66], repeatRows=1, hAlign="CENTER", splitByRow=0)
            table.setStyle(
                TableStyle(
                    [
                        ("LINEABOVE", (0, 0), (-1, 0), 0.7, colors.black),
                        ("LINEBELOW", (0, 0), (-1, 0), 0.4, colors.black),
                        ("LINEBELOW", (0, -1), (-1, -1), 0.7, colors.black),
                        ("VALIGN", (0, 0), (-1, -1), "TOP"),
                        ("TOPPADDING", (0, 0), (-1, -1), 5),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                    ]
                )
            )
            story.extend([table, Spacer(1, 7)])
            continue
        style = "body"
        if block.startswith("# "):
            style = "title"
            block = block[2:]
        elif block.startswith("## "):
            style = "heading"
            block = block[3:]
        elif block == "Yuvraj Verma":
            style = "author"
        elif block.startswith("Table "):
            style = "caption"
        elif block.startswith("["):
            style = "reference"
        elif block.startswith("`"):
            style = "code"
            block = block.strip("`")
        text = escape(block)
        text = re.sub(r"https?://[^\s]+", lambda m: '<link href="' + m[0] + '"><u>' + m[0] + "</u></link>", text)
        story.append(Paragraph(text, STYLES[style]))
    output = ROOT / "Report"
    from report_design import build_report

    build_report()
    (output / "Technical_Report.md").write_text(source, encoding="utf-8")
    (output / "report_status.json").write_text(
        json.dumps(
            {
                "experiments_available": complete,
                "style": "clean academic layout",
                "updated_at": datetime.now().isoformat(),
            },
            indent=2,
        )
        + "\n"
    )
    print(output / "Technical_Report.pdf")


if __name__ == "__main__":
    main()
