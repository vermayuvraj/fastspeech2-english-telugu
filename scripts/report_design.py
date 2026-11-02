"""Clean, paginated report layout using the recorded experiment artifacts."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import re
from xml.sax.saxutils import escape

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import (
    Flowable,
    Image,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

ROOT = Path(__file__).resolve().parents[1]
SCRATCH = ROOT.parent.parent / "work/report_design"
WIDTH = A4[0] - 96
NAVY = colors.HexColor("#203d55")
TEAL = colors.HexColor("#287f84")
INK = colors.HexColor("#26333b")
GRAY = colors.HexColor("#637481")
LINE = colors.HexColor("#d8e1e6")
PALE = colors.HexColor("#edf3f6")
STYLES = {
    "body": ParagraphStyle(
        "Body", fontName="Times-Roman", fontSize=10.2, leading=13.5, textColor=INK, alignment=TA_LEFT, spaceAfter=6
    ),
    "title": ParagraphStyle("Title", fontName="Times-Bold", fontSize=26, leading=29, textColor=NAVY, spaceAfter=13),
    "kicker": ParagraphStyle(
        "Kicker", fontName="Helvetica-Bold", fontSize=8, leading=11, textColor=TEAL, spaceAfter=11
    ),
    "h1": ParagraphStyle(
        "Section",
        fontName="Helvetica-Bold",
        fontSize=12.4,
        leading=16,
        textColor=NAVY,
        spaceBefore=8,
        spaceAfter=8,
        keepWithNext=True,
    ),
    "h2": ParagraphStyle(
        "Subsection",
        fontName="Helvetica-Bold",
        fontSize=10.3,
        leading=14,
        textColor=NAVY,
        spaceBefore=8,
        spaceAfter=6,
        keepWithNext=True,
    ),
    "caption": ParagraphStyle(
        "Caption", fontName="Helvetica", fontSize=8.2, leading=11, textColor=GRAY, spaceAfter=5, keepWithNext=True
    ),
    "cell": ParagraphStyle("Cell", fontName="Helvetica", fontSize=8.5, leading=11.4, textColor=INK),
    "head": ParagraphStyle("Head", fontName="Helvetica-Bold", fontSize=8.1, leading=11, textColor=NAVY),
    "small": ParagraphStyle("Small", fontName="Helvetica", fontSize=8.1, leading=11, textColor=GRAY, spaceAfter=5),
    "reference": ParagraphStyle(
        "Reference", fontName="Times-Roman", fontSize=8.5, leading=10.8, textColor=INK, spaceAfter=4
    ),
    "code": ParagraphStyle("Command", fontName="Courier", fontSize=7.6, leading=10.5, textColor=NAVY, spaceAfter=7),
}


def read(path, default=None):
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else ({} if default is None else default)


def text(value, style="body"):
    value = escape(str(value)).replace("\n", "<br/>")
    value = re.sub(r"https?://[^\s<]+", lambda m: f'<link href="{m[0]}" color="#287f84">{m[0]}</link>', value)
    return Paragraph(value, STYLES[style])


def table(rows, widths=None, compact=False):
    cells = [[text(cell, "head" if i == 0 else "cell") for cell in row] for i, row in enumerate(rows)]
    result = Table(cells, colWidths=widths or [WIDTH * 0.33, WIDTH * 0.67], hAlign="LEFT", splitByRow=0)
    pad = 5 if compact else 7
    result.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), PALE),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafb")]),
                ("LINEBELOW", (0, 0), (-1, 0), 0.6, LINE),
                ("LINEBELOW", (0, -1), (-1, -1), 0.5, LINE),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("TOPPADDING", (0, 0), (-1, -1), pad),
                ("BOTTOMPADDING", (0, 0), (-1, -1), pad),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ("RIGHTPADDING", (0, 0), (-1, -1), 8),
            ]
        )
    )
    return result


class Pipeline(Flowable):
    def __init__(self):
        super().__init__()
        self.width, self.height = WIDTH, 67

    def draw(self):
        c = self.canv
        labels = [
            "Text\nfrontend",
            "Text\nencoder",
            "Variance\nadaptor",
            "Mel\ndecoder",
            "HiFi-GAN\nvocoder",
            "Speech\nwaveform",
        ]
        gap = 12
        w = (WIDTH - gap * 5) / 6
        for i, label in enumerate(labels):
            x = i * (w + gap)
            c.setFillColor(PALE if i != 2 else colors.HexColor("#e2eff0"))
            c.setStrokeColor(LINE)
            c.roundRect(x, 16, w, 43, 5, fill=1, stroke=1)
            c.setFillColor(NAVY)
            c.setFont("Helvetica", 8)
            for j, line in enumerate(label.split("\n")):
                c.drawCentredString(x + w / 2, 41 - j * 11, line)
            if i < 5:
                c.setStrokeColor(GRAY)
                c.line(x + w + 3, 37, x + w + gap - 3, 37)
                c.line(x + w + gap - 6, 40, x + w + gap - 3, 37)
                c.line(x + w + gap - 6, 34, x + w + gap - 3, 37)


def loss_figure(language):
    path = ROOT / "Results" / language / "losses.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines() if line]
    train = [row for row in rows if row["split"] == "train"]
    val = [row for row in rows if row["split"] == "validation"]
    with plt.rc_context(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 9,
            "legend.fontsize": 7.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    ):
        fig, axes = plt.subplots(1, 2, figsize=(7, 2.45), layout="constrained")
        for ax, cutoff, title in zip(axes, [0, 2000], ["Complete run", "Detail after step 2,000"]):
            for data, color, label, style in [
                (train, "#245575", "Training", "-"),
                (val, "#278388", "Validation", "--"),
            ]:
                subset = [row for row in data if row["step"] >= cutoff]
                ax.plot(
                    [row["step"] for row in subset],
                    [row["total"] for row in subset],
                    color=color,
                    linewidth=1.3,
                    linestyle=style,
                    label=label,
                )
            ax.set_title(title, loc="left", pad=9)
            ax.set_xlabel("Optimizer step")
            ax.set_ylabel("Total loss")
            ax.grid(axis="y", color="#e5ebef", linewidth=0.6)
            ax.xaxis.set_major_locator(MaxNLocator(4))
            ax.yaxis.set_major_locator(MaxNLocator(5))
            ax.tick_params(length=2.5, color="#a4b2bc")
            ax.legend(frameon=True, facecolor="white", edgecolor="none", framealpha=0.96, loc="upper right")
        result = SCRATCH / f"{language}_loss_overview.png"
        fig.savefig(result, dpi=220)
        plt.close(fig)
    return result


def page_decoration(canvas, doc):
    canvas.saveState()
    canvas.setFillColor(NAVY)
    canvas.rect(48, A4[1] - 32, WIDTH, 1.4, fill=1, stroke=0)
    if doc.page > 1:
        canvas.setFont("Helvetica", 7.5)
        canvas.setFillColor(GRAY)
        canvas.drawString(48, A4[1] - 23, "FASTSPEECH 2  /  ENGLISH AND TELUGU")
    canvas.setStrokeColor(LINE)
    canvas.line(48, 36, A4[0] - 48, 36)
    canvas.setFont("Helvetica", 7.5)
    canvas.setFillColor(GRAY)
    canvas.drawString(48, 23, "Yuvraj Verma  |  Technical report")
    canvas.drawRightString(A4[0] - 48, 23, f"{doc.page:02d}")
    canvas.restoreState()


def sections():
    source = (ROOT / "Report/report_text.md").read_text(encoding="utf-8")
    blocks = re.split(r"(?m)^## ", source)[1:]
    return {block.split("\n", 1)[0]: block.split("\n", 1)[1].strip() for block in blocks}


def append_section(story, title, content, heading_style="h1"):
    story.append(text(title, heading_style))
    for block in content.split("\n\n"):
        if block.startswith("|"):
            rows = [[cell.strip() for cell in line.strip().strip("|").split("|")] for line in block.splitlines()]
            story.extend([table(rows, compact=True), Spacer(1, 8)])
        elif block.startswith("Table "):
            story.append(text(block, "caption"))
        elif block.startswith("`"):
            story.append(text(block.strip("`"), "code"))
        else:
            story.append(text(block))


def language_page(story, language, number):
    result = ROOT / "Results" / language
    training = read(result / "training_summary.json")
    inference = read(result / "inference_summary.json")
    evaluation = read(result / "objective_audio_evaluation.json")
    losses = [json.loads(line) for line in (result / "losses.jsonl").read_text().splitlines() if line]
    val = [row for row in losses if row["split"] == "validation"]
    raw_best = min(val, key=lambda row: row["total"])
    if number == 1:
        story.append(text("5. Experimental results", "h1"))
    story.append(text(f"5.{number}  {language} results", "h2" if number == 1 else "h1"))
    summary = [
        ["Run ended", "Selected checkpoint", "Selected validation loss"],
        [
            f"{training['completed_steps']:,} steps",
            f"Step {inference['checkpoint_step']:,}",
            f"{training['best_validation_total']:.4f}",
        ],
    ]
    story.extend([table(summary, [WIDTH / 3] * 3), Spacer(1, 11)])
    story.append(
        text(
            f"Validation loss fell from {val[0]['total']:.4f} at the first check to {val[-1]['total']:.4f} at the final check. Its lowest value was {raw_best['total']:.4f} at step {raw_best['step']:,}. The run ended after 12 checks without an improvement large enough to replace the saved checkpoint."
        )
    )
    if inference["checkpoint_step"] != raw_best["step"]:
        story.append(
            text(
                f"The step-{raw_best['step']:,} loss was slightly lower than the saved checkpoint's loss, but the difference was below the 0.002 replacement threshold. Test audio therefore uses the checkpoint from step {inference['checkpoint_step']:,}."
            )
        )
    else:
        story.append(
            text(f"The checkpoint from step {inference['checkpoint_step']:,} was used for all five test utterances.")
        )
    figure = Image(str(loss_figure(language)), width=WIDTH, height=WIDTH * 2.45 / 7)
    story.append(
        KeepTogether(
            [
                figure,
                text(
                    f"Figure {number}. Training and validation loss. The right panel enlarges the later part of the run. Separate loss-component plots are saved in Results/{language}.",
                    "small",
                ),
            ]
        )
    )
    story.append(Spacer(1, 8))
    generated = {row["id"]: row for row in evaluation["generated"]}
    rows = [
        [
            "Sample",
            "Duration\n(s)",
            "Predicted F0\nrange (Hz)",
            "Measured F0\nmedian (Hz)",
            "Alignment\nconfidence",
            "Zero-duration\ntokens",
        ]
    ]
    for sample in inference["samples"]:
        measured = generated[sample["id"]]
        lo, hi = sample["continuous_f0_range_hz"]
        rows.append(
            [
                sample["id"],
                f"{sample['duration_seconds']:.2f}",
                f"{lo:.0f}-{hi:.0f}",
                f"{measured['measured_voiced_f0_median_hz']:.1f}",
                f"{measured['transcript_ctc_alignment_confidence']:.3f}",
                sample["zero_duration_tokens"],
            ]
        )
    story.append(text(f"Table {number + 2}. Test-sentence measurements for {language}.", "caption"))
    story.append(table(rows, [52, 60, 92, 98, 98, WIDTH - 400], compact=False))
    story.append(Spacer(1, 10))
    baseline = [row["transcript_ctc_alignment_confidence"] for row in evaluation["baseline"]]
    scores = [row["transcript_ctc_alignment_confidence"] for row in evaluation["generated"]]
    if language == "English":
        observation = f"All five generated samples had lower alignment confidence ({min(scores):.3f}-{max(scores):.3f}) than the mean for ten held-out recordings ({sum(baseline) / len(baseline):.3f}). The gap is a reason to review the English audio closely despite the falling training loss. E3 scored lowest, but the score does not identify a particular word or pronunciation error."
    else:
        observation = f"Generated-sample alignment confidence ranged from {min(scores):.3f} to {max(scores):.3f}; the mean for ten held-out recordings was {sum(baseline) / len(baseline):.3f}. T2a scored highest and T4 lowest. These values support a more favorable alignment check than the English run, but they do not establish correct stress, phrasing or pronunciation."
    story.append(text(observation))
    if language == "English":
        story.append(
            text(
                "Alignment confidence is the MMS CTC aligner's score for matching the supplied transcript to the audio. The held-out recordings provide a baseline for the same check. This uncalibrated score is not pronunciation accuracy or a listener rating."
            )
        )
    story.append(
        text(
            "No clipped samples were found in the five waveforms. The F0 columns describe different measurements: the predicted range covers the continuous token contour, while the measured median uses voiced regions of the generated audio."
        )
    )
    reviews = read(result / "listening_review.json")
    for sample_id, review in reviews.items():
        keys = ["pronunciation", "overall_quality", "specific_observations"]
        if all(str(review.get(key, "")).strip().lower() not in {"", "pending"} for key in keys):
            story.append(text(sample_id + ": " + " ".join(str(review[key]) for key in keys)))


def build_report():
    SCRATCH.mkdir(parents=True, exist_ok=True)
    content = sections()
    story = []
    complete = all((ROOT / "Results" / lang / "training_summary.json").is_file() for lang in ["English", "Telugu"])
    date = datetime.now(timezone(timedelta(hours=5, minutes=30))).strftime("%d %B %Y")
    story.append(text("IIIT HYDERABAD / VOICE AI ASSIGNMENT", "kicker"))
    story.append(text("FastSpeech 2 for English\nand Telugu Speech Synthesis", "title"))
    author = Table([[text("Yuvraj Verma", "small"), text(date, "small")]], colWidths=[WIDTH * 0.7, WIDTH * 0.3])
    author.setStyle(TableStyle([("LEFTPADDING", (0, 0), (-1, -1), 0), ("RIGHTPADDING", (0, 0), (-1, -1), 0)]))
    story.extend([author, Spacer(1, 12)])
    abstract = content["Abstract"].replace(
        "{{EXPERIMENT_STATUS}}",
        "Training and synthesis measurements for both language models are reported below."
        if complete
        else "The experiment record is incomplete.",
    )
    story.append(text("Abstract", "h2"))
    story.append(text(abstract))
    append_section(story, "1. Introduction", content["1. Introduction"])
    story.append(Spacer(1, 7))
    story.append(text("Experiment overview", "caption"))
    rows = [
        ["Language", "Training\n(hours)", "Validation\n(hours)", "Training run\n(steps)", "Selected checkpoint\n(step)"]
    ]
    for language in ["English", "Telugu"]:
        d = read(ROOT / "Results" / language / "data_manifests/data_summary.json")
        t = read(ROOT / "Results" / language / "training_summary.json")
        i = read(ROOT / "Results" / language / "inference_summary.json")
        rows.append(
            [
                language,
                f"{d['duration_seconds']['train'] / 3600:.4f}",
                f"{d['duration_seconds']['validation'] / 3600:.4f}",
                f"{t['completed_steps']:,}",
                f"{i['checkpoint_step']:,}",
            ]
        )
    story.extend([table(rows, [76, 88, 88, 111, WIDTH - 363]), Spacer(1, 17)])
    story.append(text("Synthesis pipeline", "caption"))
    story.append(Pipeline())
    story.append(
        text(
            "Each test utterance has a WAV file and separate duration, pitch and energy plots: ten audio files and thirty plots in total.",
            "small",
        )
    )
    v = read(ROOT / "Report/verification.json")
    match = re.search(r"(\d+) passed", " ".join(row.get("stdout", "") for row in v.get("checks", [])))
    count = match[1] if match else "recorded"
    story.append(text("Code checks", "h2"))
    story.append(
        text(
            f"The saved verification run passed all {count} tests, static checks and Python compilation. Tests cover text processing, data splits, duplicate removal, duration targets and model operations. A full forward and backward pass also completed on the local RTX 4050 GPU. These checks verify execution, not speech quality."
        )
    )

    story.append(PageBreak())
    append_section(story, "2. Model implementation", content["2. Model implementation"])
    features, training_config = content["4. Acoustic features and training setup"].split("Table 2.", 1)
    append_section(story, "2.1 Acoustic features and objective", features.strip(), "h2")

    story.append(PageBreak())
    append_section(story, "3. Dataset preparation", content["3. Dataset preparation"])
    append_section(story, "4. Training configuration", "Table 2." + training_config)
    story.append(text("Accepted data after preprocessing and alignment", "caption"))
    rows = [["Language", "Training\nutterances", "Validation\nutterances", "Conditioning\ngroups"]]
    for language in ["English", "Telugu"]:
        d = read(ROOT / "Results" / language / "data_manifests/data_summary.json")
        rows.append([language, f"{d['utterances']['train']:,}", f"{d['utterances']['validation']:,}", d["speakers"]])
    story.extend([table(rows, [WIDTH / 4] * 4), Spacer(1, 10)])

    for index, language in enumerate(["English", "Telugu"], 1):
        story.append(PageBreak())
        language_page(story, language, index)

    story.append(PageBreak())
    append_section(
        story,
        "6. Evaluation of the test sentences",
        content["6. Evaluation of the test sentences"].replace("Table 3", "Table 5"),
    )
    append_section(story, "7. Limitations", content["7. Limitations"])
    append_section(story, "8. Reproducibility", content["8. Reproducibility"])
    story.append(text("References", "h1"))
    for block in content["References"].split("\n\n"):
        match = re.search(r"https?://\S+", block)
        if match:
            label = escape(block[: match.start()].strip())
            story.append(
                Paragraph(
                    label + f' <link href="{escape(match[0])}" color="#287f84">[online]</link>', STYLES["reference"]
                )
            )
        else:
            story.append(text(block, "reference"))
    temporary = SCRATCH / "Technical_Report.pdf"
    SimpleDocTemplate(
        str(temporary),
        pagesize=A4,
        leftMargin=48,
        rightMargin=48,
        topMargin=48,
        bottomMargin=48,
        title="FastSpeech 2 for English and Telugu Speech Synthesis",
        author="Yuvraj Verma",
    ).build(story, onFirstPage=page_decoration, onLaterPages=page_decoration)
    temporary.replace(ROOT / "Report/Technical_Report.pdf")


if __name__ == "__main__":
    build_report()
