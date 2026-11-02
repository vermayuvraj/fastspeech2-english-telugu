"""Create the concise, editable report from the saved experiment results."""

from pathlib import Path
import json

from docx import Document
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor


ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT.parent.parent / "work"
OUT = ROOT / "Report/Technical_Report_Concise.docx"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


doc = Document()
section = doc.sections[0]
section.page_width, section.page_height = Inches(8.27), Inches(11.69)
section.top_margin = section.bottom_margin = Inches(0.62)
section.left_margin = section.right_margin = Inches(0.7)
section.footer_distance = Inches(0.25)
for name in ["Normal", "Title", "Subtitle", "Heading 1", "Heading 2", "Caption", "List Bullet"]:
    style = doc.styles[name]
    style.font.name = "Calibri"
    style.font.color.rgb = RGBColor(0, 0, 0)
    style.font.size = Pt(10.5)
    style.paragraph_format.space_after = Pt(5)
    style.paragraph_format.line_spacing = 1.04
doc.styles["Title"].font.size = Pt(22)
doc.styles["Title"].font.bold = True
doc.styles["Title"].paragraph_format.space_after = Pt(7)
doc.styles["Heading 1"].font.size = Pt(12)
doc.styles["Heading 1"].font.bold = True
doc.styles["Heading 1"].paragraph_format.space_before = Pt(10)
doc.styles["Heading 1"].paragraph_format.space_after = Pt(5)
doc.styles["Caption"].font.size = Pt(9)
doc.styles["Caption"].font.italic = False
doc.styles["Caption"].font.bold = False
for style in doc.styles:
    for border in style.element.xpath(".//w:pBdr"):
        border.getparent().remove(border)


def paragraph(value, style=None):
    return doc.add_paragraph(value, style)


def bullet(value):
    return paragraph(value, "List Bullet")


def heading(value):
    return doc.add_heading(value, level=1)


def table(rows, widths):
    result = doc.add_table(rows=0, cols=len(rows[0]))
    result.autofit = False
    for col, width in zip(result.columns, widths):
        col.width = Inches(width)
    for i, values in enumerate(rows):
        row = result.add_row()
        trpr = row._tr.get_or_add_trPr()
        no_split = OxmlElement("w:cantSplit")
        trpr.append(no_split)
        if i == 0:
            trpr.append(OxmlElement("w:tblHeader"))
        for j, (cell, value) in enumerate(zip(row.cells, values)):
            cell.width = Inches(widths[j])
            cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
            cell.text = str(value)
            p = cell.paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.LEFT if j == 0 else WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.space_after = Pt(0)
            p.paragraph_format.line_spacing = 1.0
            for run in p.runs:
                run.font.size = Pt(9.5)
                run.font.bold = i == 0
            props = cell._tc.get_or_add_tcPr()
            borders = OxmlElement("w:tcBorders")
            for side in ["top", "left", "bottom", "right"]:
                edge = OxmlElement("w:" + side)
                for key, value in [("val", "single"), ("sz", "4"), ("color", "D9D9D9")]:
                    edge.set(qn("w:" + key), value)
                borders.append(edge)
            props.append(borders)
            margins = OxmlElement("w:tcMar")
            for side, size in [("top", "65"), ("bottom", "65"), ("left", "80"), ("right", "80")]:
                item = OxmlElement("w:" + side)
                item.set(qn("w:w"), size)
                item.set(qn("w:type"), "dxa")
                margins.append(item)
            props.append(margins)
            if i == 0:
                shading = OxmlElement("w:shd")
                shading.set(qn("w:fill"), "E7EDF2")
                props.append(shading)
    paragraph("").paragraph_format.space_after = Pt(0)
    return result


data = {}
for language in ["English", "Telugu"]:
    base = ROOT / "Results" / language
    data[language] = {
        "data": read(base / "data_manifests/data_summary.json"),
        "training": read(base / "training_summary.json"),
        "inference": read(base / "inference_summary.json"),
        "audio": read(base / "objective_audio_evaluation.json"),
    }

doc.add_paragraph("FastSpeech 2 for English and Telugu", "Title")
paragraph("Yuvraj Verma  |  IIIT Hyderabad Voice AI assignment  |  03 November 2026")
paragraph(
    "Two models were trained independently. Both generated the five required utterances, giving ten audio files and thirty duration, pitch and energy plots. Listening review is still pending."
)

heading("Model and implementation")
bullet(
    "Adapted ming024/FastSpeech2 [1, 2]: text encoder, variance adaptor, length regulator, mel decoder and postnet. Each language has separate weights, speaker embeddings and feature statistics."
)
bullet(
    "Encoder and decoder: four blocks each, hidden size 192, two attention heads. Variance predictors use 192 channels and dropout 0.3; output has 80 mel channels. Full settings are in the language config.yaml files."
)
bullet(
    "Both frontends use romanized characters; Telugu uses uroman. Pitch is mean log F0 per token, rather than the paper's wavelet representation. Fixed energy-control routing, device allocation and long-output truncation. Patches and source hashes are in docs/."
)
bullet("A frozen SpeechBrain HiFi-GAN converts predicted mels to 22.05 kHz audio. No vocoder training was performed.")

heading("Data preparation")
rows = [["Dataset and split", "English", "Telugu"]]
rows.append(["Source", "Svarah [3]", "IndicTTS Telugu [4]"])
for split in ["train", "validation"]:
    rows.append(
        ["Training" if split == "train" else "Validation"]
        + [
            f"{data[language]['data']['duration_seconds'][split] / 3600:.4f} h / {data[language]['data']['utterances'][split]:,} clips"
            for language in data
        ]
    )
rows.append(["Voice conditioning", "115 profile groups", "2 supplied voice labels"])
table(rows, [1.8, 2.53, 2.53])
bullet(
    "Removed invalid or silent clips, duplicate audio, clips outside 0.7-20 seconds and transcripts above 450 tokens. Grouped normalized prompts before seeded splitting; audio and prompts do not overlap between splits. Added 15% candidate audio to allow for alignment failures."
)
bullet(
    "Normalized numbers, dates and times; preserved the supplied internal spacing. Converted audio to mono at 22,050 Hz. MMS alignment supplies token durations, including boundary gaps. Svarah profile groups are proxies, not verified speaker IDs."
)

heading("Training setup")
paragraph(
    "Features: FFT/window 1,024, hop 256, 80 mel filters over 0-8,000 Hz. WORLD estimates F0; unvoiced gaps are interpolated. Token energy is the log STFT-magnitude norm. Normalization uses training data only."
)
paragraph(
    "AdamW, peak learning rate 0.0005, 1,000-step warmup, batch sizes 6/4, gradient clipping 1.0, CUDA mixed precision. Validation every 1,000 steps; maximum 40,000. Early stopping applies after 15,000 steps, with patience 12 and minimum improvement 0.002."
)
paragraph(
    "Loss adds mel and postnet L1 to duration, pitch and energy MSE. Validation uses reference variance features; synthesis uses predictions. All 12 saved tests, static checks and compilation passed."
)

doc.add_page_break()
heading("Training results")
rows = [["Model", "Run ended", "Selected checkpoint", "Validation loss"]]
for language, value in data.items():
    rows.append(
        [
            language,
            f"{value['training']['completed_steps']:,}",
            f"{value['inference']['checkpoint_step']:,}",
            f"{value['training']['best_validation_total']:.4f}",
        ]
    )
table(rows, [1.2, 1.5, 2.1, 2.06])
bullet(
    "Training loss continued falling while validation loss flattened in both runs, suggesting overfitting. Both runs stopped early. The final training step was not used for test synthesis."
)
bullet(
    "Telugu reached a slightly lower validation loss at step 23,000 (2.0231), but its improvement over step 14,000 was below 0.002. The saved step-14,000 checkpoint was retained."
)
for language in data:
    heading(language + " loss curves")
    p = paragraph("")
    p.paragraph_format.space_after = Pt(2)
    p.add_run().add_picture(str(WORK / "report_design" / f"{language}_loss_overview.png"), width=Inches(6.65))
    p.runs[-1]._r.xpath(".//wp:docPr")[0].set(
        "descr", f"{language} training and validation total loss across the full run and after step 2000"
    )
    paragraph(
        "Left: complete run. Right: detail after step 2,000. Component curves are saved in Results/" + language + ".",
        "Caption",
    )
paragraph("Loss values cannot rank the languages directly: their data, voices and normalization statistics differ.")

heading("Reproducibility")
paragraph(
    "Dependencies: requirements.txt. Dataset revisions, seeds and hyperparameters: English/config.yaml and Telugu/config.yaml. Checkpoints and optimizer states are saved separately. Split seeds are 20261101 and 20261102. The README gives individual preprocessing, training and inference commands."
)
p = paragraph("python scripts/run_project.py --work-dir PATH --wait-for-access")
for run in p.runs:
    run.font.name = "Consolas"
    run.font.size = Pt(9)

doc.add_page_break()
heading("Test sample results")
rows = [["Sample", "Duration\nseconds", "Predicted F0\nHz", "Predicted energy\nrange", "Alignment\nconfidence"]]
for language, value in data.items():
    measured = {s["id"]: s for s in value["audio"]["generated"]}
    for s in value["inference"]["samples"]:
        rows.append(
            [
                s["id"],
                f"{s['duration_seconds']:.2f}",
                "-".join(f"{v:.0f}" for v in s["continuous_f0_range_hz"]),
                "-".join(f"{v:.1f}" for v in s["energy_range"]),
                f"{measured[s['id']]['transcript_ctc_alignment_confidence']:.3f}",
            ]
        )
table(rows, [0.7, 1.1, 1.4, 1.8, 1.86])
paragraph(
    "F0 is continuous token-level pitch; energy is in relative STFT units. Duration plots use milliseconds per token. CSV/NPZ files retain all traces.",
    "Caption",
)
bullet(
    "English alignment confidence was 0.204-0.346 against a held-out mean of 0.746; E3 scored lowest. Telugu ranged from 0.812-0.980 against 0.946; T4 scored lowest. These uncalibrated scores do not measure pronunciation accuracy or listener quality."
)
bullet(
    "No waveform clipping was found. E4 has two zero-duration tokens and T4 has one; their token traces need checking. All other samples have none."
)

heading("Sentence checks and variance patterns")
paragraph(
    "E1/T1 test rare words; E2a/E2b test noun/verb stress in conduct; T2a/T2b test pattu in two contexts; E3/T3 retain internal spacing; E4/T4 test dates and times, with andaroo also present in E4. Pronunciation and overall speech quality remain unverified for each sample until listening notes are completed."
)
focus = [["Target", "Duration ms", "Mean F0 Hz", "Mean energy"]]
for language in data:
    for s in data[language]["inference"]["samples"]:
        f = s.get("focus_word_measurements")
        if f:
            focus.append(
                [
                    s["id"] + (" conduct" if language == "English" else " pattu"),
                    f"{f['duration_frames'] * 256 / 22050 * 1000:.0f}",
                    f"{f['mean_continuous_f0_hz']:.1f}",
                    f"{f['mean_stft_energy']:.1f}",
                ]
            )
table(focus, [1.8, 1.5, 1.5, 2.06])
paragraph(
    "Conduct has similar total duration in the two contexts; whole-word means do not establish syllable stress. Pattu also has similar duration and mean F0. Its two meanings need not change the segmental pronunciation."
)

heading("Limitations and next steps")
paragraph(
    "Romanization can merge sounds, alignment errors affect duration targets, and continuous F0 does not mark voicing. The fixed vocoder may add artifacts. Complete listening review, inspect syllable-level stress, and resynthesize reference mels to separate acoustic-model and vocoder errors."
)

heading("References")
refs = [
    ("[1] Ren et al. FastSpeech 2. ICLR 2021.", "https://arxiv.org/abs/2006.04558"),
    ("[2] ming024/FastSpeech2. Reference implementation.", "https://github.com/ming024/FastSpeech2"),
    ("[3] AI4Bharat Svarah dataset.", "https://huggingface.co/datasets/ai4bharat/Svarah"),
    ("[4] SPRINGLab IndicTTS Telugu dataset.", "https://huggingface.co/datasets/SPRINGLab/IndicTTS_Telugu"),
    ("Tools: TorchAudio MMS alignment; uroman; SpeechBrain HiFi-GAN.", None),
]
for label, url in refs:
    p = paragraph(label)
    p.paragraph_format.space_after = Pt(1)
    for run in p.runs:
        run.font.size = Pt(9)
    if url:
        link = OxmlElement("w:hyperlink")
        rel = p.part.relate_to(
            url, "http://schemas.openxmlformats.org/officeDocument/2006/relationships/hyperlink", is_external=True
        )
        link.set(qn("r:id"), rel)
        run = OxmlElement("w:r")
        props = OxmlElement("w:rPr")
        size = OxmlElement("w:sz")
        size.set(qn("w:val"), "18")
        props.append(size)
        run.append(props)
        t = OxmlElement("w:t")
        t.text = " [source]"
        run.append(t)
        link.append(run)
        p._p.append(link)

footer = section.footer.paragraphs[0]
footer.alignment = WD_ALIGN_PARAGRAPH.RIGHT
field = OxmlElement("w:fldSimple")
field.set(qn("w:instr"), "PAGE")
footer._p.append(field)
doc.core_properties.title = "FastSpeech 2 for English and Telugu"
doc.core_properties.author = "Yuvraj Verma"
doc.core_properties.subject = "IIIT Hyderabad Voice AI assignment"
doc.save(OUT)
print(OUT)
