"""Build the concise report from completed automatic assessment evidence.

Run with the document runtime's python-docx, then render with render_docx.py.
The output is staged for visual review; this script does not publish or submit.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

from docx import Document
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


class Report:
    def __init__(self):
        self.doc = Document()
        section = self.doc.sections[0]
        section.page_width, section.page_height = Inches(8.27), Inches(11.69)
        section.left_margin = section.right_margin = Inches(0.7)
        section.top_margin = section.bottom_margin = Inches(0.6)
        section.footer_distance = Inches(0.25)
        for name in ["Normal", "Title", "Heading 1", "Caption", "List Bullet"]:
            style = self.doc.styles[name]
            style.font.name = "Calibri"
            style.font.size = Pt(10.5)
            style.font.color.rgb = RGBColor(0, 0, 0)
            style.paragraph_format.line_spacing = 1.03
            style.paragraph_format.space_after = Pt(5)
        self.doc.styles["Title"].font.size = Pt(22)
        self.doc.styles["Title"].font.bold = True
        self.doc.styles["Heading 1"].font.size = Pt(12)
        self.doc.styles["Heading 1"].font.bold = True
        self.doc.styles["Heading 1"].paragraph_format.space_before = Pt(9)
        self.doc.styles["Caption"].font.size = Pt(9)
        self.doc.styles["Caption"].font.bold = False
        self.doc.styles["Caption"].font.italic = False
        for style in self.doc.styles:
            for border in style.element.xpath(".//w:pBdr"):
                border.getparent().remove(border)
        footer = section.footer.paragraphs[0]
        footer.alignment = WD_ALIGN_PARAGRAPH.RIGHT
        field = OxmlElement("w:fldSimple")
        field.set(qn("w:instr"), "PAGE")
        footer._p.append(field)

    def paragraph(self, text, style=None):
        return self.doc.add_paragraph(text, style)

    def heading(self, text):
        return self.doc.add_heading(text, level=1)

    def bullet(self, text):
        return self.paragraph(text, "List Bullet")

    def page(self):
        self.doc.add_page_break()

    def table(self, rows, widths, size=9.5):
        table = self.doc.add_table(rows=0, cols=len(rows[0]))
        table.autofit = False
        for column, width in zip(table.columns, widths):
            column.width = Inches(width)
        for i, values in enumerate(rows):
            row = table.add_row()
            props = row._tr.get_or_add_trPr()
            props.append(OxmlElement("w:cantSplit"))
            if not i:
                props.append(OxmlElement("w:tblHeader"))
            for j, (cell, value) in enumerate(zip(row.cells, values)):
                cell.width = Inches(widths[j])
                cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
                cell.text = str(value)
                p = cell.paragraphs[0]
                p.alignment = WD_ALIGN_PARAGRAPH.LEFT if j == 0 else WD_ALIGN_PARAGRAPH.CENTER
                p.paragraph_format.space_after = Pt(0)
                p.paragraph_format.line_spacing = 1
                for run in p.runs:
                    run.font.size = Pt(size)
                    run.font.bold = i == 0
                tc = cell._tc.get_or_add_tcPr()
                borders = OxmlElement("w:tcBorders")
                for side in ["top", "left", "bottom", "right"]:
                    edge = OxmlElement("w:" + side)
                    for key, val in [("val", "single"), ("sz", "4"), ("color", "D9D9D9")]:
                        edge.set(qn("w:" + key), val)
                    borders.append(edge)
                tc.append(borders)
                margins = OxmlElement("w:tcMar")
                for side, val in [("top", "60"), ("bottom", "60"), ("left", "80"), ("right", "80")]:
                    edge = OxmlElement("w:" + side)
                    edge.set(qn("w:w"), val)
                    edge.set(qn("w:type"), "dxa")
                    margins.append(edge)
                tc.append(margins)
                if not i:
                    shade = OxmlElement("w:shd")
                    shade.set(qn("w:fill"), "E7EDF2")
                    tc.append(shade)
        gap = self.paragraph("")
        gap.paragraph_format.space_after = Pt(0)
        gap.paragraph_format.line_spacing = 0.25

    def picture(self, path, description, width=6.65):
        p = self.paragraph("")
        p.paragraph_format.space_after = Pt(2)
        shape = p.add_run().add_picture(str(path), width=Inches(width))
        shape._inline.docPr.set("descr", description)

    def reference(self, label, url):
        p = self.paragraph(label)
        p.paragraph_format.space_after = Pt(2)
        for run in p.runs:
            run.font.size = Pt(9)
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
        text = OxmlElement("w:t")
        text.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
        text.text = " [source]"
        run.append(text)
        link.append(run)
        p._p.append(link)


def compact_observations(acoustic):
    values = {}
    for language, data in acoustic.items():
        for s in data["samples"]:
            name = s["id"]
            words = {w["word"]: w for w in s["words"]}
            if name in {"E1", "T1"}:
                target = words["antidisestablishmentarianism"] if name == "E1" else s["words"][0]
                values[name] = (
                    f"The target word takes {target['duration_ms']:.0f} ms; its mean F0 is {target['mean_predicted_f0_hz']:.1f} Hz and energy {target['mean_predicted_stft_energy']:.1f}."
                )
            elif name.startswith("E2"):
                a, b = s["conduct_grapheme_regions"]
                values[name] = (
                    f"Conduct takes {words['conduct']['duration_ms']:.0f} ms. Con/duct mean F0 is {a['mean_predicted_f0_hz']:.1f}/{b['mean_predicted_f0_hz']:.1f} Hz; energy is {a['mean_predicted_energy']:.1f}/{b['mean_predicted_energy']:.1f}."
                )
            elif name.startswith("T2"):
                w = words["pattttu"]
                values[name] = (
                    f"Pattu takes {w['duration_ms']:.0f} ms, with mean F0 {w['mean_predicted_f0_hz']:.1f} Hz and energy {w['mean_predicted_stft_energy']:.1f}."
                )
            elif name in {"E3", "T3"}:
                left, right = ("rhyth", "m") if name == "E3" else ("adda", "viloo")
                a, b = words[left], words[right]
                gap = next(x["duration_ms"] for x in s["boundaries"] if x["token_index"] == a["token_indices"][-1] + 1)
                values[name] = (
                    f"The boundary at the supplied internal split lasts {gap:.0f} ms. Fragment mean F0 is {a['mean_predicted_f0_hz']:.0f}/{b['mean_predicted_f0_hz']:.0f} Hz and energy {a['mean_predicted_stft_energy']:.1f}/{b['mean_predicted_stft_energy']:.1f}."
                )
            elif name == "E4":
                a, b = words["a"], words["m"]
                w = words["andaroo"]
                values[name] = (
                    f"The a/m tokens each receive {a['duration_ms']:.0f} ms. Andaroo has mean F0 {w['mean_predicted_f0_hz']:.1f} Hz and energy {w['mean_predicted_stft_energy']:.1f}. Both zero-duration tokens are boundaries."
                )
            else:
                a, b = s["words"][0], s["words"][-1]
                gap = next(
                    x["duration_ms"]
                    for x in s["boundaries"]
                    if x["token_index"] == words["nalabhai"]["token_indices"][-1] + 1
                )
                values[name] = (
                    f"The forty/five boundary is {gap:.0f} ms. Mean word F0 falls {a['mean_predicted_f0_hz']:.0f} to {b['mean_predicted_f0_hz']:.0f} Hz and energy {a['mean_predicted_stft_energy']:.1f} to {b['mean_predicted_stft_energy']:.1f}. The zero-duration token is a boundary."
                )
    return values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    protocol = read(ROOT / "Report/automatic_assessment_protocol.json")
    assert protocol["human_listening_performed"] is False
    data = {}
    for language in ["English", "Telugu"]:
        directory = ROOT / "Results" / language
        data[language] = {
            "data": read(directory / "data_manifests/data_summary.json"),
            "training": read(directory / "training_summary.json"),
            "inference": read(directory / "inference_summary.json"),
            "review": read(directory / "automatic_review.json"),
            "acoustics": read(directory / "automatic_acoustic_analysis.json"),
        }
        assert data[language]["review"]["generated"]["count"] == 5
        assert data[language]["review"]["controls_mms"]["count"] == 10
    r = Report()
    r.paragraph("FastSpeech 2 for English and Telugu", "Title")
    r.paragraph("IIIT Hyderabad Voice AI assignment  |  Submitted by Yuvraj Verma")
    r.paragraph(
        "Two models were trained independently on about five hours of audio per language. All ten test utterances were generated, with thirty duration, pitch and energy plots. This report adds automatic assessment of every sample and comparison with held-out recordings."
    )
    r.heading("Assessment method")
    r.paragraph(protocol["author_statement"])
    r.paragraph(
        "Whisper large-v3 and MMS-1b-all independently transcribed the audio without the reference sentences as prompts. Word and character errors were measured against the expanded spoken text. UTMOSv2 predicted quality using three crop repeats. Ten held-out recordings per language provide descriptive controls. These controls are not matched to the test sentences by speaker or length."
    )
    r.heading("Model and implementation")
    r.bullet(
        "Adapted ming024/FastSpeech2: text encoder, variance adaptor, length regulator, mel decoder and postnet. Encoder/decoder use four blocks each, hidden size 192 and two attention heads. Each language has separate weights, speaker embeddings and statistics."
    )
    r.bullet(
        "Both models use romanized characters; Telugu uses uroman. Pitch is predicted as mean log F0 per token. Fixed energy-control routing, device allocation and long-output truncation. A frozen SpeechBrain HiFi-GAN generates 22.05 kHz waveforms."
    )
    r.heading("Data preparation")
    rows = [["Data", "English", "Telugu"], ["Source", "Svarah", "IndicTTS Telugu"]]
    for split, label in [("train", "Training"), ("validation", "Validation")]:
        rows.append(
            [label]
            + [
                f"{data[language]['data']['duration_seconds'][split] / 3600:.4f} h / {data[language]['data']['utterances'][split]:,} clips"
                for language in data
            ]
        )
    rows.append(["Voice conditioning", "115 profile groups", "2 voice labels"])
    r.table(rows, [1.7, 2.5, 2.5])
    r.paragraph(
        "Removed invalid/silent clips and duplicate audio; retained 0.7-20 s recordings with at most 450 tokens. Grouped normalized prompts before seeded splitting, with no audio or prompt overlap. Added 15% candidate audio for alignment failures. Svarah profile groups are proxies, not verified speaker IDs."
    )
    r.heading("Features and training")
    r.paragraph(
        "NFC normalization expands numbers, dates and times while retaining the supplied internal spacing. MMS alignment supplies token durations. Features use FFT/window 1,024, hop 256 and 80 mel filters over 0-8,000 Hz. WORLD estimates F0; unvoiced gaps are interpolated. Energy is the log STFT-magnitude norm. Statistics use training data only."
    )
    r.paragraph(
        "AdamW; peak learning rate 0.0005; 1,000-step warmup; batch sizes 6/4; gradient clipping 1.0; CUDA mixed precision. Validation runs every 1,000 steps, with a 40,000-step limit. Early stopping starts after 15,000 steps, with patience 12 and minimum improvement 0.002. Loss adds mel/postnet L1 to duration, pitch and energy MSE."
    )

    r.page()
    r.heading("Training and automatic assessment results")
    rows = [["Model", "Run ended", "Selected checkpoint", "Validation loss"]]
    for language, v in data.items():
        rows.append(
            [
                language,
                f"{v['training']['completed_steps']:,}",
                f"{v['inference']['checkpoint_step']:,}",
                f"{v['training']['best_validation_total']:.4f}",
            ]
        )
    r.table(rows, [1.2, 1.5, 2.1, 1.9])
    r.paragraph(
        "Training loss kept falling while validation loss flattened, suggesting overfitting. Both runs stopped early. Telugu's minimum at step 23,000 was 2.0231, but the improvement over the saved step-14,000 checkpoint was below 0.002. Validation uses reference variance features; test synthesis uses predictions."
    )
    r.picture(
        ROOT / "Report/assets/training_loss_comparison.png",
        "Training and validation loss for English and Telugu, with full-run and later-run panels",
    )
    r.paragraph(
        "Training and validation losses. Separate component curves are retained in Results/English and Results/Telugu. The two languages' normalized losses cannot be ranked directly.",
        "Caption",
    )
    rows = [["Language and audio", "N", "Whisper CER %", "MMS CER %", "Raw UTMOS mean"]]
    for language, v in data.items():
        rev = v["review"]
        for split, label, other in [
            ("generated", "Generated", "generated_mms"),
            ("controls", "Held-out", "controls_mms"),
        ]:
            a, b = rev[split], rev[other]
            rows.append(
                [
                    language + " " + label,
                    a["count"],
                    f"{100 * a['corpus_cer_without_spaces']:.1f}",
                    f"{100 * b['corpus_cer_without_spaces']:.1f}",
                    f"{a['predicted_mos_mean']:.2f}",
                ]
            )
    r.table(rows, [2.2, 0.4, 1.3, 1.1, 1.7])
    r.paragraph(
        "Both recognizers show a larger synthetic-versus-control CER gap for English than Telugu. Quality predictions are below control means for both languages. CER counts normalized Unicode characters with spaces removed; WER and raw transcripts are also saved. Equivalent number readings and Telugu spelling/segmentation can affect scores. The two recognizers and real-speech controls expose this uncertainty. UTMOS values are raw predictions, not human ratings; the regressor can fall outside the nominal 1-5 listener scale."
    )

    r.page()
    r.heading("Results for each test sentence")
    rows = [["ID", "Time s", "CER W %", "CER M %", "UTMOS", "Predicted F0 Hz", "Energy range"]]
    for language, v in data.items():
        acoustics = {s["id"]: s for s in v["acoustics"]["samples"]}
        for s in v["review"]["samples"]:
            a = acoustics[s["id"]]
            rows.append(
                [
                    s["id"],
                    f"{s['duration_seconds']:.2f}",
                    f"{100 * s['cer_without_spaces']:.1f}",
                    f"{100 * s['mms_cer_without_spaces']:.1f}",
                    f"{s['predicted_mos']:.2f}",
                    "-".join(f"{x:.0f}" for x in a["predicted_f0_range_hz"]),
                    "-".join(f"{x:.1f}" for x in a["predicted_energy_range"]),
                ]
            )
    r.table(rows, [0.45, 0.6, 0.8, 0.8, 0.75, 1.4, 1.9], 9.3)
    r.paragraph(
        "W = Whisper; M = MMS. Energy uses relative STFT units. Continuous predicted F0 is not a voicing decision. All waveforms are unclipped. Native-script transcripts, error counts and per-sample measurements are linked from the repository's results indexes.",
        "Caption",
    )
    notes = compact_observations({language: v["acoustics"] for language, v in data.items()})
    r.heading("Duration, pitch and energy observations")
    for language, v in data.items():
        for s in v["review"]["samples"]:
            p = r.paragraph("")
            p.add_run(s["id"] + "  ").bold = True
            p.add_run(notes[s["id"]])
            extra = {
                "E1": " Whisper did not recover the target word exactly.",
                "E2a": " Whisper recovered the sentence text exactly."
                if s["cer_without_spaces"] == 0
                else " ASR text differs from the reference.",
                "E2b": " Whisper retained conduct but changed surrounding words.",
                "E3": " Whisper did not recover rhythm.",
                "E4": " Whisper did not retain the supplied date or andaroo.",
                "T2a": " Whisper retained pattu.",
                "T2b": " Whisper retained pattu.",
            }
            p.add_run(extra.get(s["id"], ""))
    r.paragraph(
        "Word and con/duct regions follow the model's grapheme durations; they are not independently validated phonetic boundaries. The supplied E3/T3 spacing was preserved. No zero-duration item is a lexical character. Measured waveform F0, energy and low-energy intervals are plotted separately for every sample."
    )

    r.page()
    r.heading("Target word comparisons")
    r.picture(
        ROOT / "Report/assets/target_variance_comparison.png",
        "Duration, mean F0 and mean energy for conduct and pattu in their two contexts",
    )
    r.paragraph(
        "E2a/E2b are conduct; T2a/T2b are pattu. These are within-model descriptions, not cross-language quality scores.",
        "Caption",
    )
    r.bullet(
        "Conduct has similar total duration in the two contexts. Both contexts assign higher predicted energy to the first grapheme region; a clear noun/verb stress reversal is not demonstrated. Whole-word means and ASR text recovery cannot establish syllable stress."
    )
    r.bullet(
        "Pattu has similar duration and mean F0 in both contexts. The two meanings need not change segmental pronunciation. Phrase-level correctness remains uncertain without a Telugu speaker's judgment."
    )
    en = data["English"]["review"]
    e2 = next(s for s in en["samples"] if s["id"] == "E2a")
    if e2["cer_without_spaces"] == 0:
        r.bullet(
            f"Whisper recovered E2a's normalized transcript exactly, while UTMOS predicted {e2['predicted_mos']:.2f}. Text recovery and predicted naturalness assess different properties. Rare-word, spacing and date/time cases show transcript mismatches; neither recognizer is treated as a pronunciation authority."
        )
    else:
        r.bullet(
            "Transcript recovery and predicted naturalness assess different properties. Rare-word, spacing and date/time cases show mismatches; neither recognizer is treated as a pronunciation authority."
        )
    r.heading("Limitations and improvements")
    r.paragraph(
        "The assessment is automatic. Telugu ASR errors on real recordings limit the strength of pronunciation claims. UTMOS is not calibrated for these Telugu voices, and predictions below 1 are extrapolations rather than valid listener ratings. The ten controls per language are descriptive, not a matched perceptual study. Native pronunciation, stress and naturalness remain unverified."
    )
    r.paragraph(
        "Priorities are better phonetic text processing, more reliable voice conditioning and review of low-confidence alignments. Romanization can merge sounds; alignment errors affect duration targets. Resynthesizing reference mels through the fixed vocoder would help separate vocoder artifacts from acoustic-model errors."
    )
    r.heading("Reproducibility")
    r.paragraph(
        "Configurations, dataset revisions and split seeds are saved separately for each language. Checkpoints contain tokens, speaker mappings and feature statistics; optimizer states stay outside Git. Install requirements.txt for training and requirements-evaluation.txt for assessment. Model revisions, decoder settings, audio hashes and scores are recorded with the results. The README gives individual training-stage commands."
    )
    commands = "python scripts/run_project.py --work-dir PATH --wait-for-access\npython scripts/run_assessment.py --work-dir PATH"
    p = r.paragraph(commands)
    for run in p.runs:
        run.font.name = "Consolas"
        run.font.size = Pt(8)
    verification = read(ROOT / "Report/verification.json")
    match = re.search(r"(\d+) passed", " ".join(c.get("stdout", "") for c in verification.get("checks", [])))
    r.paragraph(
        f"Verification: {match[1] if match else 'recorded'} tests, static checks and compilation passed. Raw recordings, model caches and credentials remain outside the repository.",
        "Caption",
    )
    r.heading("Sources")
    for label, url in [
        (
            "Ren et al. FastSpeech 2, ICLR 2021.",
            "https://arxiv.org/abs/2006.04558",
        ),
        (
            "English data: AI4Bharat Svarah.",
            "https://huggingface.co/datasets/ai4bharat/Svarah",
        ),
        ("Telugu data: SPRINGLab IndicTTS Telugu.", "https://huggingface.co/datasets/SPRINGLab/IndicTTS_Telugu"),
        ("Reference implementation: ming024/FastSpeech2.", "https://github.com/ming024/FastSpeech2"),
        ("Whisper large-v3 decoding: SYSTRAN faster-whisper.", "https://github.com/SYSTRAN/faster-whisper"),
        (
            "Recognition: Whisper large-v3 via faster-whisper; Meta MMS-1b-all eng/tel adapters.",
            "https://huggingface.co/facebook/mms-1b-all",
        ),
        (
            "Quality prediction: SaruLab UTMOSv2, fusion_stage3 fold0 seed42.",
            "https://github.com/sarulab-speech/UTMOSv2",
        ),
    ]:
        r.reference(label, url)
    r.paragraph(
        "Full attribution, model revisions and licensing links are in docs/THIRD_PARTY.md and docs/automatic_evaluation_models.json.",
        "Caption",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    r.doc.core_properties.title = "FastSpeech 2 for English and Telugu"
    r.doc.core_properties.author = "Yuvraj Verma"
    r.doc.core_properties.subject = "IIIT Hyderabad Voice AI assignment with automatic assessment"
    r.doc.save(args.output)
    print(args.output)


if __name__ == "__main__":
    main()
