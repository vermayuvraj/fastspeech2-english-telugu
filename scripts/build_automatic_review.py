"""Build per-sentence findings only after all automatic measurements exist."""

from datetime import datetime, timezone
import json
from pathlib import Path
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fs2.config import ROOT, sha256, write_json


DISCLOSURE = (
    "I do not know Telugu, so I used automatic transcription, predicted-quality scores and acoustic measurements "
    "to assess the Telugu outputs. Assessment of both languages is automatic. No native-speaker listening "
    "or human MOS study was performed."
)


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def aggregate(asr, quality, split):
    rows = [s for s in asr["samples"] if s["split"] == split]
    scores = [s["predicted_mos"] for s in quality["samples"] if s["split"] == split]
    return {
        "count": len(rows),
        "corpus_wer": sum(s["word_substitutions"] + s["word_deletions"] + s["word_insertions"] for s in rows)
        / sum(s["reference_words"] for s in rows),
        "corpus_cer_without_spaces": sum(
            s["character_substitutions"] + s["character_deletions"] + s["character_insertions"] for s in rows
        )
        / sum(s["reference_characters_without_spaces"] for s in rows),
        "predicted_mos_mean": statistics.mean(scores),
        "predicted_mos_median": statistics.median(scores),
        "predicted_mos_range": [min(scores), max(scores)],
    }


def word(sample, name):
    return next(w for w in sample["words"] if w["word"] == name)


def gap_after(sample, name):
    w = word(sample, name)
    return next(b["duration_ms"] for b in sample["boundaries"] if b["token_index"] == w["token_indices"][-1] + 1)


def specific_findings(sample):
    name = sample["id"]
    words = sample["words"]
    lo, hi = sample["predicted_f0_range_hz"]
    elo, ehi = sample["predicted_energy_range"]
    duration = f"Utterance duration is {sample['duration_seconds']:.2f} s."
    pitch = f"Predicted continuous F0 spans {lo:.0f}-{hi:.0f} Hz."
    energy = f"Predicted relative STFT energy spans {elo:.1f}-{ehi:.1f}."
    if name in {"E1", "T1"}:
        target = word(sample, "antidisestablishmentarianism") if name == "E1" else words[0]
        duration += f" The target word occupies {target['duration_ms']:.0f} ms ({target['duration_ms'] / (sample['duration_seconds'] * 10):.0f}% of the utterance)."
        pitch += f" Its duration-weighted F0 is {target['mean_predicted_f0_hz']:.1f} Hz."
        energy += f" Its duration-weighted mean is {target['mean_predicted_stft_energy']:.1f}."
    elif name.startswith("E2"):
        target = word(sample, "conduct")
        con, duct = sample["conduct_grapheme_regions"]
        duration += f" Conduct takes {target['duration_ms']:.0f} ms; con/duct grapheme regions receive {con['duration_ms']:.0f}/{duct['duration_ms']:.0f} ms."
        pitch += f" Con/duct means are {con['mean_predicted_f0_hz']:.1f}/{duct['mean_predicted_f0_hz']:.1f} Hz."
        energy += f" Con/duct means are {con['mean_predicted_energy']:.1f}/{duct['mean_predicted_energy']:.1f}. Higher first-region energy occurs in both contexts; this cue does not demonstrate the requested stress reversal."
    elif name.startswith("T2"):
        target = word(sample, "pattttu")
        duration += f" Pattu takes {target['duration_ms']:.0f} ms."
        pitch += f" Pattu's mean is {target['mean_predicted_f0_hz']:.1f} Hz."
        energy += f" Pattu's mean is {target['mean_predicted_stft_energy']:.1f}. The meaning difference does not itself require a different segmental pronunciation."
    elif name in {"E3", "T3"}:
        left, right = ("rhyth", "m") if name == "E3" else ("adda", "viloo")
        gap = gap_after(sample, left)
        interword = [b["duration_ms"] for b in sample["boundaries"] if b["label"] == "|"]
        duration += f" The retained {left}/{right} split receives a {gap:.0f} ms boundary; the sentence's median interword boundary is {statistics.median(interword):.0f} ms."
        pitch += f" The two fragments' mean F0 values are {word(sample, left)['mean_predicted_f0_hz']:.1f}/{word(sample, right)['mean_predicted_f0_hz']:.1f} Hz."
        energy += f" Their mean energy values are {word(sample, left)['mean_predicted_stft_energy']:.1f}/{word(sample, right)['mean_predicted_stft_energy']:.1f}. Character-error scoring ignores whitespace, while word-error scoring retains this split."
    elif name == "E4":
        duration += f" The a/m tokens each receive {word(sample, 'a')['duration_ms']:.0f} ms. The boundary after the year ends is {gap_after(sample, 'six'):.0f} ms."
        pitch += f" Andaroo's mean is {word(sample, 'andaroo')['mean_predicted_f0_hz']:.1f} Hz."
        energy += " Two zero-duration items are word-boundary tokens, not omitted lexical characters. ASR checks the expanded time/date and the mixed-language word."
    elif name == "T4":
        duration += f" The boundary between nalabhai/aidu (forty/five) is {gap_after(sample, 'nalabhai'):.0f} ms; the later clause boundary is {gap_after(sample, 'praarambhamavutumdi'):.0f} ms."
        pitch += f" Mean word F0 falls from {words[0]['mean_predicted_f0_hz']:.1f} Hz at the start to {words[-1]['mean_predicted_f0_hz']:.1f} Hz at the end."
        energy += " One zero-duration word-boundary token occurs in the year; no lexical token has zero duration. This is not evidence of an omitted spoken word."
    return duration, pitch, energy


def main():
    protocol = {
        "method": "automatic_assessment",
        "human_listening_performed": False,
        "author_does_not_know_telugu": True,
        "author_statement": DISCLOSURE,
        "asr_reference_supplied_to_recognizer": False,
        "recognizers": ["Whisper large-v3", "MMS-1b-all with eng/tel language adapters"],
        "model_revisions": read(ROOT / "docs/automatic_evaluation_models.json"),
        "controls": "First ten validation utterances in each language, ordered by ID; same selection as the earlier MMS diagnostic. They were excluded from TTS training but belong to the validation set used for checkpoint selection. Controls are not duration- or speaker-matched to test sentences.",
        "limitations": [
            "ASR transcript disagreement can be caused by the recognizer, spelling, word spacing, accent or synthesis; it is not phoneme-level pronunciation accuracy.",
            "UTMOSv2 predicts naturalness; its output is not human MOS. Telugu calibration is unverified. Cross-language quality ranking is not justified.",
            "The quality regressor is unbounded: a prediction below 1 or above 5 is retained as a raw extrapolation, not a valid listener rating on the nominal 1-5 MOS scale.",
            "Model grapheme durations define the analysed word and con/duct spans; these are not independently validated syllable or phoneme alignments.",
            "Native-speaker correctness, conversational naturalness and the intended stress contrast are not certified by these automatic tests.",
        ],
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }
    summary = {}
    for language in ["English", "Telugu"]:
        directory = ROOT / "Results" / language
        asr = read(directory / "automatic_asr.json")
        quality = read(directory / "automatic_quality.json")
        crosscheck = read(directory / "automatic_asr_mms.json")
        acoustic = read(directory / "automatic_acoustic_analysis.json")
        expected = read(ROOT / "test_sentences" / f"{language}.json")
        assert len([s for s in asr["samples"] if s["split"] == "held_out"]) == 10
        assert len([s for s in quality["samples"] if s["split"] == "held_out"]) == 10
        assert len([s for s in crosscheck["samples"] if s["split"] == "held_out"]) == 10
        assert {s["id"] for s in asr["samples"] if s["split"] == "generated"} == {s["id"] for s in expected}
        controls = aggregate(asr, quality, "held_out")
        generated = aggregate(asr, quality, "generated")
        controls_mms = aggregate(crosscheck, quality, "held_out")
        generated_mms = aggregate(crosscheck, quality, "generated")
        findings = []
        for row in expected:
            name = row["id"]
            a = next(s for s in asr["samples"] if s["id"] == name and s["split"] == "generated")
            q = next(s for s in quality["samples"] if s["id"] == name and s["split"] == "generated")
            m = next(s for s in crosscheck["samples"] if s["id"] == name and s["split"] == "generated")
            c = next(s for s in acoustic["samples"] if s["id"] == name)
            digest = sha256(directory / "test_generated_audios" / f"{name}.wav")
            assert a["wav_sha256"] == m["wav_sha256"] == q["wav_sha256"] == c["wav_sha256"] == digest
            duration, pitch, energy = specific_findings(c)
            pronunciation = f"Whisper WER is {a['wer']:.1%}; whitespace-free CER is {a['cer_without_spaces']:.1%}. Independent MMS gives WER {m['wer']:.1%}, CER {m['cer_without_spaces']:.1%}. Whisper has {a['word_substitutions']} word substitutions, {a['word_deletions']} deletions and {a['word_insertions']} insertions. Recognizer disagreement is not a listening judgment or proof of a synthesis error."
            naturalness = (
                f"UTMOSv2 predicts {q['predicted_mos']:.2f}, compared with a mean of {controls['predicted_mos_mean']:.2f} for ten same-language held-out recordings. No waveform clipping was found."
                if c["clipped_fraction"] == 0
                else f"UTMOSv2 predicts {q['predicted_mos']:.2f}; clipped fraction is {c['clipped_fraction']:.2%}."
            )
            findings.append(
                {
                    "id": name,
                    "method": "automatic_assessment",
                    "human_listening_performed": False,
                    "raw_text": row["text"],
                    "wav_sha256": digest,
                    "pronunciation_observations": pronunciation,
                    "speech_quality_observations": naturalness,
                    "duration_observations": duration,
                    "pitch_observations": pitch,
                    "energy_observations": energy,
                    "asr_transcript": a["transcript"],
                    "mms_transcript": m["transcript"],
                    "wer": a["wer"],
                    "cer_without_spaces": a["cer_without_spaces"],
                    "mms_wer": m["wer"],
                    "mms_cer_without_spaces": m["cer_without_spaces"],
                    "predicted_mos": q["predicted_mos"],
                    "duration_seconds": c["duration_seconds"],
                    "evidence": {
                        p: sha256(directory / p)
                        for p in [
                            "automatic_asr.json",
                            "automatic_asr_mms.json",
                            "automatic_quality.json",
                            "automatic_acoustic_analysis.json",
                        ]
                    },
                }
            )
        review = {
            **protocol,
            "language": language,
            "controls": controls,
            "generated": generated,
            "controls_mms": controls_mms,
            "generated_mms": generated_mms,
            "samples": findings,
        }
        write_json(directory / "automatic_review.json", review)
        lines = [
            f"# {language} automatic assessment",
            "",
            DISCLOSURE,
            "",
            "These are automatic observations. Human listening fields remain unfilled and are not represented as completed.",
            "",
        ]
        for sample in findings:
            lines.extend(
                [
                    "## " + sample["id"],
                    "",
                    "- " + sample["pronunciation_observations"],
                    "- " + sample["speech_quality_observations"],
                    "- " + sample["duration_observations"],
                    "- " + sample["pitch_observations"],
                    "- " + sample["energy_observations"],
                    "",
                    "Whisper transcript: " + json.dumps(sample["asr_transcript"], ensure_ascii=False),
                    "",
                    "MMS transcript: " + json.dumps(sample["mms_transcript"], ensure_ascii=False),
                    "",
                    f"[Waveform and acoustic overlay](automatic_assessment_plots/{sample['id']}.png)",
                    "",
                ]
            )
        (directory / "AUTOMATIC_ASSESSMENT.md").write_text("\n".join(lines), encoding="utf-8")
        summary[language] = {
            "held_out": controls,
            "generated": generated,
            "held_out_mms": controls_mms,
            "generated_mms": generated_mms,
        }
    write_json(ROOT / "Report/automatic_assessment_protocol.json", {**protocol, "summary": summary})
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
