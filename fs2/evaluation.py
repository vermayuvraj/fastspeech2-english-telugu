"""Deterministic scoring and token timing helpers for automatic evaluation."""

from __future__ import annotations

import unicodedata
import re
import json
import hashlib
from pathlib import Path

import numpy as np


def scoring_text(value: str) -> str:
    """Keep native scripts and combining marks; standardize case and punctuation."""
    value = unicodedata.normalize("NFC", value).casefold()
    value = value.replace("’", "'").replace("'", "")
    value = "".join(" " if unicodedata.category(c)[0] in {"P", "S"} else c for c in value)
    return " ".join(value.split())


def transcript_errors(reference: str, hypothesis: str) -> dict:
    from jiwer import process_characters, process_words

    reference, hypothesis = scoring_text(reference), scoring_text(hypothesis)
    words = process_words(reference, hypothesis)
    # Ignore whitespace for CER: supplied E3/T3 deliberately split a word.
    chars = process_characters(reference.replace(" ", ""), hypothesis.replace(" ", ""))
    return {
        "reference_normalized": reference,
        "hypothesis_normalized": hypothesis,
        "wer": float(words.wer),
        "cer_without_spaces": float(chars.cer),
        "reference_words": len(reference.split()),
        "reference_characters_without_spaces": len(reference.replace(" ", "")),
        "word_substitutions": words.substitutions,
        "word_deletions": words.deletions,
        "word_insertions": words.insertions,
        "character_substitutions": chars.substitutions,
        "character_deletions": chars.deletions,
        "character_insertions": chars.insertions,
    }


def score_recognition(reference: str, hypothesis: str, language: str) -> dict:
    """Normalize written clock variants without correcting recognized words."""
    from fs2.frontend import expand_numbers

    prepared = hypothesis
    if language == "English":
        prepared = re.sub(
            r"\b(\d{1,2})[:.](\d{2})\s*([ap])\.?\s*m\.?",
            lambda m: f"{m[1]}:{m[2]} {m[3].upper()}M",
            prepared,
            flags=re.I,
        )
    elif language == "Telugu":
        prepared = re.sub(r"\b(\d{1,2})\.(\d{2})(?=\s*గంటల)", r"\1:\2", prepared)
    try:
        expanded = expand_numbers(prepared, language)
        warning = None
    except ValueError as error:
        expanded, warning = prepared, str(error)
    return {
        "transcript_expanded": expanded,
        "hypothesis_normalization_warning": warning,
        "verbatim_errors": transcript_errors(reference, hypothesis),
        **transcript_errors(reference, expanded),
    }


def weighted_mean(values, weights):
    values, weights = np.asarray(values), np.asarray(weights)
    if weights.sum() <= 0:
        return None
    return float(np.average(values, weights=weights))


def automatic_review_is_current(root: Path, language: str, sample_id: str) -> bool:
    """Require declared automatic methods and matching audio/evidence, not just prose."""
    try:
        directory = root / "Results" / language
        data = json.loads((directory / "automatic_review.json").read_text(encoding="utf-8"))
        if data.get("method") != "automatic_assessment" or data.get("human_listening_performed") is not False:
            return False
        if not data.get("author_does_not_know_telugu") or not data.get("author_statement"):
            return False
        for key in ["controls", "controls_mms"]:
            if data[key]["count"] != 10:
                return False
        row = next(s for s in data["samples"] if s["id"] == sample_id)
        fields = [
            "pronunciation_observations",
            "speech_quality_observations",
            "duration_observations",
            "pitch_observations",
            "energy_observations",
        ]
        if any(not str(row.get(key, "")).strip() or str(row[key]).lower() == "pending" for key in fields):
            return False
        digest = hashlib.sha256((directory / "test_generated_audios" / f"{sample_id}.wav").read_bytes()).hexdigest()
        if row["wav_sha256"] != digest:
            return False
        for filename in [
            "automatic_asr.json",
            "automatic_asr_mms.json",
            "automatic_quality.json",
            "automatic_acoustic_analysis.json",
        ]:
            path = directory / filename
            if row["evidence"][filename] != hashlib.sha256(path.read_bytes()).hexdigest():
                return False
            source = json.loads(path.read_text(encoding="utf-8"))
            sample = next(s for s in source["samples"] if s["id"] == sample_id)
            if sample["wav_sha256"] != digest:
                return False
        return (directory / "automatic_assessment_plots" / f"{sample_id}.png").is_file()
    except (OSError, ValueError, KeyError, StopIteration):
        return False


def token_spans(tokens, duration_frames, f0, energy, hop, sample_rate):
    """Recover model-timed word and boundary spans without claiming phone alignment."""
    arrays = [np.asarray(x) for x in [duration_frames, f0, energy]]
    if any(len(x) != len(tokens) for x in arrays) or np.any(arrays[0] < 0):
        raise ValueError("Invalid or inconsistent token predictions")
    duration, f0, energy = arrays
    words, boundaries = [], []
    cursor, start, indices = 0, 0, []

    def word():
        if not indices:
            return
        weights = duration[indices]
        words.append(
            {
                "word": "".join(tokens[i] for i in indices),
                "token_indices": list(indices),
                "start_seconds": float(start * hop / sample_rate),
                "end_seconds": float(cursor * hop / sample_rate),
                "duration_ms": float(weights.sum() * hop / sample_rate * 1000),
                "mean_predicted_f0_hz": weighted_mean(f0[indices], weights),
                "mean_predicted_stft_energy": weighted_mean(energy[indices], weights),
            }
        )

    for i, token in enumerate(tokens):
        if token in {"|", "<sil>"}:
            word()
            indices = []
            boundaries.append(
                {
                    "token_index": i,
                    "label": token,
                    "start_seconds": float(cursor * hop / sample_rate),
                    "duration_ms": float(duration[i] * hop / sample_rate * 1000),
                    "predicted_stft_energy": float(energy[i]),
                }
            )
        else:
            if not indices:
                start = cursor
            indices.append(i)
        cursor += int(duration[i])
    word()
    return words, boundaries
