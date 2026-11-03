"""Checks for scoring conventions that could otherwise distort the findings."""

import numpy as np
import pytest
import hashlib
import json

from fs2.evaluation import automatic_review_is_current, score_recognition, scoring_text, token_spans, transcript_errors


def test_punctuation_and_case_do_not_change_words():
    assert scoring_text("The orchestra’s CONDUCT!") == "the orchestras conduct"
    assert scoring_text("పట్టు చీరను.") == "పట్టు చీరను"


def test_internal_spacing_affects_wer_but_not_character_content():
    result = transcript_errors("the rhyth m changed", "the rhythm changed")
    assert result["wer"] > 0
    assert result["cer_without_spaces"] == 0


def test_deletions_are_counted_and_insertions_can_exceed_one_hundred_percent():
    result = transcript_errors("one two three", "one three")
    assert result["word_deletions"] == 1
    assert result["wer"] == pytest.approx(1 / 3)
    assert transcript_errors("one", "one two three four")["wer"] == 3


def test_clock_punctuation_is_not_mistaken_for_decimal_pronunciation():
    clock = score_recognition("at ten forty-five a m", "at 10.45 a.m.", "English")
    assert clock["wer"] == 0
    decimal = score_recognition("ten point four five", "10.45", "English")
    assert decimal["wer"] == 0


def test_word_spans_preserve_gaps_and_ignore_zero_duration_pitch():
    words, gaps = token_spans(
        ["<sil>", "a", "b", "|", "c", "<sil>"],
        [2, 1, 0, 3, 4, 1],
        [100, 120, 900, 100, 150, 100],
        [1, 2, 999, 1, 3, 1],
        100,
        1000,
    )
    assert words[0]["word"] == "ab"
    assert words[0]["mean_predicted_f0_hz"] == 120
    assert words[1]["start_seconds"] == pytest.approx(0.6)
    assert sum(w["duration_ms"] for w in words) + sum(g["duration_ms"] for g in gaps) == 1100
    with pytest.raises(ValueError):
        token_spans(["a"], np.array([-1]), [100], [2], 100, 1000)


def test_assessment_rejects_changed_audio_or_measurements(tmp_path):
    directory = tmp_path / "Results/Telugu"
    (directory / "test_generated_audios").mkdir(parents=True)
    (directory / "automatic_assessment_plots").mkdir()
    wave = directory / "test_generated_audios/T1.wav"
    wave.write_bytes(b"test fixture audio")
    digest = hashlib.sha256(wave.read_bytes()).hexdigest()
    evidence = {}
    for name in [
        "automatic_asr.json",
        "automatic_asr_mms.json",
        "automatic_quality.json",
        "automatic_acoustic_analysis.json",
    ]:
        path = directory / name
        path.write_text(json.dumps({"samples": [{"id": "T1", "wav_sha256": digest}]}))
        evidence[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    row = {"id": "T1", "wav_sha256": digest, "evidence": evidence}
    row.update(
        {
            k: "Automatic measurement recorded"
            for k in [
                "pronunciation_observations",
                "speech_quality_observations",
                "duration_observations",
                "pitch_observations",
                "energy_observations",
            ]
        }
    )
    review = {
        "method": "automatic_assessment",
        "human_listening_performed": False,
        "author_does_not_know_telugu": True,
        "author_statement": "I do not know Telugu.",
        "controls": {"count": 10},
        "controls_mms": {"count": 10},
        "samples": [row],
    }
    (directory / "automatic_review.json").write_text(json.dumps(review))
    (directory / "automatic_assessment_plots/T1.png").write_bytes(b"test fixture plot")
    assert automatic_review_is_current(tmp_path, "Telugu", "T1")
    wave.write_bytes(b"changed audio")
    assert not automatic_review_is_current(tmp_path, "Telugu", "T1")
    wave.write_bytes(b"test fixture audio")
    (directory / "automatic_quality.json").write_text("{}")
    assert not automatic_review_is_current(tmp_path, "Telugu", "T1")


def test_atomic_json_replace_retries_a_temporary_reader_lock(tmp_path, monkeypatch):
    from fs2 import config

    original = config.os.replace
    attempts = []

    def locked_once(source, destination):
        attempts.append(1)
        if len(attempts) == 1:
            raise PermissionError("temporary reader lock")
        return original(source, destination)

    monkeypatch.setattr(config.os, "replace", locked_once)
    monkeypatch.setattr(config.time, "sleep", lambda _: None)
    destination = tmp_path / "status.json"
    config.write_json(destination, {"complete": True})
    assert json.loads(destination.read_text()) == {"complete": True}
    assert b"\r\n" not in destination.read_bytes()
    assert len(attempts) == 2
