"""Deterministic, transcript-grouped, audio-deduplicated selection."""
from __future__ import annotations

import hashlib
from collections import defaultdict


def select_splits(rows: list[dict], train_hours: float, validation_hours: float, seed: int) -> tuple[list[dict], list[dict], dict]:
    groups = defaultdict(list)
    seen_audio = set()
    duplicate_audio = 0
    for row in sorted(rows, key=lambda x: x["id"]):
        if row["audio_sha256"] in seen_audio:
            duplicate_audio += 1
            continue
        seen_audio.add(row["audio_sha256"])
        groups[row["normalized_text"]].append(row)
    ordered = sorted(groups, key=lambda text: hashlib.sha256(f"{seed}\0{text}".encode()).hexdigest())
    validation, train = [], []
    seconds = {"train": 0.0, "validation": 0.0}
    # Reserve complete groups for training so every validation voice is learnable.
    reserved, covered = set(), set()
    all_speakers = {row["speaker"] for rows_in_group in groups.values() for row in rows_in_group}
    for text in ordered:
        speakers_here = {row["speaker"] for row in groups[text]}
        if speakers_here - covered:
            reserved.add(text)
            covered.update(speakers_here)
            train.extend(groups[text])
            seconds["train"] += sum(row["duration_seconds"] for row in groups[text])
        if covered == all_speakers:
            break
    # Select entire transcript groups; no identical normalized prompt crosses splits.
    for text in ordered:
        if text in reserved:
            continue
        key = "validation" if seconds["validation"] < validation_hours * 3600 else "train"
        if key == "train" and seconds["train"] >= train_hours * 3600:
            break
        target = validation if key == "validation" else train
        target.extend(groups[text])
        seconds[key] += sum(row["duration_seconds"] for row in groups[text])
    assert_disjoint(train, validation)
    speakers = sorted({row["speaker"] for row in train})
    unseen = sorted({row["speaker"] for row in validation} - set(speakers))
    if unseen:
        raise ValueError(f"Validation includes speakers absent from training: {unseen}. Adjust seed/selection.")
    summary = {"seed": seed, "strategy": "sha256(seed + normalized transcript), whole transcript groups, reserve training speaker coverage, then validation and training budgets", "duration_seconds": seconds, "utterances": {"train": len(train), "validation": len(validation)}, "duplicate_audio_removed": duplicate_audio, "train_speakers": speakers, "speaker_disjoint": False, "text_disjoint": True, "audio_disjoint": True}
    return train, validation, summary


def assert_disjoint(train: list[dict], validation: list[dict]) -> None:
    for field in ["id", "audio_sha256", "normalized_text"]:
        overlap = {r[field] for r in train} & {r[field] for r in validation}
        if overlap:
            raise ValueError(f"Train/validation {field} overlap: {len(overlap)}")
