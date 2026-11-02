"""Construct the adapted reference model with portable, embedded metadata."""
from __future__ import annotations

from copy import deepcopy

from vendor.ming024.model.fastspeech2 import FastSpeech2

from .frontend import SYMBOLS


def model_configs(config: dict, stats: dict, speakers: dict) -> tuple[dict, dict]:
    preprocess = {"stats": stats, "preprocessing": {"mel": {"n_mel_channels": config["audio"]["n_mels"]}, "pitch": {"feature": "phoneme_level"}, "energy": {"feature": "phoneme_level"}}}
    # 'phoneme_level' is the upstream API name; our units are romanized graphemes.
    model = deepcopy(config["model"])
    model["vocab_size"] = len(SYMBOLS)
    model["n_speakers"] = len(speakers)
    model["multi_speaker"] = True
    return preprocess, model


def create_model(config: dict, stats: dict, speakers: dict) -> FastSpeech2:
    preprocess, model = model_configs(config, stats, speakers)
    return FastSpeech2(preprocess, model)
