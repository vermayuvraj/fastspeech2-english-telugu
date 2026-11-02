"""SpeechBrain's published multispeaker HiFi-GAN weights in the reference generator.

SpeechBrain wraps convolutions in a .conv submodule. The parameter names are
translated and then loaded strictly; no unknown weights are silently discarded.
"""
from __future__ import annotations

import json

import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download

from vendor.ming024.hifigan import AttrDict, Generator

from .config import ROOT, sha256


class Vocoder:
    def __init__(self, config: dict, cache_dir, device):
        setting = config["vocoder"]
        path = hf_hub_download(setting["repository"], setting["filename"], revision=setting["revision"], cache_dir=cache_dir)
        state = torch.load(path, map_location="cpu", weights_only=True)
        state = {key.replace(".conv.", "."): value for key, value in state.items()}
        h = AttrDict(json.loads((ROOT / "vendor/ming024/hifigan/config.json").read_text()))
        self.generator = Generator(h)
        self.generator.load_state_dict(state, strict=True)
        self.generator.eval().remove_weight_norm()
        self.generator.to(device)
        self.device, self.padding = device, setting["inference_padding"]
        self.hop = config["audio"]["hop_length"]
        self.provenance = {**setting, "sha256": sha256(__import__("pathlib").Path(path)), "parameter_name_conversion": ".conv. -> .; strict state_dict match", "fine_tuned": False}

    @torch.inference_mode()
    def __call__(self, mel):
        # Match SpeechBrain's replicate padding, then remove its output context.
        padded = F.pad(mel.to(self.device), (self.padding, self.padding), mode="replicate")
        audio = self.generator(padded)[0, 0]
        crop = self.padding * self.hop
        if crop:
            audio = audio[crop:-crop]
        return audio.detach().float().cpu().numpy()
