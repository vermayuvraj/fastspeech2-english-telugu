"""HiFi-GAN-compatible 22.05 kHz log mels and measured variance targets."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def acoustic_features(y: np.ndarray, config: dict) -> dict:
    import librosa
    import pyworld
    sr, hop, nfft = config["sample_rate"], config["hop_length"], config["n_fft"]
    wave = torch.from_numpy(np.asarray(y, dtype=np.float32)).unsqueeze(0)
    # HiFi-GAN uses reflect padding and center=False. Exactly floor(N / hop) frames.
    wave = F.pad(wave.unsqueeze(1), ((nfft - hop) // 2, (nfft - hop) // 2), mode="reflect").squeeze(1)
    spec = torch.stft(wave, nfft, hop, config["win_length"], window=torch.hann_window(config["win_length"]), center=False, return_complex=True)
    magnitude = torch.sqrt(spec.real.square() + spec.imag.square() + 1e-9)[0].numpy()
    basis = librosa.filters.mel(sr=sr, n_fft=nfft, n_mels=config["n_mels"], fmin=config["fmin"], fmax=config["fmax"])
    mel = np.log(np.maximum(basis @ magnitude, 1e-5)).T.astype(np.float32)
    energy = np.linalg.norm(magnitude, axis=0)
    f0, times = pyworld.dio(np.asarray(y, dtype=np.float64), sr, f0_floor=config["f0_min"], f0_ceil=config["f0_max"], frame_period=hop / sr * 1000)
    f0 = pyworld.stonemask(np.asarray(y, dtype=np.float64), f0, times, sr)
    # WORLD samples frame centers at t=0, hop/sr, ...; interpolate onto STFT centers.
    mel_times = (np.arange(len(mel)) * hop + hop / 2) / sr
    nearest = np.clip(np.rint(mel_times * sr / hop).astype(int), 0, len(f0) - 1)
    voiced = f0[nearest] > 0
    good = f0 > 0
    if good.sum() < 2:
        raise ValueError("Fewer than two voiced pitch frames")
    log_f0 = np.interp(mel_times, times[good], np.log(f0[good]))
    return {"mel": mel, "log_f0": log_f0.astype(np.float32), "log_energy": np.log(np.maximum(energy, 1e-5)).astype(np.float32), "voiced": voiced}


def token_average(values: np.ndarray, durations: np.ndarray) -> np.ndarray:
    result = np.zeros(len(durations), dtype=np.float32)
    start = 0
    for i, count in enumerate(durations):
        if count:
            result[i] = values[start:start + count].mean()
        start += count
    if start != len(values):
        raise ValueError("Durations do not cover all feature frames")
    return result
