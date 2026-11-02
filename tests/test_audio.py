import numpy as np

from fs2.audio import acoustic_features, token_average
from fs2.config import ROOT, load_config


def test_measured_pitch_energy_and_mel_frame_convention():
    config = load_config(ROOT / "English/config.yaml")["audio"]
    time = np.arange(config["sample_rate"]) / config["sample_rate"]
    wave = (0.3 * np.sin(2 * np.pi * 220 * time) + 0.1 * np.sin(2 * np.pi * 440 * time)).astype(np.float32)
    full = acoustic_features(wave, config)
    quiet = acoustic_features(wave / 2, config)
    assert full["mel"].shape == (len(wave) // config["hop_length"], 80)
    assert abs(np.median(np.exp(full["log_f0"])) - 220) < 5
    assert abs(np.mean(full["log_energy"] - quiet["log_energy"]) - np.log(2)) < 0.02
    duration = np.array([10, 0, len(full["mel"]) - 10])
    averaged = token_average(full["log_f0"], duration)
    assert averaged[1] == 0
    assert np.isfinite(averaged).all()
