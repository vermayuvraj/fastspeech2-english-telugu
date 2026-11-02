"""MMS CTC alignment converted to an exact partition of mel frames."""
from __future__ import annotations

import numpy as np
import torch


def spans_to_durations(words, spans, emission_frames: int, audio_seconds: float, mel_frames: int, sample_rate: int = 22050, hop: int = 256):
    if len(words) != len(spans) or not spans:
        raise ValueError("Alignment word count mismatch")
    ratio = audio_seconds / emission_frames * sample_rate / hop
    ends = [0]
    tokens = []
    for wi, (word, word_spans) in enumerate(zip(words, spans)):
        if len(word) != len(word_spans):
            raise ValueError("Alignment character count mismatch")
        # Explicit leading silence and interword gap tokens own actual blank gaps.
        tokens.append("<sil>" if wi == 0 else "|")
        ends.append(round(word_spans[0].start * ratio))
        for ci, (char, span) in enumerate(zip(word, word_spans)):
            tokens.append(char)
            end = span.end if ci == len(word_spans) - 1 else (span.end + word_spans[ci + 1].start) / 2
            ends.append(round(end * ratio))
    tokens.append("<sil>")
    ends.append(mel_frames)
    boundaries = np.maximum.accumulate(np.clip(ends, 0, mel_frames))
    durations = np.diff(boundaries).astype(np.int64)
    if durations.sum() != mel_frames or len(durations) != len(tokens):
        raise ValueError("Invalid duration partition")
    return tokens, durations


class ForcedAligner:
    def __init__(self, device: str = "cuda"):
        import torchaudio
        self.bundle = torchaudio.pipelines.MMS_FA
        self.device = torch.device(device)
        self.model = self.bundle.get_model(with_star=False).to(self.device).eval()
        self.tokenizer = self.bundle.get_tokenizer()
        self.aligner = self.bundle.get_aligner()

    @torch.inference_mode()
    def __call__(self, waveform: np.ndarray, sample_rate: int, words: list[str], mel_frames: int, hop: int):
        import torchaudio.functional as AF
        tensor = torch.from_numpy(waveform).float().unsqueeze(0)
        tensor = AF.resample(tensor, sample_rate, self.bundle.sample_rate).to(self.device)
        emissions, _ = self.model(tensor)
        spans = self.aligner(emissions[0], self.tokenizer(words))
        tokens, durations = spans_to_durations(words, spans, emissions.shape[1], len(waveform) / sample_rate, mel_frames, sample_rate, hop)
        total = sum(span.end - span.start for word in spans for span in word)
        score = sum(span.score * (span.end - span.start) for word in spans for span in word) / max(total, 1)
        return tokens, durations, float(score)
