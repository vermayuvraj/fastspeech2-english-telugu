"""Duration-aware batches with padding excluded from every loss term."""
from __future__ import annotations

import random

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler
from torch.nn.utils.rnn import pad_sequence

from .config import read_jsonl


class AcousticDataset(Dataset):
    def __init__(self, directory, split: str, stats: dict, speakers: dict):
        self.directory = directory
        self.rows = read_jsonl(directory / f"{split}.jsonl")
        self.stats, self.speakers = stats, speakers
        if not self.rows:
            raise ValueError(f"Empty {split} dataset")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        with np.load(self.directory / row["feature_path"], allow_pickle=False) as archive:
            values = {key: archive[key].copy() for key in ["tokens", "durations", "mel", "pitch", "energy"]}
        for key in ["pitch", "energy"]:
            _, _, mean, std = self.stats[key]
            values[key] = (values[key] - mean) / std
            values[key][values["durations"] == 0] = 0
        if int(values["durations"].sum()) != len(values["mel"]):
            raise ValueError(f"Broken frame partition: {row['id']}")
        return {**values, "speaker": self.speakers[row["speaker"]], "id": row["id"]}


class BucketBatchSampler(Sampler):
    def __init__(self, rows: list[dict], batch_size: int, seed: int, shuffle: bool):
        self.rows, self.batch_size, self.seed, self.shuffle = rows, batch_size, seed, shuffle
        self.epoch = 0

    def __len__(self):
        return (len(self.rows) + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        indices = list(range(len(self.rows)))
        if self.shuffle:
            rng.shuffle(indices)
        batches = []
        size = self.batch_size * 32
        for start in range(0, len(indices), size):
            bucket = sorted(indices[start:start + size], key=lambda i: self.rows[i]["frames"])
            batches.extend(bucket[j:j + self.batch_size] for j in range(0, len(bucket), self.batch_size))
        if self.shuffle:
            rng.shuffle(batches)
        yield from batches


def collate(rows: list[dict]) -> dict:
    result = {key: pad_sequence([torch.as_tensor(row[key]) for row in rows], batch_first=True) for key in ["tokens", "durations", "mel", "pitch", "energy"]}
    result["speakers"] = torch.tensor([row["speaker"] for row in rows], dtype=torch.long)
    result["src_lens"] = torch.tensor([len(row["tokens"]) for row in rows], dtype=torch.long)
    result["mel_lens"] = torch.tensor([len(row["mel"]) for row in rows], dtype=torch.long)
    result["ids"] = [row["id"] for row in rows]
    return result


def forward_batch(model, batch):
    return model(batch["speakers"], batch["tokens"], batch["src_lens"], batch["tokens"].shape[1], mels=batch["mel"], mel_lens=batch["mel_lens"], max_mel_len=batch["mel"].shape[1], p_targets=batch["pitch"], e_targets=batch["energy"], d_targets=batch["durations"])
