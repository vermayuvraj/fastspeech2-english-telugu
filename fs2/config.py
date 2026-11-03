"""Configuration, provenance and atomic file operations."""
from __future__ import annotations

import hashlib
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]


def load_config(path: str | Path) -> dict:
    path = Path(path).resolve()
    with path.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    config["_config_path"] = str(path)
    config["_config_sha256"] = sha256(path)
    return config


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    # Windows readers (including the live monitor) can briefly hold a file
    # without delete-sharing. Keep replacement atomic and retry that short lock.
    for attempt in range(8):
        try:
            os.replace(temp, path)
            break
        except PermissionError:
            if attempt == 7:
                raise
            time.sleep(min(0.05 * (attempt + 1), 0.2))


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(temp, path)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Repeatable seeds, with nondeterministic CUDA kernels explicitly recorded.
    torch.backends.cudnn.benchmark = False


def workspace(config: dict, work_dir: str | Path) -> Path:
    result = Path(work_dir).resolve() / config["language"]
    result.mkdir(parents=True, exist_ok=True)
    return result
