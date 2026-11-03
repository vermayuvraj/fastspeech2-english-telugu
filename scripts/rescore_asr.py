"""Update text-scoring conventions without rerunning recognition or changing audio."""

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fs2.config import ROOT, write_json
from fs2.evaluation import score_recognition

for language in ["English", "Telugu"]:
    path = ROOT / "Results" / language / "automatic_asr.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    for sample in data["samples"]:
        sample.update(score_recognition(sample["reference_expanded"], sample["transcript"], language))
    data.update(
        scoring_version=3,
        scoring="Same deterministic number/time expansion on reference and hypothesis; clock notation such as 10.45 a.m. is treated as 10:45 AM. NFC/case/punctuation normalized. CER counts Unicode codepoints and ignores spaces. Raw transcripts and unexpanded-hypothesis scores are retained.",
    )
    write_json(path, data)
print("Saved transcripts rescored; no recognition or audio was changed.")
