"""Download pinned public evaluation models into the external work directory."""

import argparse
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--models", choices=["all", "asr", "quality", "mms"], default="all")
    args = parser.parse_args()
    work = args.work_dir.resolve()
    os.environ["HF_HUB_DISABLE_XET"] = "1"
    os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "60"
    from huggingface_hub import snapshot_download

    root = Path(__file__).resolve().parents[1]
    revisions = json.loads((root / "docs/automatic_evaluation_models.json").read_text(encoding="utf-8"))
    (work / "evaluation_models.json").write_text(json.dumps(revisions, indent=2) + "\n", encoding="utf-8")
    files = {
        "Systran/faster-whisper-large-v3": [
            "config.json",
            "model.bin",
            "tokenizer.json",
            "preprocessor_config.json",
            "vocabulary.json",
        ],
        "sarulab-speech/UTMOSv2": ["fold0_s42_best_model.pth"],
        "facebook/wav2vec2-base": ["config.json", "preprocessor_config.json", "pytorch_model.bin"],
        "timm/tf_efficientnetv2_s.in21k_ft_in1k": ["model.safetensors"],
        "facebook/mms-1b-all": [
            "model.safetensors",
            "config.json",
            "preprocessor_config.json",
            "vocab.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "adapter.eng.safetensors",
            "adapter.tel.safetensors",
        ],
    }
    for repo, patterns in files.items():
        is_asr = repo.startswith("Systran/") or repo == "facebook/mms-1b-all"
        if args.models == "mms" and repo != "facebook/mms-1b-all":
            continue
        if args.models == "asr" and not is_asr or args.models == "quality" and is_asr:
            continue
        print("Preparing", repo, revisions[repo], flush=True)
        path = Path(
            snapshot_download(
                repo,
                revision=revisions[repo],
                cache_dir=work / "cache/huggingface",
                allow_patterns=patterns,
                max_workers=3,
            )
        )
        # The upstream quality loader uses default branch names. Pin those cache
        # references in this task-owned cache so offline loading is reproducible.
        refs = path.parent.parent / "refs"
        refs.mkdir(exist_ok=True)
        (refs / "main").write_text(revisions[repo], encoding="utf-8")
        print("Ready", repo, flush=True)


if __name__ == "__main__":
    main()
