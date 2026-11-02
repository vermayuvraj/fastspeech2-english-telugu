"""Check imports, CUDA, official model access and free storage without exposing secrets."""
import importlib.metadata
import json
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from huggingface_hub import get_hf_file_metadata, get_token, hf_hub_url
from fs2.config import ROOT, load_config


def main():
    report = {"python": sys.version.split()[0], "cuda_available": torch.cuda.is_available(), "free_disk_gb": round(shutil.disk_usage(ROOT).free / 1e9, 1), "packages": {}}
    for package in ["torch", "torchaudio", "numpy", "scipy", "librosa", "pyworld", "soundfile", "matplotlib", "PyYAML", "pyarrow", "huggingface-hub", "uroman", "num2words", "reportlab"]:
        try:
            report["packages"][package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            report["packages"][package] = "MISSING"
    if torch.cuda.is_available():
        report["gpu"] = torch.cuda.get_device_name(0)
        report["gpu_memory_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2)
    report["dataset_access"] = {}
    for language in ["English", "Telugu"]:
        data = load_config(ROOT / language / "config.yaml")["dataset"]
        try:
            metadata = get_hf_file_metadata(hf_hub_url(data["repository"], data["files"][0], repo_type="dataset", revision=data["revision"]), token=get_token(), timeout=15)
            report["dataset_access"][language] = {"accessible": True, "first_file_bytes": metadata.size}
        except Exception as error:
            report["dataset_access"][language] = {"accessible": False, "error_type": type(error).__name__, "next_action": "Accept Svarah conditions and run scripts/sign_in_huggingface.py" if language == "English" else "Check network access to Hugging Face"}
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
