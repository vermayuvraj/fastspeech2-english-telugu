"""Resume official MMS-FA weights using validated HTTP byte ranges.

The URL, byte size and S3 ETag identify the exact TorchAudio MMS-FA asset.
No mirror, alternate checkpoint or untrusted pickle loader is used.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import time
import urllib.request

URL = "https://dl.fbaipublicfiles.com/mms/torchaudio/ctc_alignment_mling_uroman/model.pt"
SIZE = 1262047414
ETAG = '"97a1af3bb69ad1fcade02350fe0453d0-151"'


def download(directory: Path, workers=6):
    target = directory / "cache/torch/hub/checkpoints/model.pt"
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_file() and target.stat().st_size == SIZE:
        print("MMS-FA weights already downloaded", flush=True)
        return target
    parts = directory / "cache/mms_parts"
    parts.mkdir(parents=True, exist_ok=True)
    chunk = 8 * 1024 * 1024
    count = (SIZE + chunk - 1) // chunk

    def fetch(index):
        start, end = index * chunk, min((index + 1) * chunk, SIZE) - 1
        path = parts / f"{index:04d}.part"
        expected = end - start + 1
        if path.is_file() and path.stat().st_size == expected:
            return expected
        for attempt in range(4):
            try:
                request = urllib.request.Request(URL, headers={"Range": f"bytes={start}-{end}", "If-Match": ETAG})
                with urllib.request.urlopen(request, timeout=60) as response:
                    if response.status != 206 or response.headers.get("Content-Range") != f"bytes {start}-{end}/{SIZE}":
                        raise RuntimeError("Server did not return the requested byte range")
                    if response.headers.get("ETag") != ETAG:
                        raise RuntimeError("Official model asset ETag changed")
                    temporary = path.with_suffix(".tmp")
                    with temporary.open("wb") as stream:
                        copied = 0
                        while copied < expected:
                            block = response.read(min(256 * 1024, expected - copied))
                            if not block:
                                raise RuntimeError("Truncated model range")
                            stream.write(block)
                            copied += len(block)
                    os.replace(temporary, path)
                return expected
            except Exception:
                if attempt == 3:
                    raise
                time.sleep(attempt + 1)

    done = 0
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for future in as_completed([executor.submit(fetch, i) for i in range(count)]):
            done += future.result()
            print(f"Official MMS-FA: {done / SIZE * 100:.1f}% ({done / 1e6:.0f} MB)", flush=True)
    temporary = target.with_suffix(".assembling")
    digest = hashlib.sha256()
    with temporary.open("wb") as output:
        for index in range(count):
            with (parts / f"{index:04d}.part").open("rb") as source:
                for block in iter(lambda: source.read(1024 * 1024), b""):
                    output.write(block)
                    digest.update(block)
    if temporary.stat().st_size != SIZE:
        raise RuntimeError("Assembled model size mismatch")
    os.replace(temporary, target)
    target.with_suffix(".provenance.json").write_text(json.dumps({"source": URL, "bytes": SIZE, "etag": ETAG, "sha256": digest.hexdigest()}, indent=2) + "\n")
    print("MMS-FA ready. SHA256:", digest.hexdigest(), flush=True)
    return target


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    if not 1 <= args.workers <= 12:
        parser.error("--workers must be between 1 and 12")
    download(args.work_dir.resolve(), args.workers)
