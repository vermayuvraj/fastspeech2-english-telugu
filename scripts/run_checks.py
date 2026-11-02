"""Run offline verification and save exact command outcomes for the report."""
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


if __name__ == "__main__":
    results = []
    for command in [[sys.executable, "-m", "pytest", "-q"], [sys.executable, "-m", "ruff", "check", "."], [sys.executable, "-m", "compileall", "-q", "fs2", "English", "Telugu", "scripts"]]:
        start = time.monotonic()
        completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
        print(completed.stdout, end="")
        print(completed.stderr, end="", file=sys.stderr)
        results.append({"command": command[1:], "returncode": completed.returncode, "seconds": time.monotonic() - start, "stdout": completed.stdout, "stderr": completed.stderr})
    target = ROOT / "Report/verification.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps({"all_passed": all(r["returncode"] == 0 for r in results), "checks": results}, indent=2) + "\n")
    raise SystemExit(0 if all(r["returncode"] == 0 for r in results) else 1)
