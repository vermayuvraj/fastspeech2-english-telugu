"""Run the automatic evaluation pipeline without publishing or submitting it."""

import argparse
import os
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, required=True)
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    root = Path(__file__).resolve().parents[1]
    work = str(args.work_dir.resolve())
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    stages = [
        ["scripts/prepare_evaluation_models.py", "--work-dir", work],
        ["scripts/automatic_assessment.py", "asr", "--work-dir", work],
        ["scripts/automatic_assessment.py", "quality", "--work-dir", work],
        ["scripts/automatic_assessment.py", "acoustics", "--work-dir", work],
        ["scripts/mms_crosscheck.py", "--work-dir", work],
        ["scripts/plot_automatic_assessment.py"],
        ["scripts/build_automatic_review.py"],
        ["scripts/update_assessment_indexes.py"],
    ]
    for stage in stages:
        print("Running", " ".join(stage), flush=True)
        subprocess.run([sys.executable, *stage], cwd=root, env=env, check=True)
    print("Automatic evidence is complete. The selected report and Google Form remain unchanged.", flush=True)


if __name__ == "__main__":
    main()
