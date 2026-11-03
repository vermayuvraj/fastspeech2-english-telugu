"""Preserve a user-supplied report when experiment scripts are rerun."""

import hashlib
import json
from pathlib import Path


def preserve_selected_report():
    root = Path(__file__).resolve().parents[1]
    status_path = root / "Report/report_status.json"
    if not status_path.is_file():
        return False
    status = json.loads(status_path.read_text(encoding="utf-8"))
    if status.get("source") not in {"user_supplied_pdf", "automatic_evaluation_revision_of_user_report"}:
        return False
    report = root / "Report/Technical_Report.pdf"
    if not report.is_file() or hashlib.sha256(report.read_bytes()).hexdigest() != status["sha256"]:
        raise ValueError("The selected report is missing or changed. Restore the selected PDF or record an explicitly reviewed replacement.")
    print("Preserving the selected Report/Technical_Report.pdf.")
    return True
