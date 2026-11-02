"""Read-only loopback progress viewer for the existing experiment."""

from __future__ import annotations

import argparse
import ctypes
import json
import os
from pathlib import Path
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ROOT = Path(__file__).resolve().parents[1]


def read_json(path, default=None):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {} if default is None else default


def alive(pid):
    if not pid:
        return False
    if os.name == "nt":
        kernel = ctypes.windll.kernel32
        kernel.OpenProcess.restype = ctypes.c_void_p
        handle = kernel.OpenProcess(0x1000, False, int(pid))
        if not handle:
            return False
        code = ctypes.c_ulong()
        try:
            return bool(kernel.GetExitCodeProcess(ctypes.c_void_p(handle), ctypes.byref(code))) and code.value == 259
        finally:
            kernel.CloseHandle(ctypes.c_void_p(handle))
    try:
        os.kill(int(pid), 0)
        return True
    except OSError:
        return False


def language_state(language, work):
    result = ROOT / "Results" / language
    rows = []
    log = result / "losses.jsonl"
    if log.is_file():
        for line in log.read_text(encoding="utf-8").splitlines():
            try:
                rows.append(json.loads(line))
            except ValueError:
                pass
    training = [row for row in rows if row.get("split") == "train"]
    validation = [row for row in rows if row.get("split") == "validation"]
    summary = read_json(result / "training_summary.json")
    data = read_json(work / language / "data_summary.json")
    inference = read_json(result / "inference_summary.json")
    provenance = read_json(result / "run_provenance.json")
    maximum = provenance.get("config", {}).get("training", {}).get("max_steps", 40000)
    step = summary.get("completed_steps", training[-1]["step"] if training else 0)
    seconds_per_step = None
    if len(training) > 1:
        last = training[-1]
        for first in training[-21:-1]:
            dt = last["elapsed_seconds_this_session"] - first["elapsed_seconds_this_session"]
            ds = last["step"] - first["step"]
            if dt > 0 and ds > 0:
                seconds_per_step = dt / ds
                break
    stride = max(1, len(training) // 400)
    chart_train = training[::stride]
    if training and (not chart_train or chart_train[-1] != training[-1]):
        chart_train.append(training[-1])
    return {
        "language": language,
        "step": step,
        "maximum_steps": maximum,
        "training_complete": bool(summary),
        "stopped_early": summary.get("stopped_early", False),
        "training_loss": training[-1].get("total") if training else None,
        "validation_loss": validation[-1].get("total") if validation else None,
        "best_validation": min((row["total"] for row in validation), default=None),
        "selected_checkpoint_step": read_json(ROOT / language / "checkpoints/best.metadata.json").get("step"),
        "eta_seconds": (maximum - step) * seconds_per_step if seconds_per_step and not summary else None,
        "last_output_at": log.stat().st_mtime if log.is_file() else None,
        "data_hours": {key: value / 3600 for key, value in data.get("duration_seconds", {}).items()},
        "data_ready": data.get("hours_requirement_met", False),
        "export_ready": (work / language / "candidates.jsonl").is_file(),
        "audio_count": len(list((result / "test_generated_audios").glob("*.wav"))),
        "plot_count": len(list((result / "variance_adaptor_plots").rglob("*.png"))),
        "synthesis_complete": bool(inference),
        "evaluation_complete": (result / "objective_audio_evaluation.json").is_file(),
        "chart_train": [[row["step"], row["total"]] for row in chart_train],
        "chart_validation": [[row["step"], row["total"]] for row in validation],
    }


def snapshot(work):
    status = read_json(ROOT / "STATUS.json")
    fallback_stage = {
        "experiments_finished_review_required": "Experiments complete: final review remains",
        "failed": "The pipeline stopped: check its error log",
    }.get(status.get("state"), "Waiting for experiment status")
    logs = sorted(work.glob("english_run*.stdout.log"), key=lambda p: p.stat().st_mtime, reverse=True)
    lines = logs[0].read_text(encoding="utf-8", errors="replace").splitlines()[-22:] if logs else []
    return {
        "now": time.time(),
        "state": status.get("state", "unknown"),
        "stage": status.get("stage", fallback_stage),
        "process_alive": alive(status.get("pid")),
        "english": language_state("English", work),
        "telugu": language_state("Telugu", work),
        "report_updated_at": (ROOT / "Report/Technical_Report.pdf").stat().st_mtime
        if (ROOT / "Report/Technical_Report.pdf").is_file()
        else None,
        "submission_ready": read_json(ROOT / "Report/submission_audit.json").get("ready_for_submission", False),
        "logs": lines,
    }


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.headers.get("Host", "").split(":", 1)[0] not in {"127.0.0.1", "localhost"}:
            self.send_error(403)
            return
        path = self.path.split("?", 1)[0]
        if path == "/api/status":
            body = json.dumps(snapshot(self.server.work)).encode()
            kind = "application/json; charset=utf-8"
        elif path in {"/", "/index.html"}:
            body = (Path(__file__).parent / "index.html").read_bytes()
            kind = "text/html; charset=utf-8"
        elif path == "/report.pdf":
            body = (ROOT / "Report/Technical_Report.pdf").read_bytes()
            kind = "application/pdf"
        elif path == "/favicon.ico":
            self.send_response(204)
            self.end_headers()
            return
        else:
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_):
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    server.work = args.work_dir.resolve()
    print(f"Local progress: http://127.0.0.1:{server.server_port}", flush=True)
    server.serve_forever()
