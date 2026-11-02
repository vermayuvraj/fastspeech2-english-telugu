"""Independent FS2 training, real validation, resumable state and loss plots."""
from __future__ import annotations

import argparse
import json
import math
import os
import platform
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .config import ROOT, load_config, seed_everything, sha256, workspace, write_json
from .data import AcousticDataset, BucketBatchSampler, collate, forward_batch
from .frontend import SYMBOLS
from .loss import denominators, loss_terms
from .model import create_model


def to_device(batch, device):
    return {k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}


@torch.inference_mode()
def validate(model, loader, device):
    model.eval()
    sums = {key: 0.0 for key in ["mel", "postnet", "duration", "pitch", "energy"]}
    counts = dict.fromkeys(sums, 0)
    for batch in loader:
        batch = to_device(batch, device)
        terms = loss_terms(batch, forward_batch(model, batch))
        denoms = denominators(batch)
        for key in sums:
            sums[key] += float(terms[key]) * denoms[key]
            counts[key] += denoms[key]
    result = {key: sums[key] / counts[key] for key in sums}
    result["total"] = sum(result.values())
    if not all(math.isfinite(v) for v in result.values()):
        raise RuntimeError("Non-finite validation loss")
    model.train()
    return result


def atomic_save(value, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    torch.save(value, temp)
    os.replace(temp, path)


def plot_losses(log_path: Path, output: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines() if line]
    output.mkdir(parents=True, exist_ok=True)
    for split, name in [("train", "training_loss.png"), ("validation", "validation_loss.png")]:
        selected = [r for r in rows if r["split"] == split]
        if not selected:
            continue
        fig, ax = plt.subplots(figsize=(9, 4.8), constrained_layout=True)
        for key in ["total", "mel", "postnet", "duration", "pitch", "energy"]:
            ax.plot([r["step"] for r in selected], [r[key] for r in selected], label=key, linewidth=1.4)
        ax.set(xlabel="Optimizer step", ylabel="Loss (native objective units)", title=f"{split.capitalize()} loss | {selected[0]['language']}")
        ax.grid(alpha=0.2)
        ax.legend(ncol=3)
        fig.savefig(output / name, dpi=160)
        plt.close(fig)


def train(config: dict, directory: Path, results: Path, checkpoints: Path, device: str, steps: int | None, resume: bool, smoke: bool = False):
    seed_everything(config["seed"])
    torch.set_num_threads(config["training"]["cpu_threads"])
    device = torch.device(device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; use --device cpu explicitly for a small diagnostic run")
    summary = json.loads((directory / "data_summary.json").read_text(encoding="utf-8"))
    if summary.get("config_sha256") != config["_config_sha256"]:
        raise RuntimeError("Preprocessing used a different configuration")
    if not summary["hours_requirement_met"] and not smoke:
        raise RuntimeError("The required 5 h / 1 h split has not been prepared. Use --smoke only for an isolated diagnostic.")
    stats = json.loads((directory / "stats.json").read_text(encoding="utf-8"))
    speakers = json.loads((directory / "speakers.json").read_text(encoding="utf-8"))
    model = create_model(config, stats, speakers).to(device)
    train_data = AcousticDataset(directory, "train", stats, speakers)
    val_data = AcousticDataset(directory, "validation", stats, speakers)
    setting = config["training"]
    sampler = BucketBatchSampler(train_data.rows, setting["batch_size"], config["seed"], True)
    val_sampler = BucketBatchSampler(val_data.rows, setting["validation_batch_size"], config["seed"], False)
    loader = DataLoader(train_data, batch_sampler=sampler, collate_fn=collate, num_workers=0, pin_memory=device.type == "cuda")
    val_loader = DataLoader(val_data, batch_sampler=val_sampler, collate_fn=collate, num_workers=0, pin_memory=device.type == "cuda")
    optimizer = torch.optim.AdamW(model.parameters(), lr=setting["learning_rate"], betas=(0.9, 0.98), eps=1e-9, weight_decay=setting["weight_decay"])
    amp = device.type == "cuda" and setting["mixed_precision"]
    amp_dtype = torch.bfloat16 if amp and torch.cuda.is_bf16_supported() else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=amp and amp_dtype == torch.float16, init_scale=1024)
    log_path = results / "losses.jsonl"
    state_path = directory / "last_training_state.pt"
    results.mkdir(parents=True, exist_ok=True)
    checkpoints.mkdir(parents=True, exist_ok=True)
    maximum = steps if steps is not None else setting["max_steps"]
    step, epoch, batch_offset, best, stale = 0, 0, 0, math.inf, 0
    if resume and state_path.is_file():
        # This file is generated locally and includes Python/NumPy RNG state.
        state = torch.load(state_path, map_location="cpu", weights_only=False)
        if state["config_sha256"] != config["_config_sha256"]:
            raise RuntimeError("Refusing to resume with a different configuration")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        for values in optimizer.state.values():
            for key, value in values.items():
                if isinstance(value, torch.Tensor):
                    values[key] = value.to(device)
        scaler.load_state_dict(state["scaler"])
        step, epoch, batch_offset, best, stale = (state[k] for k in ["step", "epoch", "batch_offset", "best", "stale"])
        random.setstate(state["random_state"])
        np.random.set_state(state["numpy_state"])
        torch.set_rng_state(state["torch_state"])
        if device.type == "cuda":
            torch.cuda.set_rng_state_all(state["cuda_state"])
        # Remove records newer than the saved optimizer state after an interruption.
        if log_path.exists():
            lines = [line for line in log_path.read_text(encoding="utf-8").splitlines() if json.loads(line)["step"] <= step]
            log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    elif log_path.is_file():
        raise RuntimeError("Existing experiment log found. Use --resume or a new work directory.")
    provenance = {"language": config["language"], "config": config, "config_sha256": config["_config_sha256"], "data_summary": summary, "dataset_revision": config["dataset"]["revision"], "seed": config["seed"], "torch": torch.__version__, "python": platform.python_version(), "platform": platform.platform(), "cuda": torch.version.cuda, "device": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu", "parameters": sum(p.numel() for p in model.parameters() if p.requires_grad), "mixed_precision": amp, "deterministic_algorithms": False, "smoke": smoke, "validation_protocol": "entire held-out set, teacher-forced duration/pitch/energy; loss weighted by valid element count"}
    write_json(results / "run_provenance.json", provenance)
    print(f"{config['language']}: {provenance['parameters']:,} parameters, {device}, {len(train_data)} train / {len(val_data)} validation", flush=True)
    start = time.monotonic()
    log_interval = min(setting["log_interval"], maximum)
    validation_interval = min(setting["validation_interval"], maximum)
    model.train()
    window = []
    stopped_early = False
    while step < maximum:
        sampler.epoch = epoch
        for batch_index, batch in enumerate(loader):
            if batch_index < batch_offset:
                continue
            batch = to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            lr = setting["learning_rate"] * min((step + 1) / setting["warmup_steps"], math.sqrt(setting["warmup_steps"] / (step + 1)))
            for group in optimizer.param_groups:
                group["lr"] = lr
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp):
                prediction = forward_batch(model, batch)
                terms = loss_terms(batch, prediction)
            if not torch.isfinite(terms["total"]):
                raise RuntimeError(f"Non-finite loss at step {step + 1}; utterances={batch['ids']}")
            scaler.scale(terms["total"]).backward()
            scaler.unscale_(optimizer)
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), setting["gradient_clip"])
            if not torch.isfinite(norm):
                raise RuntimeError(f"Non-finite gradient at step {step + 1}")
            scaler.step(optimizer)
            scaler.update()
            step += 1
            batch_offset = batch_index + 1
            window.append({key: float(value.detach()) for key, value in terms.items()})
            if step % log_interval == 0 or step == maximum:
                metrics = {key: sum(r[key] for r in window) / len(window) for key in window[0]}
                row = {"step": step, "epoch": epoch, "split": "train", "language": config["language"], "lr": lr, "gradient_norm": float(norm), "elapsed_seconds_this_session": time.monotonic() - start, **metrics}
                with log_path.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(row) + "\n")
                print(f"step {step}/{maximum}: train={metrics['total']:.4f}, mel={metrics['mel']:.4f}, lr={lr:.3g}", flush=True)
                window.clear()
            if step % validation_interval == 0 or step == maximum:
                metrics = validate(model, val_loader, device)
                with log_path.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps({"step": step, "epoch": epoch, "split": "validation", "language": config["language"], **metrics}) + "\n")
                improved = metrics["total"] < best - setting["minimum_improvement"]
                if improved:
                    best, stale = metrics["total"], 0
                    export = {"format_version": 1, "language": config["language"], "step": step, "config": config, "stats": stats, "speakers": speakers, "symbols": SYMBOLS, "validation_loss": metrics, "data_summary": summary, "model": {k: v.detach().cpu() for k, v in model.state_dict().items()}, "smoke": smoke}
                    atomic_save(export, checkpoints / "best.pt")
                    write_json(checkpoints / "best.metadata.json", {k: v for k, v in export.items() if k != "model"})
                else:
                    stale += 1
                state = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(), "step": step, "epoch": epoch, "batch_offset": batch_offset, "best": best, "stale": stale, "config_sha256": config["_config_sha256"], "random_state": random.getstate(), "numpy_state": np.random.get_state(), "torch_state": torch.get_rng_state(), "cuda_state": torch.cuda.get_rng_state_all() if device.type == "cuda" else []}
                atomic_save(state, state_path)
                plot_losses(log_path, results)
                print(f"validation={metrics['total']:.4f}; best={best:.4f}; no-improvement checks={stale}", flush=True)
                if step >= setting["min_steps"] and stale >= setting["early_stop_patience"]:
                    stopped_early = True
                    break
            if step >= maximum:
                break
        if stopped_early:
            break
        epoch += 1
        batch_offset = 0
    plot_losses(log_path, results)
    write_json(results / "training_summary.json", {"language": config["language"], "completed_steps": step, "best_validation_total": best, "stopped_early": stopped_early, "elapsed_seconds_this_session": time.monotonic() - start, "checkpoint_sha256": sha256(checkpoints / "best.pt"), "smoke": smoke})
    return model


def main(default_language: str | None = None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / default_language / "config.yaml" if default_language else None, required=not bool(default_language))
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--steps", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    if default_language and config["language"] != default_language:
        parser.error("Configuration language mismatch")
    if args.steps is not None and args.steps < 1:
        parser.error("--steps must be positive")
    directory = workspace(config, args.work_dir)
    if args.smoke:
        if "smoke" not in str(args.work_dir).lower():
            parser.error("Use a directory containing 'smoke'")
        results, checkpoints = directory / "Results", directory / "checkpoints"
    else:
        results, checkpoints = ROOT / "Results" / config["language"], ROOT / config["language"] / "checkpoints"
    train(config, directory, results, checkpoints, args.device, args.steps, args.resume, args.smoke)


if __name__ == "__main__":
    main()
