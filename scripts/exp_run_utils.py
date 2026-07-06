"""Shared helpers for fair-budget experiment run status and completion checks."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = PROJECT_ROOT / "results/exp_runs/fair_budget_manifest.json"
EXP_RUNS_DIR = PROJECT_ROOT / "results/exp_runs"


def load_manifest() -> list[dict[str, Any]]:
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def target_epochs(entry: dict[str, Any]) -> int:
    cfg_path = PROJECT_ROOT / entry["config"]
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return int(cfg["train"]["epochs"])


def read_train_log(run_id: str) -> list[dict[str, Any]]:
    log_path = EXP_RUNS_DIR / run_id / "train_log.jsonl"
    if not log_path.exists():
        return []
    rows = []
    with log_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def is_run_done(entry: dict[str, Any]) -> bool:
    """True when the final target epoch has been logged (not merely when best.pt exists)."""
    rows = read_train_log(entry["run_id"])
    if not rows:
        return False
    ckpt = EXP_RUNS_DIR / entry["run_id"] / "checkpoints/best.pt"
    if not ckpt.exists():
        return False
    return int(rows[-1]["epoch"]) >= target_epochs(entry)


def running_run_ids() -> set[str]:
    """Detect active trainers by matching manifest config paths in process list."""
    try:
        proc = subprocess.check_output(["pgrep", "-af", "scripts/train_"], text=True)
    except subprocess.CalledProcessError:
        return set()

    active: set[str] = set()
    for entry in load_manifest():
        cfg = entry["config"]
        if cfg in proc:
            active.add(entry["run_id"])
    return active


def count_running_kd() -> int:
    return sum(1 for rid in running_run_ids() if "kd" in rid)


def gpu_snapshot() -> dict[str, str] | None:
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=memory.used,memory.total,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    parts = [p.strip() for p in out.split(",")]
    if len(parts) != 3:
        return None
    used_mib, total_mib, util = parts
    return {
        "memory_used_mib": used_mib,
        "memory_total_mib": total_mib,
        "utilization_pct": util,
    }


def run_status(entry: dict[str, Any]) -> dict[str, Any]:
    run_id = entry["run_id"]
    rows = read_train_log(run_id)
    target_ep = target_epochs(entry)
    running = run_id in running_run_ids()
    done = is_run_done(entry)

    status = {
        "run_id": run_id,
        "model": entry.get("model"),
        "lambda_kd": entry.get("lambda_kd"),
        "updates_target": entry.get("updates_target"),
        "target_epochs": target_ep,
        "done": done,
        "running": running,
        "epoch": None,
        "global_step": None,
        "val_psnr": None,
        "best_val_psnr": None,
        "best_epoch": None,
        "lr": None,
        "elapsed_sec": None,
        "progress_pct": 0.0,
        "has_checkpoint": (EXP_RUNS_DIR / run_id / "checkpoints/best.pt").exists(),
        "has_benchmark_eval": (EXP_RUNS_DIR / run_id / "benchmark_metrics.json").exists(),
    }

    if rows:
        last = rows[-1]
        best = max(rows, key=lambda r: r.get("val_psnr", float("-inf")))
        epoch = int(last["epoch"])
        status.update(
            {
                "epoch": epoch,
                "global_step": int(last.get("global_step", 0)),
                "val_psnr": float(last.get("val_psnr", float("nan"))),
                "best_val_psnr": float(best.get("val_psnr", float("nan"))),
                "best_epoch": int(best.get("epoch", 0)),
                "lr": last.get("lr"),
                "elapsed_sec": last.get("elapsed_sec"),
                "progress_pct": round(100.0 * epoch / target_ep, 1),
            }
        )

    if done:
        status["state"] = "done"
    elif running:
        status["state"] = "running"
    elif rows:
        status["state"] = "paused"
    else:
        status["state"] = "pending"

    return status


def all_run_statuses() -> list[dict[str, Any]]:
    return [run_status(e) for e in load_manifest()]


def pending_run_ids() -> list[str]:
    return [s["run_id"] for s in all_run_statuses() if not s["done"]]


def summary_dict() -> dict[str, Any]:
    statuses = all_run_statuses()
    gpu = gpu_snapshot()
    done = sum(1 for s in statuses if s["done"])
    running = [s["run_id"] for s in statuses if s["running"]]
    pending = [s["run_id"] for s in statuses if not s["done"]]
    return {
        "total_runs": len(statuses),
        "done": done,
        "running": running,
        "pending": pending,
        "kd_running": count_running_kd(),
        "gpu": gpu,
        "runs": statuses,
    }
