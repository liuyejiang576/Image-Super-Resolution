#!/usr/bin/env python3
"""Run fair-budget training jobs in parallel on a single GPU."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYTHON = "/home/hyb/miniforge3/envs/cv_env/bin/python"
LOG_DIR = PROJECT_ROOT / "results" / "exp_runs" / "logs"
SCRIPT_DIR = Path(__file__).resolve().parent

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from exp_run_utils import (  # noqa: E402
    count_running_kd,
    is_run_done,
    load_manifest,
    running_run_ids,
    summary_dict,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--max-parallel", type=int, default=3)
    p.add_argument("--max-kd-parallel", type=int, default=2)
    p.add_argument("--poll-sec", type=int, default=30)
    return p.parse_args()


def write_status_snapshot() -> None:
    snapshot = summary_dict()
    out = PROJECT_ROOT / "results/exp_runs/status.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    snapshot["updated_at"] = datetime.now(timezone.utc).isoformat()
    with out.open("w", encoding="utf-8") as f:
        json.dump(snapshot, f, indent=2)


def build_cmd(entry: dict) -> list[str]:
    run_id = entry["run_id"]
    cfg = entry["config"]
    if "fsrcnn" in run_id:
        cmd = [PYTHON, "scripts/train_fsrcnn.py", "--config", cfg]
    elif entry.get("lambda_kd") is not None or "kd" in run_id:
        cmd = [PYTHON, "scripts/train_mobile_srnet_kd.py", "--config", cfg]
        if entry.get("lambda_kd") is not None:
            cmd += ["--lambda-kd", str(entry["lambda_kd"])]
    else:
        cmd = [PYTHON, "scripts/train_mobile_srnet.py", "--config", cfg]
    return cmd


def is_kd(entry: dict) -> bool:
    return entry.get("lambda_kd") is not None or "kd" in entry["run_id"]


def main() -> None:
    args = parse_args()
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest()

    active: dict[str, subprocess.Popen] = {}

    def refresh_active():
        dead = [rid for rid, p in active.items() if p.poll() is not None]
        for rid in dead:
            code = active[rid].returncode
            print(f"[parallel] finished {rid} exit={code}")
            del active[rid]

    def external_running() -> set[str]:
        return running_run_ids() - set(active.keys())

    def already_running(run_id: str) -> bool:
        return run_id in active or run_id in running_run_ids()

    while True:
        refresh_active()

        pending = [
            e for e in manifest
            if not is_run_done(e) and not already_running(e["run_id"])
        ]

        if all(is_run_done(e) for e in manifest):
            print("[parallel] all runs complete")
            write_status_snapshot()
            subprocess.run([PYTHON, "scripts/eval_exp_runs.py"], cwd=PROJECT_ROOT, check=False)
            subprocess.run([PYTHON, "scripts/build_enhanced_report.py"], cwd=PROJECT_ROOT, check=False)
            break

        external = external_running()
        slots = args.max_parallel - len(active) - len(external)
        kd_slots = max(0, args.max_kd_parallel - count_running_kd())

        launched = 0
        for entry in pending:
            if slots <= 0:
                break
            if is_kd(entry) and kd_slots <= 0:
                continue
            run_id = entry["run_id"]
            cmd = build_cmd(entry)
            log_path = LOG_DIR / f"train_{run_id}.log"
            print(f"[parallel] starting {run_id}: {' '.join(cmd)}")
            with log_path.open("w", encoding="utf-8") as log:
                active[run_id] = subprocess.Popen(
                    cmd, cwd=PROJECT_ROOT, stdout=log, stderr=subprocess.STDOUT
                )
            slots -= 1
            launched += 1
            if is_kd(entry):
                kd_slots -= 1

        if launched == 0 and (active or external):
            active_all = sorted(set(active.keys()) | external)
            print(f"[parallel] waiting... active={active_all}")
        elif launched == 0 and not active:
            print("[parallel] nothing to launch but not all done — rechecking")
        write_status_snapshot()
        time.sleep(args.poll_sec)


if __name__ == "__main__":
    main()
