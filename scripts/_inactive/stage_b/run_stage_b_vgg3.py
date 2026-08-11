#!/usr/bin/env python3
"""Run Stage B vgg_relu3 probe sequentially (one KD job at a time on GPU).

Uses measured throughput: 2 concurrent KD jobs ~2.9x slower — sequential ~0.7h for 2 runs.

Example:
  python scripts/stage_b_vgg3.py resume    # pause / resume / watch
  python scripts/run_stage_b_vgg3.py --skip-done   # called by resume
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable
if (Path("/home/hyb/miniforge3/envs/cv_env/bin/python")).exists():
    PYTHON = "/home/hyb/miniforge3/envs/cv_env/bin/python"

MANIFEST = PROJECT_ROOT / "results/exp_runs/stage_b_vgg3_manifest.json"
LOG_DIR = PROJECT_ROOT / "results/exp_runs/logs"
TRAIN_SCRIPT = PROJECT_ROOT / "scripts/train_mobile_srnet_kd.py"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", default=str(MANIFEST))
    p.add_argument("--skip-done", action="store_true", help="Skip runs that reached target epoch")
    p.add_argument("--run-id", default=None, help="Run only this run_id from manifest")
    return p.parse_args()


def is_done(entry: dict) -> bool:
    log_path = PROJECT_ROOT / "results/exp_runs" / entry["run_id"] / "train_log.jsonl"
    ckpt = PROJECT_ROOT / "results/exp_runs" / entry["run_id"] / "checkpoints/best.pt"
    if not log_path.exists() or not ckpt.exists():
        return False
    rows = [json.loads(l) for l in log_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    if not rows:
        return False
    return int(rows[-1]["epoch"]) >= int(entry["epochs"])


def main() -> None:
    args = parse_args()
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    entries = manifest
    if args.run_id:
        entries = [e for e in manifest if e["run_id"] == args.run_id]
        if not entries:
            raise SystemExit(f"run_id not in manifest: {args.run_id}")

    print("Stage B vgg_relu3 — sequential execution (1 GPU job at a time)")
    for entry in entries:
        run_id = entry["run_id"]
        if args.skip_done and is_done(entry):
            print(f"[skip] {run_id} already complete")
            continue

        cfg = entry["config"]
        ckpt_latest = PROJECT_ROOT / "results/exp_runs" / run_id / "checkpoints/latest.pt"
        cmd = [
            PYTHON,
            str(TRAIN_SCRIPT),
            "--config",
            cfg,
            "--lambda-kd",
            str(entry["lambda_kd"]),
            "--kd-method",
            entry.get("kd_method", "vgg_relu3"),
        ]
        if ckpt_latest.exists() and not is_done(entry):
            cmd += ["--resume-from", str(ckpt_latest.relative_to(PROJECT_ROOT))]
            log_mode = "a"
        else:
            log_mode = "w"

        log_path = LOG_DIR / f"train_{run_id}.log"
        print(f"\n[start] {run_id}")
        print(" ".join(cmd))
        with log_path.open(log_mode, encoding="utf-8") as log:
            if log_mode == "a":
                log.write(f"\n--- resume {time.strftime('%Y-%m-%d %H:%M:%S')} ---\n")
            proc = subprocess.run(cmd, cwd=PROJECT_ROOT, stdout=log, stderr=subprocess.STDOUT)
        if proc.returncode != 0:
            print(f"[FAIL] {run_id} exit={proc.returncode} — see {log_path}")
            sys.exit(proc.returncode)
        print(f"[done] {run_id}")

    print("\nStage B complete. Compare val_psnr in train logs or run eval_exp_runs.py")


if __name__ == "__main__":
    main()
