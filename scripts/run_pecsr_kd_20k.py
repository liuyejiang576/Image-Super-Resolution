#!/usr/bin/env python3
"""Run PECSR matched KD fair-budget training (λ=0 then λ=0.2).

Called by pecsr_kd_20k.py resume. Do not start while MSE families are running
(IMPLEMENTATION §9).

  python scripts/run_pecsr_kd_20k.py --skip-done
  python scripts/run_pecsr_kd_20k.py --skip-done --run-id pecsr_kd0_20k
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
if Path("/home/hyb/miniforge3/envs/cv_env/bin/python").exists():
    PYTHON = "/home/hyb/miniforge3/envs/cv_env/bin/python"

MANIFEST = PROJECT_ROOT / "results/exp_runs/pecsr_kd_20k_manifest.json"
LOG_DIR = PROJECT_ROOT / "results/exp_runs/logs"
TRAIN_SCRIPT = PROJECT_ROOT / "scripts/train_mobile_srnet_kd.py"
DEFAULT_RUN_IDS = ["pecsr_kd0_20k", "pecsr_kd02_20k"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", default=str(MANIFEST))
    p.add_argument("--skip-done", action="store_true")
    p.add_argument("--run-id", default=None, help="Train one run_id.")
    p.add_argument(
        "--all-gated",
        action="store_true",
        help="Train every manifest entry.",
    )
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

    if args.run_id:
        entries = [e for e in manifest if e["run_id"] == args.run_id]
        if not entries:
            raise SystemExit(f"run_id not in manifest: {args.run_id}")
    elif args.all_gated:
        entries = manifest
    else:
        entries = [e for e in manifest if e["run_id"] in DEFAULT_RUN_IDS]
        if not entries:
            raise SystemExit(f"default run ids missing from manifest: {DEFAULT_RUN_IDS}")

    print("PECSR matched KD 20k — training")
    for entry in entries:
        run_id = entry["run_id"]
        if args.skip_done and is_done(entry):
            print(f"[skip] {run_id} already complete")
            continue

        cfg = entry["config"]
        ckpt_latest = PROJECT_ROOT / "results/exp_runs" / run_id / "checkpoints/latest.pt"
        cmd = [PYTHON, str(TRAIN_SCRIPT), "--config", cfg]
        if ckpt_latest.exists() and not is_done(entry):
            cmd += ["--resume-from", str(ckpt_latest.relative_to(PROJECT_ROOT))]
            log_mode = "a"
        else:
            log_mode = "w"

        log_path = LOG_DIR / f"train_{run_id}.log"
        print(f"\n[start] {run_id} variant={entry.get('variant')}")
        print(" ".join(cmd))
        with log_path.open(log_mode, encoding="utf-8") as log:
            if log_mode == "a":
                log.write(f"\n--- resume {time.strftime('%Y-%m-%d %H:%M:%S')} ---\n")
            proc = subprocess.run(cmd, cwd=PROJECT_ROOT, stdout=log, stderr=subprocess.STDOUT)
        if proc.returncode != 0:
            print(f"[FAIL] {run_id} exit={proc.returncode} — see {log_path}")
            sys.exit(proc.returncode)
        print(f"[done] {run_id}")

    print("\nPECSR KD train complete. Next: matched λ table for report (not vs MSE PECSR).")


if __name__ == "__main__":
    main()
