#!/usr/bin/env python3
"""Run A1a FSRCNN bs24 fair-budget 20k training.

Called by a1a_20k.py resume. Do not start manually unless debugging.

  python scripts/run_a1a_20k.py --skip-done
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

MANIFEST = PROJECT_ROOT / "results/exp_runs/a1a_20k_manifest.json"
LOG_DIR = PROJECT_ROOT / "results/exp_runs/logs"
TRAIN_SCRIPT = PROJECT_ROOT / "scripts/train_fsrcnn.py"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", default=str(MANIFEST))
    p.add_argument("--skip-done", action="store_true")
    p.add_argument("--run-id", default=None)
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

    print("A1a FSRCNN bs24 20k — training")
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

    print("\nA1a 20k complete. Next: eval best.pt; side-note fair table vs bs=8 FSRCNN.")


if __name__ == "__main__":
    main()
