#!/usr/bin/env python3
"""Stage B Plus: watch training progress and print pass/fail verdict.

  python scripts/stage_b_plus.py watch [--interval 60]
  python scripts/stage_b_plus.py verdict
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)

MANIFEST = ROOT / "results/exp_runs/stage_b_plus_manifest.json"
REFERENCE = ROOT / "results/exp_runs/stage_b_plus_reference.json"
RUN_ID = "mobile_srnet_plus_2k"


def read_rows(run_id: str) -> list[dict]:
    log = ROOT / f"results/exp_runs/{run_id}/train_log.jsonl"
    if not log.exists():
        return []
    rows = []
    for line in log.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return rows


def load_reference() -> dict:
    return json.loads(REFERENCE.read_text(encoding="utf-8"))


def load_target_epochs() -> int:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    for e in manifest:
        if e["run_id"] == RUN_ID:
            return int(e["epochs"])
    return 61


def print_verdict() -> int:
    ref = load_reference()
    rows = read_rows(RUN_ID)
    target_epochs = load_target_epochs()
    base_psnr = float(ref["reference_val_psnr"])
    margin = float(ref["pass_margin_db"])

    print("=" * 72)
    print("Stage B MobileSRNet-Plus — verdict")
    print("=" * 72)

    if not rows:
        print(f"No training log for {RUN_ID}. Start: python scripts/run_stage_b_plus.py")
        return 1

    last = rows[-1]
    best = max(rows, key=lambda r: r.get("val_psnr", -1))
    plus_psnr = float(best["val_psnr"])
    delta = plus_psnr - base_psnr
    done = int(last["epoch"]) >= target_epochs

    print(f"Plus best val_psnr:  {plus_psnr:.4f} dB @ epoch {best['epoch']}")
    print(f"Base reference:      {base_psnr:.4f} dB @ ~{ref['reference_updates_approx']} updates")
    print(f"                     ({ref['source']})")
    print(f"Delta (Plus − Base): {delta:+.4f} dB")
    print(f"Progress:            epoch {last['epoch']}/{target_epochs} {'DONE' if done else 'in progress'}")
    print()

    if not done:
        print("Training not finished — verdict provisional.")
        if delta > margin:
            print(f"  Currently ahead by {delta:.3f} dB (pass threshold +{margin:.2f} dB).")
        return 0

    if delta >= margin:
        print(f"PROCEED — Plus beats Base by {delta:.3f} dB (>= +{margin:.2f} dB margin).")
        print("  Next: fair-budget 20k run for mobile_srnet_plus_20k.")
        return 0

    if delta > 0:
        print(f"MARGINAL — Plus +{delta:.3f} dB but below +{margin:.2f} dB margin. Optional 20k; not required.")
        return 0

    print(f"REJECT — Plus did not beat Base at 2k ({delta:+.3f} dB). Skip 20k Plus run.")
    print("  Ship report with Base (mobile_srnet_20k) as headline model.")
    return 0


def watch(interval: int) -> None:
    target = load_target_epochs()
    ref = load_reference()
    print(f"Watching {RUN_ID} (target {target} epochs, base ref {ref['reference_val_psnr']:.4f} dB)")
    print("Ctrl+C to stop.\n")
    try:
        while True:
            rows = read_rows(RUN_ID)
            now = datetime.now().strftime("%H:%M:%S")
            if not rows:
                print(f"[{now}] waiting for train log...")
            else:
                last = rows[-1]
                best = max(rows, key=lambda r: r.get("val_psnr", -1))
                delta = float(best["val_psnr"]) - float(ref["reference_val_psnr"])
                print(
                    f"[{now}] epoch {last['epoch']}/{target} "
                    f"val={last['val_psnr']:.4f} best={best['val_psnr']:.4f} "
                    f"Δvs_base={delta:+.4f} dB"
                )
                if int(last["epoch"]) >= target:
                    print("\nTraining complete.")
                    print_verdict()
                    break
            time.sleep(interval)
    except KeyboardInterrupt:
        print("\nStopped watch.")


def main() -> None:
    ap = argparse.ArgumentParser(description="Stage B Plus: watch | verdict")
    ap.add_argument("command", choices=["watch", "verdict"])
    ap.add_argument("--interval", type=int, default=60)
    args = ap.parse_args()

    if args.command == "watch":
        watch(args.interval)
    else:
        raise SystemExit(print_verdict())


if __name__ == "__main__":
    main()
