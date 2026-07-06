#!/usr/bin/env python3
"""Compare Stage B Plus probe vs Base reference at 2k updates."""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REF_PATH = PROJECT_ROOT / "results/exp_runs/stage_b_plus_reference.json"
PLUS_LOG = PROJECT_ROOT / "results/exp_runs/mobile_srnet_plus_2k/train_log.jsonl"
OUT_PATH = PROJECT_ROOT / "results/exp_runs/stage_b_plus_summary.json"


def main() -> int:
    ref = json.loads(REF_PATH.read_text(encoding="utf-8"))
    if not PLUS_LOG.exists():
        print(f"No Plus log yet: {PLUS_LOG}")
        return 1

    rows = [json.loads(l) for l in PLUS_LOG.read_text(encoding="utf-8").splitlines() if l.strip()]
    if not rows:
        print("Plus train log is empty")
        return 1

    best = max(rows, key=lambda r: r.get("val_psnr", -1))
    last = rows[-1]
    baseline = float(ref["baseline_val_psnr_at_2k"])
    margin = float(ref["pass_margin_db"])
    plus_psnr = float(last["val_psnr"])
    delta = plus_psnr - baseline
    passed = delta >= margin

    summary = {
        "baseline_val_psnr_at_2k": baseline,
        "plus_last_val_psnr": plus_psnr,
        "plus_best_val_psnr": float(best["val_psnr"]),
        "plus_best_epoch": int(best["epoch"]),
        "plus_last_epoch": int(last["epoch"]),
        "delta_db": delta,
        "pass_margin_db": margin,
        "verdict": "PROCEED" if passed else "REJECT",
        "verdict_reason": (
            f"Plus beats Base by {delta:+.3f} dB (need >={margin:.2f})"
            if passed
            else f"Plus only {delta:+.3f} dB vs Base at 2k (need >={margin:.2f})"
        ),
    }
    OUT_PATH.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print("\n" + "=" * 60)
    print("STAGE B CAPACITY PROBE — MobileSRNet-Plus vs Base @ 2k")
    print("=" * 60)
    print(f"  Base reference (20k run @ epoch 61): {baseline:.4f} dB")
    print(f"  Plus last val_psnr (epoch {last['epoch']}): {plus_psnr:.4f} dB")
    print(f"  Plus best val_psnr (epoch {best['epoch']}): {best['val_psnr']:.4f} dB")
    print(f"  Δ(Plus − Base): {delta:+.3f} dB  (pass if ≥ {margin:.2f})")
    print(f"  Verdict: {summary['verdict']} — {summary['verdict_reason']}")
    if passed:
        print("\n  → Eligible for fair-budget mobile_srnet_plus_20k run")
    else:
        print("\n  → Do not commit 20k Plus training; architecture may be saturated")
    print(f"\nWrote {OUT_PATH}")
    return 0 if passed else 2


if __name__ == "__main__":
    sys.exit(main())
