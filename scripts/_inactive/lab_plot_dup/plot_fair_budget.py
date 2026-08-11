#!/usr/bin/env python3
"""Plot validation PSNR vs global_step from experiment train logs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", default="results/exp_runs/fair_budget_manifest.json")
    p.add_argument("--baseline-logs", nargs="*", default=[
        "results/fsrcnn_fix_clean/train_log.jsonl",
        "results/mobile_srnet/train_log.jsonl",
    ])
    p.add_argument("--output", default="results/exp_runs/fair_budget_curves.png")
    return p.parse_args()


def load_curve(log_path: Path) -> tuple[list[int], list[float]]:
    steps, psnrs = [], []
    if not log_path.exists():
        return steps, psnrs
    for line in log_path.read_text(encoding="utf-8").strip().splitlines():
        row = json.loads(line)
        steps.append(int(row["global_step"]))
        psnrs.append(float(row["val_psnr"]))
    return steps, psnrs


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    fig, ax = plt.subplots(figsize=(10, 6))

    for log_rel in args.baseline_logs:
        steps, psnrs = load_curve(root / log_rel)
        if steps:
            label = Path(log_rel).parent.name + " (baseline)"
            ax.plot(steps, psnrs, label=label, alpha=0.7, linestyle="--")

    manifest_path = root / args.manifest
    if manifest_path.exists():
        with manifest_path.open("r", encoding="utf-8") as f:
            manifest = json.load(f)
        for entry in manifest:
            log_path = root / "results/exp_runs" / entry["run_id"] / "train_log.jsonl"
            steps, psnrs = load_curve(log_path)
            if steps:
                ax.plot(steps, psnrs, label=entry["run_id"])

    ax.set_xlabel("Global training steps")
    ax.set_ylabel("DIV2K-valid PSNR (dB)")
    ax.set_title("Fair-budget training curves")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    out = root / args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
