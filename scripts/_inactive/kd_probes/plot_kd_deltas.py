#!/usr/bin/env python3
"""Plot per-image KD delta distributions."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default="results/kd_analysis/per_image_deltas.csv")
    p.add_argument("--output", default="results/kd_analysis/delta_boxplot.png")
    args = p.parse_args()

    root = Path(__file__).resolve().parents[1]
    rows = []
    with (root / args.csv).open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    datasets = ["Set5", "Set14", "BSD100", "Urban100"]
    data_psnr = []
    labels = []
    for ds in datasets:
        vals = [float(r["delta_psnr"]) for r in rows if r["dataset"] == ds]
        if vals:
            data_psnr.append(vals)
            labels.append(ds)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].boxplot(data_psnr, labels=labels)
    axes[0].axhline(0, color="gray", linestyle="--", linewidth=0.8)
    axes[0].set_ylabel("ΔPSNR (KD − base)")
    axes[0].set_title("Per-image PSNR gain from KD")

    data_lpips = []
    for ds in datasets:
        vals = [float(r["delta_lpips"]) for r in rows if r["dataset"] == ds]
        if vals:
            data_lpips.append(vals)
    axes[1].boxplot(data_lpips, labels=labels)
    axes[1].axhline(0, color="gray", linestyle="--", linewidth=0.8)
    axes[1].set_ylabel("ΔLPIPS (KD − base, lower is better)")
    axes[1].set_title("Per-image LPIPS change from KD")

    out = root / args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
