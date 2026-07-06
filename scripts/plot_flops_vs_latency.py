#!/usr/bin/env python3
"""Scatter plot: FLOPs vs audited latency (RQ3 figure)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--latency-json", default="results/latency_audit/latency_audit.json")
    p.add_argument("--snapshot-json", default="results/exp_runs/baseline_snapshot.json")
    p.add_argument("--plus-profile", default="results/exp_runs/mobile_srnet_plus_20k/profile.json")
    p.add_argument("--fair-budget", default="results/exp_runs/fair_budget_runs.json")
    p.add_argument("--output", default="results/latency_audit/flops_vs_latency.png")
    return p.parse_args()


def load_flops_psnr(snapshot: dict, fair: dict, plus_profile: dict) -> dict[str, dict]:
    """Return name -> {flops_g, avg_psnr} for labeled models."""
    snap_prof = {
        "FSRCNN": ("fsrcnn_fix_clean", "fsrcnn_fix_clean_20k"),
        "FSRCNN-Small": ("fsrcnn_small", None),
        "MobileSRNet-Base": ("mobile_srnet", "mobile_srnet_20k"),
        "SwinIR": ("swinir", None),
    }
    out: dict[str, dict] = {}
    for label, (snap_key, fair_key) in snap_prof.items():
        prof = snapshot.get(snap_key, {}).get("profile", {})
        avg_psnr = None
        if fair_key and fair_key in fair:
            avg_psnr = fair[fair_key].get("avg_psnr")
        elif snap_key in snapshot:
            m = snapshot[snap_key].get("benchmark_metrics", {})
            psnrs = [m[d]["psnr"] for d in m if isinstance(m.get(d), dict)]
            avg_psnr = sum(psnrs) / len(psnrs) if psnrs else None
        out[label] = {"flops_g": prof.get("flops_g"), "avg_psnr": avg_psnr}

    out["MobileSRNet-Plus"] = {
        "flops_g": plus_profile.get("flops_g"),
        "avg_psnr": fair.get("mobile_srnet_plus_20k", {}).get("avg_psnr"),
    }
    return out


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]

    with (root / args.latency_json).open("r", encoding="utf-8") as f:
        latency = json.load(f)
    snapshot = json.loads((root / args.snapshot_json).read_text(encoding="utf-8")) if (root / args.snapshot_json).exists() else {}
    fair = json.loads((root / args.fair_budget).read_text(encoding="utf-8"))
    plus_profile = json.loads((root / args.plus_profile).read_text(encoding="utf-8"))

    meta = load_flops_psnr(snapshot, fair, plus_profile)

    # Map latency JSON keys to display labels
    latency_labels = {
        "FSRCNN": "FSRCNN",
        "FSRCNN-Small": "FSRCNN-Small",
        "MobileSRNet": "MobileSRNet-Base",
        "MobileSRNet-Base": "MobileSRNet-Base",
        "MobileSRNet-Plus": "MobileSRNet-Plus",
        "SwinIR": "SwinIR",
    }

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True)
    titles = ["FP32 latency", "FP16 latency"]
    prec_keys = ["fp32", "fp16"]

    colors = {
        "FSRCNN": "#e45756",
        "FSRCNN-Small": "#f58518",
        "MobileSRNet-Base": "#4c78a8",
        "MobileSRNet-Plus": "#54a24b",
        "SwinIR": "#b279a2",
    }

    for ax, title, pk in zip(axes, titles, prec_keys):
        for lat_name, lat_entry in latency.items():
            if lat_name in ("input_lr", "protocol"):
                continue
            label = latency_labels.get(lat_name, lat_name)
            if label not in meta or meta[label].get("flops_g") is None:
                continue
            prec = lat_entry.get(pk)
            if not isinstance(prec, dict) or "median_ms" not in prec:
                continue
            flops = meta[label]["flops_g"]
            ms = prec["median_ms"]
            psnr = meta[label].get("avg_psnr")
            ax.scatter(flops, ms, s=80, c=colors.get(label, "#333333"), label=label, zorder=3)
            ax.annotate(
                label,
                (flops, ms),
                textcoords="offset points",
                xytext=(4, 4),
                fontsize=7,
            )
        ax.set_xlabel("FLOPs (G)")
        ax.set_ylabel("Median latency (ms)")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.set_xscale("log")

    fig.suptitle(
        f"FLOPs vs latency (LR {latency.get('input_lr', ['?', '?'])[0]}×"
        f"{latency.get('input_lr', ['?', '?'])[1]}, batch=1)",
        fontsize=11,
    )
    fig.tight_layout()
    out = root / args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
