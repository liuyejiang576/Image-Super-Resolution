#!/usr/bin/env python3
"""Freeze current baseline artifacts into results/exp_runs/baseline_snapshot.json."""

from __future__ import annotations

import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def avg_metric(metrics: dict, key: str) -> float | None:
    vals = [v[key] for v in metrics.values() if isinstance(v, dict) and key in v]
    return sum(vals) / len(vals) if vals else None


def main() -> None:
    entries = [
        ("fsrcnn_fix_clean", "FSRCNN"),
        ("fsrcnn_small", "FSRCNN-Small"),
        ("mobile_srnet", "MobileSRNet"),
        ("mobile_srnet_kd", "MobileSRNet+KD"),
        ("swinir", "SwinIR"),
    ]
    snapshot = {}
    for dir_name, label in entries:
        root = PROJECT_ROOT / "results" / dir_name
        metrics = load_json(root / "benchmark_metrics.json")
        profile = load_json(root / "profile.json")
        train_log = root / "train_log.jsonl"
        last_val_psnr = None
        if train_log.exists():
            for line in train_log.read_text(encoding="utf-8").strip().splitlines():
                row = json.loads(line)
                last_val_psnr = row.get("val_psnr")
        snapshot[label] = {
            "dir": dir_name,
            "benchmark_metrics": metrics,
            "profile": profile,
            "avg_psnr": avg_metric(metrics, "psnr"),
            "avg_ssim": avg_metric(metrics, "ssim"),
            "avg_lpips": avg_metric(metrics, "lpips"),
            "final_div2k_val_psnr": last_val_psnr,
        }

    out_dir = PROJECT_ROOT / "results" / "exp_runs"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "baseline_snapshot.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(snapshot, f, indent=2)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
