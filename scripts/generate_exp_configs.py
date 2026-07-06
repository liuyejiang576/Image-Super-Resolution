#!/usr/bin/env python3
"""Generate fair-budget experiment configs from GPU probe recommendations."""

from __future__ import annotations

import json
import math
from copy import deepcopy
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAIN_IMAGES = 800


def epochs_for_updates(batch_size: int, target_updates: int) -> tuple[int, list[int]]:
    steps_per_epoch = TRAIN_IMAGES // batch_size
    epochs = math.ceil(target_updates / steps_per_epoch)
    m1 = max(1, round(0.60 * epochs))
    m2 = max(m1 + 1, round(0.85 * epochs))
    return epochs, [m1, m2]


def write_config(base_path: Path, out_path: Path, overrides: dict) -> dict:
    with base_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    for key, val in overrides.items():
        if "." in key:
            top, sub = key.split(".", 1)
            cfg[top][sub] = val
        else:
            cfg[key] = val
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
    return cfg


def main() -> None:
    rec_path = PROJECT_ROOT / "results/exp_runs/gpu_probe_recommendations.json"
    if rec_path.exists():
        with rec_path.open("r", encoding="utf-8") as f:
            rec = json.load(f)
    else:
        rec = {
            "fsrcnn_fix_clean": {"batch_size": 12, "amp": True, "workers": 4},
            "mobile_srnet": {"batch_size": 32, "amp": True, "workers": 8},
            "mobile_srnet_kd": {"batch_size": 20, "amp": True, "workers": 4},
        }

    manifest = []
    budgets = [10000, 20000]

    fsrcnn_bs = int(rec.get("fsrcnn_fix_clean", {}).get("batch_size", 12))
    fsrcnn_amp = bool(rec.get("fsrcnn_fix_clean", {}).get("amp", True))
    mobile_bs = int(rec.get("mobile_srnet", {}).get("batch_size", 32))
    mobile_amp = bool(rec.get("mobile_srnet", {}).get("amp", True))
    kd_bs = int(rec.get("mobile_srnet_kd", {}).get("batch_size", 20))
    kd_amp = bool(rec.get("mobile_srnet_kd", {}).get("amp", True))
    kd_workers = int(rec.get("mobile_srnet_kd", {}).get("workers", 4))

    for updates in budgets:
        for model_key, base_cfg, bs, tag in [
            ("fsrcnn", "configs/train_fsrcnn_fix_clean.yaml", fsrcnn_bs, "fsrcnn_fix_clean"),
            ("mobile_srnet", "configs/train_mobile_srnet.yaml", mobile_bs, "mobile_srnet"),
        ]:
            epochs, milestones = epochs_for_updates(bs, updates)
            run_id = f"{tag}_{updates//1000}k"
            out = PROJECT_ROOT / "configs" / "exp" / f"{run_id}.yaml"
            write_config(
                PROJECT_ROOT / base_cfg,
                out,
                {
                    "train.batch_size": bs,
                    "train.epochs": epochs,
                    "train.milestones": milestones,
                    "train.amp": fsrcnn_amp if tag == "fsrcnn_fix_clean" else True,
                    "validation.max_images": 100,
                    "checkpoint.dir": f"results/exp_runs/{run_id}/checkpoints",
                    "checkpoint.log_path": f"results/exp_runs/{run_id}/train_log.jsonl",
                },
            )
            manifest.append({
                "run_id": run_id,
                "model": model_key,
                "config": str(out.relative_to(PROJECT_ROOT)),
                "updates_target": updates,
                "epochs": epochs,
                "batch_size": bs,
                "milestones": milestones,
            })

    for updates in budgets:
        for lam, suffix in [(0.0, "kd0"), (0.2, "kd02")]:
            epochs, milestones = epochs_for_updates(kd_bs, updates)
            run_id = f"mobile_srnet_{suffix}_{updates//1000}k"
            out = PROJECT_ROOT / "configs" / "exp" / f"{run_id}.yaml"
            write_config(
                PROJECT_ROOT / "configs/train_mobile_srnet_kd.yaml",
                out,
                {
                    "train.batch_size": kd_bs,
                    "train.epochs": epochs,
                    "train.milestones": milestones,
                    "train.amp": kd_amp,
                    "train.num_workers": kd_workers,
                    "distillation.lambda_kd": lam,
                    "validation.max_images": 100,
                    "checkpoint.dir": f"results/exp_runs/{run_id}/checkpoints",
                    "checkpoint.log_path": f"results/exp_runs/{run_id}/train_log.jsonl",
                },
            )
            manifest.append({
                "run_id": run_id,
                "model": "mobile_srnet_kd",
                "lambda_kd": lam,
                "config": str(out.relative_to(PROJECT_ROOT)),
                "updates_target": updates,
                "epochs": epochs,
                "batch_size": kd_bs,
                "milestones": milestones,
            })

    out_dir = PROJECT_ROOT / "results" / "exp_runs"
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "fair_budget_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"Wrote {len(manifest)} configs to configs/exp/")
    print(f"Wrote results/exp_runs/fair_budget_manifest.json")


if __name__ == "__main__":
    main()
