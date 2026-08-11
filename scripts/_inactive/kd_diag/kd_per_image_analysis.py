#!/usr/bin/env python3
"""Per-image KD delta analysis across benchmark datasets."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Callable, List, Tuple

import numpy as np
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from utils.lpips_metric import LPIPSMetric  # noqa: E402
from utils.model_loader import load_checkpoint_model  # noqa: E402
from utils.sr_metrics import compute_psnr, compute_ssim, crop_border, rgb_to_y_channel  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--base-ckpt", default="results/mobile_srnet/checkpoints/best.pt")
    p.add_argument("--kd-ckpt", default="results/mobile_srnet_kd/checkpoints/best.pt")
    p.add_argument("--benchmark-root", default="data/benchmarks")
    p.add_argument("--scale", type=int, default=4)
    p.add_argument("--crop-border", type=int, default=4)
    p.add_argument("--output-csv", default="results/kd_analysis/per_image_deltas.csv")
    p.add_argument("--output-json", default="results/kd_analysis/summary.json")
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def pil_to_tensor(image: Image.Image) -> torch.Tensor:
    array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)


def tensor_to_uint8_image(tensor: torch.Tensor) -> np.ndarray:
    array = tensor.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
    return np.clip(array * 255.0, 0, 255).astype(np.uint8)


@torch.no_grad()
def eval_one(
    predict_fn: Callable[[torch.Tensor], torch.Tensor],
    lr_path: Path,
    hr_path: Path,
    crop: int,
    device: torch.device,
    lpips_metric: LPIPSMetric | None,
) -> Tuple[float, float, float | None]:
    with Image.open(lr_path) as lr_im, Image.open(hr_path) as hr_im:
        lr = lr_im.convert("RGB")
        hr = hr_im.convert("RGB")
    pred = predict_fn(pil_to_tensor(lr).to(device))
    pred = torch.clamp(pred, 0.0, 1.0)
    pred_np = tensor_to_uint8_image(pred)
    hr_np = np.asarray(hr, dtype=np.uint8)
    h = min(pred_np.shape[0], hr_np.shape[0])
    w = min(pred_np.shape[1], hr_np.shape[1])
    pred_np = pred_np[:h, :w]
    hr_np = hr_np[:h, :w]
    y_pred = crop_border(rgb_to_y_channel(pred_np), crop)
    y_true = crop_border(rgb_to_y_channel(hr_np), crop)
    psnr = compute_psnr(y_pred, y_true)
    ssim = compute_ssim(y_pred, y_true)
    lpips = lpips_metric.compute(pred_np, hr_np) if lpips_metric else None
    return psnr, ssim, lpips


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    base_model, _ = load_checkpoint_model(PROJECT_ROOT / args.base_ckpt, device, args.scale)
    kd_model, _ = load_checkpoint_model(PROJECT_ROOT / args.kd_ckpt, device, args.scale)
    lpips_metric = LPIPSMetric(device)

    def base_fn(x):
        return base_model(x)

    def kd_fn(x):
        return kd_model(x)

    rows = []
    root = PROJECT_ROOT / args.benchmark_root
    for ds in ["Set5", "Set14", "BSD100", "Urban100"]:
        dataset_dir = root / ds / f"image_SRF_{args.scale}"
        if not dataset_dir.exists():
            continue
        for lr_path in sorted(dataset_dir.glob("*_LR.png")):
            hr_path = Path(str(lr_path).replace("_LR.png", "_HR.png"))
            b_psnr, b_ssim, b_lpips = eval_one(base_fn, lr_path, hr_path, args.crop_border, device, lpips_metric)
            k_psnr, k_ssim, k_lpips = eval_one(kd_fn, lr_path, hr_path, args.crop_border, device, lpips_metric)
            rows.append({
                "dataset": ds,
                "image": lr_path.name.replace("_LR.png", ""),
                "base_psnr": b_psnr,
                "kd_psnr": k_psnr,
                "delta_psnr": k_psnr - b_psnr,
                "base_ssim": b_ssim,
                "kd_ssim": k_ssim,
                "delta_ssim": k_ssim - b_ssim,
                "base_lpips": b_lpips,
                "kd_lpips": k_lpips,
                "delta_lpips": (k_lpips or 0) - (b_lpips or 0),
            })

    out_csv = PROJECT_ROOT / args.output_csv
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary = {}
    for ds in ["Set5", "Set14", "BSD100", "Urban100"]:
        ds_rows = [r for r in rows if r["dataset"] == ds]
        if not ds_rows:
            continue
        n = len(ds_rows)
        summary[ds] = {
            "n_images": n,
            "mean_delta_psnr": sum(r["delta_psnr"] for r in ds_rows) / n,
            "median_delta_psnr": sorted(r["delta_psnr"] for r in ds_rows)[n // 2],
            "pct_improved_psnr": 100.0 * sum(1 for r in ds_rows if r["delta_psnr"] > 0) / n,
            "mean_delta_lpips": sum(r["delta_lpips"] for r in ds_rows) / n,
            "pct_improved_lpips": 100.0 * sum(1 for r in ds_rows if r["delta_lpips"] < 0) / n,
        }

    all_n = len(rows)
    summary["overall"] = {
        "n_images": all_n,
        "mean_delta_psnr": sum(r["delta_psnr"] for r in rows) / all_n,
        "median_delta_psnr": sorted(r["delta_psnr"] for r in rows)[all_n // 2],
        "pct_improved_psnr": 100.0 * sum(1 for r in rows if r["delta_psnr"] > 0) / all_n,
        "mean_delta_lpips": sum(r["delta_lpips"] for r in rows) / all_n,
        "pct_improved_lpips": 100.0 * sum(1 for r in rows if r["delta_lpips"] < 0) / all_n,
    }

    out_json = PROJECT_ROOT / args.output_json
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote {out_csv}")
    print(f"Wrote {out_json}")


if __name__ == "__main__":
    main()
