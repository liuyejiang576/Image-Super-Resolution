#!/usr/bin/env python3
"""Evaluate bicubic baseline on SR benchmark datasets."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-root", default="data/benchmarks")
    parser.add_argument("--scale", type=int, default=4)
    parser.add_argument("--crop-border", type=int, default=4)
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["Set5", "Set14", "BSD100", "Urban100"],
    )
    parser.add_argument("--save-json", default="results/bicubic_metrics.json")
    return parser.parse_args()


def rgb_to_y_channel(rgb: np.ndarray) -> np.ndarray:
    """Convert RGB uint8 image to Y channel in YCbCr space."""
    rgb = rgb.astype(np.float64)
    return (65.738 * rgb[..., 0] + 129.057 * rgb[..., 1] + 25.064 * rgb[..., 2]) / 256.0 + 16.0


def crop_border(img: np.ndarray, border: int) -> np.ndarray:
    if border <= 0:
        return img
    return img[border:-border, border:-border]


def compute_psnr(img1: np.ndarray, img2: np.ndarray) -> float:
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float("inf")
    return 20.0 * math.log10(255.0 / math.sqrt(mse))


def evaluate_dataset(dataset_dir: Path, crop: int) -> Tuple[float, float, int]:
    lr_paths = sorted(dataset_dir.glob("*_LR.png"))
    if not lr_paths:
        raise FileNotFoundError(f"No LR files found in {dataset_dir}")

    psnrs: List[float] = []
    ssims: List[float] = []
    for lr_path in lr_paths:
        hr_path = Path(str(lr_path).replace("_LR.png", "_HR.png"))
        if not hr_path.exists():
            raise FileNotFoundError(f"Missing matching HR file for {lr_path.name}")

        with Image.open(lr_path) as lr_im, Image.open(hr_path) as hr_im:
            lr = lr_im.convert("RGB")
            hr = hr_im.convert("RGB")
            bicubic = lr.resize(hr.size, Image.BICUBIC)

            bicubic_np = np.asarray(bicubic, dtype=np.uint8)
            hr_np = np.asarray(hr, dtype=np.uint8)

        y_pred = crop_border(rgb_to_y_channel(bicubic_np), crop)
        y_true = crop_border(rgb_to_y_channel(hr_np), crop)
        psnrs.append(compute_psnr(y_pred, y_true))
        ssims.append(
            structural_similarity(
                y_pred,
                y_true,
                data_range=255.0,
                gaussian_weights=True,
                sigma=1.5,
                use_sample_covariance=False,
            )
        )

    return float(np.mean(psnrs)), float(np.mean(ssims)), len(lr_paths)


def main() -> None:
    args = parse_args()
    root = Path(args.benchmark_root)
    results: Dict[str, Dict[str, float | int]] = {}

    print("Dataset   Images   PSNR(dB)   SSIM")
    print("-----------------------------------")
    for name in args.datasets:
        dataset_dir = root / name / f"image_SRF_{args.scale}"
        psnr, ssim, count = evaluate_dataset(dataset_dir, args.crop_border)
        results[name] = {"images": count, "psnr": psnr, "ssim": ssim}
        print(f"{name:<8} {count:>6}   {psnr:>8.4f}   {ssim:>6.4f}")

    save_path = Path(args.save_json)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with save_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved metrics to: {save_path}")


if __name__ == "__main__":
    main()
