#!/usr/bin/env python3
"""Unified SR benchmark evaluation with PSNR, SSIM, and LPIPS."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Callable, Dict, List, Tuple

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
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--benchmark-root", default="data/benchmarks")
    parser.add_argument("--scale", type=int, default=4)
    parser.add_argument("--crop-border", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["Set5", "Set14", "BSD100", "Urban100"],
    )
    parser.add_argument("--save-json", required=True)
    parser.add_argument("--compute-lpips", action="store_true")
    return parser.parse_args()


def pil_to_tensor(image: Image.Image) -> torch.Tensor:
    array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)


def tensor_to_uint8_image(tensor: torch.Tensor) -> np.ndarray:
    array = tensor.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
    return np.clip(array * 255.0, 0, 255).astype(np.uint8)


@torch.no_grad()
def evaluate_dataset(
    predict_fn: Callable[[torch.Tensor], torch.Tensor],
    dataset_dir: Path,
    crop: int,
    device: torch.device,
    lpips_metric: LPIPSMetric | None,
) -> Tuple[float, float, float | None, int]:
    lr_paths = sorted(dataset_dir.glob("*_LR.png"))
    if not lr_paths:
        raise FileNotFoundError(f"No LR files found in {dataset_dir}")

    psnrs: List[float] = []
    ssims: List[float] = []
    lpips_vals: List[float] = []

    for lr_path in lr_paths:
        hr_path = Path(str(lr_path).replace("_LR.png", "_HR.png"))
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
        psnrs.append(compute_psnr(y_pred, y_true))
        ssims.append(compute_ssim(y_pred, y_true))
        if lpips_metric is not None:
            lpips_vals.append(lpips_metric.compute(pred_np, hr_np))

    lpips_avg = float(np.mean(lpips_vals)) if lpips_vals else None
    return float(np.mean(psnrs)), float(np.mean(ssims)), lpips_avg, len(lr_paths)


def main() -> None:
    args = parse_args()
    device = torch.device(
        args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu"
    )

    model, _ = load_checkpoint_model(PROJECT_ROOT / args.checkpoint, device, args.scale)

    def predict_fn(x: torch.Tensor) -> torch.Tensor:
        return model(x)

    lpips_metric = LPIPSMetric(device) if args.compute_lpips else None
    root = PROJECT_ROOT / args.benchmark_root

    header = "Dataset   Images   PSNR(dB)   SSIM"
    if args.compute_lpips:
        header += "   LPIPS"
    print(header)
    print("-" * len(header))

    results: Dict[str, Dict[str, float | int | None]] = {}
    for name in args.datasets:
        dataset_dir = root / name / f"image_SRF_{args.scale}"
        psnr, ssim, lpips_val, count = evaluate_dataset(
            predict_fn, dataset_dir, args.crop_border, device, lpips_metric
        )
        row = {"images": count, "psnr": psnr, "ssim": ssim}
        if lpips_val is not None:
            row["lpips"] = lpips_val
        results[name] = row
        line = f"{name:<8} {count:>6}   {psnr:>8.4f}   {ssim:>6.4f}"
        if lpips_val is not None:
            line += f"   {lpips_val:>6.4f}"
        print(line)

    save_path = PROJECT_ROOT / args.save_json
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with save_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved metrics to: {save_path}")


if __name__ == "__main__":
    main()
