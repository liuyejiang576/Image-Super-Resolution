#!/usr/bin/env python3
"""Evaluate a trained FSRCNN checkpoint on SR benchmark datasets."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from models import FSRCNN  # noqa: E402
from utils.sr_metrics import compute_psnr, compute_ssim, crop_border, rgb_to_y_channel  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        default="results/fsrcnn/checkpoints/best.pt",
        help="Path to FSRCNN checkpoint (.pt).",
    )
    parser.add_argument("--benchmark-root", default="data/benchmarks")
    parser.add_argument("--scale", type=int, default=4)
    parser.add_argument("--crop-border", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["Set5", "Set14", "BSD100", "Urban100"],
    )
    parser.add_argument("--save-json", default="results/fsrcnn/benchmark_metrics.json")
    return parser.parse_args()


def pil_to_tensor(image: Image.Image) -> torch.Tensor:
    array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)


def tensor_to_uint8_image(tensor: torch.Tensor) -> np.ndarray:
    array = tensor.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
    array = np.clip(array * 255.0, 0, 255).astype(np.uint8)
    return array


def _load_model(checkpoint_path: Path, device: torch.device, scale: int) -> FSRCNN:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    cfg = checkpoint.get("config", {})
    model_cfg = cfg.get("model", {})
    data_cfg = cfg.get("dataset", {})
    model = FSRCNN(
        scale_factor=int(data_cfg.get("scale", scale)),
        num_channels=int(model_cfg.get("num_channels", 3)),
        d=int(model_cfg.get("d", 56)),
        s=int(model_cfg.get("s", 12)),
        m=int(model_cfg.get("m", 4)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


@torch.no_grad()
def evaluate_dataset(
    model: FSRCNN,
    dataset_dir: Path,
    crop: int,
    device: torch.device,
) -> Tuple[float, float, int]:
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

        pred = model(pil_to_tensor(lr).to(device))
        pred = torch.clamp(pred, 0.0, 1.0)

        pred_np = tensor_to_uint8_image(pred)
        hr_np = np.asarray(hr, dtype=np.uint8)
        # Deconvolution output can differ by <= a few pixels; align by crop.
        h = min(pred_np.shape[0], hr_np.shape[0])
        w = min(pred_np.shape[1], hr_np.shape[1])
        pred_np = pred_np[:h, :w]
        hr_np = hr_np[:h, :w]

        y_pred = crop_border(rgb_to_y_channel(pred_np), crop)
        y_true = crop_border(rgb_to_y_channel(hr_np), crop)
        psnrs.append(compute_psnr(y_pred, y_true))
        ssims.append(compute_ssim(y_pred, y_true))

    return float(np.mean(psnrs)), float(np.mean(ssims)), len(lr_paths)


def main() -> None:
    args = parse_args()
    device = torch.device(
        args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu"
    )
    if args.device.startswith("cuda") and device.type == "cpu":
        print("CUDA not available, fallback to CPU.")

    model = _load_model(PROJECT_ROOT / args.checkpoint, device, args.scale)
    root = PROJECT_ROOT / args.benchmark_root

    results: Dict[str, Dict[str, float | int]] = {}
    print("Dataset   Images   PSNR(dB)   SSIM")
    print("-----------------------------------")
    for name in args.datasets:
        dataset_dir = root / name / f"image_SRF_{args.scale}"
        psnr, ssim, count = evaluate_dataset(model, dataset_dir, args.crop_border, device)
        results[name] = {"images": count, "psnr": psnr, "ssim": ssim}
        print(f"{name:<8} {count:>6}   {psnr:>8.4f}   {ssim:>6.4f}")

    save_path = PROJECT_ROOT / args.save_json
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with save_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved metrics to: {save_path}")


if __name__ == "__main__":
    main()
