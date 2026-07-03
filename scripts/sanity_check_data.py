#!/usr/bin/env python3
"""Sanity-check DIV2K data pipeline and export sample patches."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from data.div2k_dataset import DIV2KFullImageDataset, DIV2KPatchDataset  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/data.yaml")
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--output-dir", default="results/sanity")
    return parser.parse_args()


def tensor_to_pil(tensor) -> Image.Image:
    array = tensor.detach().cpu().numpy().transpose(1, 2, 0)
    array = np.clip(array * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(array)


def main() -> None:
    args = parse_args()
    cfg_path = PROJECT_ROOT / args.config
    out_dir = PROJECT_ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    dcfg = cfg["dataset"]
    train_ds = DIV2KPatchDataset(
        hr_dir=PROJECT_ROOT / dcfg["train_hr_dir"],
        scale=dcfg["scale"],
        hr_patch_size=dcfg["hr_patch_size"],
        augment=dcfg["augment"],
        seed=42,
    )
    valid_ds = DIV2KFullImageDataset(
        hr_dir=PROJECT_ROOT / dcfg["valid_hr_dir"],
        scale=dcfg["scale"],
    )

    print(f"Train images: {len(train_ds)}")
    print(f"Valid images: {len(valid_ds)}")

    sample = train_ds[0]
    hr_shape = tuple(sample["hr"].shape)
    lr_shape = tuple(sample["lr"].shape)
    print(f"Sample tensor shapes: lr={lr_shape}, hr={hr_shape}")

    metadata = {
        "train_count": len(train_ds),
        "valid_count": len(valid_ds),
        "sample_lr_shape": lr_shape,
        "sample_hr_shape": hr_shape,
        "scale": dcfg["scale"],
        "hr_patch_size": dcfg["hr_patch_size"],
    }

    for i in range(min(args.num_samples, len(train_ds))):
        item = train_ds[i]
        lr_img = tensor_to_pil(item["lr"])
        hr_img = tensor_to_pil(item["hr"])
        lr_up = lr_img.resize(hr_img.size, Image.BICUBIC)

        lr_img.save(out_dir / f"sample_{i:02d}_lr.png")
        hr_img.save(out_dir / f"sample_{i:02d}_hr.png")

        vis = Image.new("RGB", (hr_img.width * 2, hr_img.height))
        vis.paste(lr_up, (0, 0))
        vis.paste(hr_img, (hr_img.width, 0))
        vis.save(out_dir / f"sample_{i:02d}_vis_lrUp_vs_hr.png")

    with (out_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved sanity samples to: {out_dir}")


if __name__ == "__main__":
    main()
