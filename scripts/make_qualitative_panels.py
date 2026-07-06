#!/usr/bin/env python3
"""Build qualitative comparison panels (fair-budget checkpoints, no KD)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from utils.model_loader import load_checkpoint_model  # noqa: E402
from utils.swinir_loader import load_swinir_classical_x4, swinir_forward  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--benchmark-root", default="data/benchmarks")
    p.add_argument("--scale", type=int, default=4)
    p.add_argument("--output-dir", default="results/qualitative")
    p.add_argument("--device", default="cuda")
    return p.parse_args()


PANELS = [
    ("Urban100", "img_092_SRF_4", 200, 200, 128),
    ("Set14", "img_006_SRF_4", 120, 120, 96),
    ("Set14", "img_012_SRF_4", 80, 80, 96),
    ("BSD100", "img_080_SRF_4", 150, 150, 112),
]

CHECKPOINTS = [
    ("FSRCNN", "results/exp_runs/fsrcnn_fix_clean_20k/checkpoints/best.pt"),
    ("MobileSRNet-Base", "results/exp_runs/mobile_srnet_20k/checkpoints/best.pt"),
    ("MobileSRNet-Plus", "results/exp_runs/mobile_srnet_plus_20k/checkpoints/best.pt"),
]

COLUMNS = ["Bicubic", "FSRCNN", "MobileSRNet-Base", "MobileSRNet-Plus", "SwinIR", "HR"]


def pil_to_tensor(image: Image.Image) -> torch.Tensor:
    arr = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)


def tensor_to_image(tensor: torch.Tensor) -> np.ndarray:
    arr = tensor.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
    return np.clip(arr, 0, 1)


def load_models(device: torch.device) -> dict:
    models = {}
    for name, ckpt_rel in CHECKPOINTS:
        ckpt = PROJECT_ROOT / ckpt_rel
        if not ckpt.exists():
            raise FileNotFoundError(f"Missing checkpoint for {name}: {ckpt}")
        model, _ = load_checkpoint_model(ckpt, device)
        models[name] = model
    models["SwinIR"] = load_swinir_classical_x4(device)
    return models


@torch.no_grad()
def predict(models: dict, lr_tensor: torch.Tensor, device: torch.device) -> dict:
    outs = {}
    for name, model in models.items():
        if name == "SwinIR":
            outs[name] = swinir_forward(model, lr_tensor.to(device).float()).cpu()
        else:
            outs[name] = model(lr_tensor.to(device)).cpu()
        outs[name] = torch.clamp(outs[name], 0, 1)
    return outs


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    models = load_models(device)
    out_dir = PROJECT_ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    bench = PROJECT_ROOT / args.benchmark_root

    for ds, stem, cx, cy, size in PANELS:
        lr_path = bench / ds / f"image_SRF_{args.scale}" / f"{stem}_LR.png"
        hr_path = bench / ds / f"image_SRF_{args.scale}" / f"{stem}_HR.png"
        if not lr_path.exists():
            print(f"Skip missing {lr_path}")
            continue

        with Image.open(lr_path) as im:
            lr_full = im.convert("RGB")
        with Image.open(hr_path) as im:
            hr_full = im.convert("RGB")

        lr_t = pil_to_tensor(lr_full)
        preds = predict(models, lr_t, device)

        hr_np = np.asarray(hr_full, dtype=np.float32) / 255.0
        x0, y0 = cx, cy
        x1, y1 = x0 + size, y0 + size
        crop_hr = hr_np[y0:y1, x0:x1]

        bicubic = np.asarray(
            lr_full.resize((hr_np.shape[1], hr_np.shape[0]), Image.BICUBIC), dtype=np.float32
        ) / 255.0

        images = {
            "Bicubic": bicubic[y0:y1, x0:x1],
            "HR": crop_hr,
        }
        for name, pred in preds.items():
            pred_np = tensor_to_image(pred)
            h, w = pred_np.shape[:2]
            px1, py1 = min(x1, w), min(y1, h)
            images[name] = pred_np[y0:py1, x0:px1]

        fig, axes = plt.subplots(1, len(COLUMNS), figsize=(2.8 * len(COLUMNS), 3))
        for ax, col in zip(axes, COLUMNS):
            ax.imshow(images[col])
            ax.set_title(col, fontsize=8)
            ax.axis("off")

        fig.suptitle(f"{ds}/{stem} crop ({size}×{size})", fontsize=11)
        fig.tight_layout()
        out_path = out_dir / f"{ds}_{stem}.png"
        fig.savefig(out_path, dpi=160, bbox_inches="tight")
        plt.close(fig)
        print(f"Wrote {out_path}")

    index_path = out_dir / "README.txt"
    index_path.write_text(
        "Columns: Bicubic | FSRCNN (20k) | MobileSRNet-Base (20k) | MobileSRNet-Plus (20k) | SwinIR | HR\n"
        "Checkpoints from results/exp_runs/*_20k/ (fair-budget). No KD column (confounded baseline-era).\n",
        encoding="utf-8",
    )
    print(f"Wrote {index_path}")


if __name__ == "__main__":
    main()
