#!/usr/bin/env python3
"""Train FSRCNN baseline on DIV2K with periodic validation.

For A1a (bs24 / 20k) prefer the control plane:

  python scripts/a1a_20k.py resume
  python scripts/a1a_20k.py watch --interval 60
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from data.div2k_dataset import DIV2KFullImageDataset, DIV2KPatchDataset  # noqa: E402
from models import FSRCNN  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train_fsrcnn.yaml")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--max-train-steps-per-epoch", type=int, default=None)
    parser.add_argument("--val-max-images", type=int, default=None)
    parser.add_argument("--loss-type", choices=["l1", "mse"], default=None)
    parser.add_argument(
        "--resume-from",
        default=None,
        help="Path to checkpoint to resume training from.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _match_size(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if pred.shape[-2:] == target.shape[-2:]:
        return pred
    target_h, target_w = target.shape[-2:]
    return pred[..., :target_h, :target_w]


def psnr_from_tensors(pred: torch.Tensor, target: torch.Tensor) -> float:
    pred = torch.clamp(pred, 0.0, 1.0)
    target = torch.clamp(target, 0.0, 1.0)
    mse = F.mse_loss(pred, target).item()
    if mse == 0:
        return float("inf")
    return -10.0 * np.log10(mse)


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    max_images: int | None = None,
) -> Dict[str, float]:
    model.eval()
    losses = []
    psnrs = []
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if max_images is not None and i >= max_images:
                break
            lr = batch["lr"].to(device, non_blocking=True)
            hr = batch["hr"].to(device, non_blocking=True)
            pred = _match_size(model(lr), hr)
            loss = criterion(pred, hr)
            losses.append(loss.item())
            psnrs.append(psnr_from_tensors(pred, hr))
    return {
        "val_loss": float(np.mean(losses)) if losses else float("nan"),
        "val_psnr": float(np.mean(psnrs)) if psnrs else float("nan"),
    }


def main() -> None:
    args = parse_args()
    cfg_path = PROJECT_ROOT / args.config
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    train_cfg = cfg["train"]
    data_cfg = cfg["dataset"]
    val_cfg = cfg["validation"]
    model_cfg = cfg["model"]
    ckpt_cfg = cfg["checkpoint"]

    if args.epochs is not None:
        train_cfg["epochs"] = args.epochs
    if args.max_train_steps_per_epoch is not None:
        train_cfg["max_train_steps_per_epoch"] = args.max_train_steps_per_epoch
    if args.val_max_images is not None:
        val_cfg["max_images"] = args.val_max_images
    if args.loss_type is not None:
        train_cfg["loss_type"] = args.loss_type

    set_seed(train_cfg["seed"])

    device = torch.device(
        args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu"
    )
    if args.device.startswith("cuda") and device.type == "cpu":
        print("CUDA not available, fallback to CPU.")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    train_ds = DIV2KPatchDataset(
        hr_dir=PROJECT_ROOT / data_cfg["train_hr_dir"],
        scale=data_cfg["scale"],
        hr_patch_size=data_cfg["hr_patch_size"],
        augment=data_cfg["augment"],
        seed=train_cfg["seed"],
    )
    valid_ds = DIV2KFullImageDataset(
        hr_dir=PROJECT_ROOT / data_cfg["valid_hr_dir"],
        scale=data_cfg["scale"],
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=train_cfg["batch_size"],
        shuffle=True,
        num_workers=train_cfg["num_workers"],
        pin_memory=device.type == "cuda",
        drop_last=True,
        persistent_workers=bool(train_cfg["num_workers"] > 0),
    )
    valid_loader = DataLoader(
        valid_ds,
        batch_size=val_cfg["batch_size"],
        shuffle=False,
        num_workers=val_cfg["num_workers"],
        pin_memory=device.type == "cuda",
    )

    model = FSRCNN(
        scale_factor=data_cfg["scale"],
        num_channels=model_cfg["num_channels"],
        d=model_cfg["d"],
        s=model_cfg["s"],
        m=model_cfg["m"],
    ).to(device)
    loss_type = str(train_cfg.get("loss_type", "l1")).lower()
    if loss_type == "mse":
        criterion = nn.MSELoss()
    elif loss_type == "l1":
        criterion = nn.L1Loss()
    else:
        raise ValueError(f"Unsupported loss_type: {loss_type}")
    optimizer = Adam(
        model.parameters(),
        lr=train_cfg["learning_rate"],
        weight_decay=train_cfg["weight_decay"],
    )
    scheduler = MultiStepLR(
        optimizer,
        milestones=train_cfg["milestones"],
        gamma=train_cfg["gamma"],
    )
    scaler = torch.GradScaler(enabled=bool(train_cfg["amp"] and device.type == "cuda"))

    ckpt_dir = PROJECT_ROOT / ckpt_cfg["dir"]
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_path = PROJECT_ROOT / ckpt_cfg["log_path"]
    log_path.parent.mkdir(parents=True, exist_ok=True)

    best_psnr = -1.0
    global_step = 0
    start_epoch = 1

    if args.resume_from:
        resume_path = PROJECT_ROOT / args.resume_from
        resume_ckpt = torch.load(resume_path, map_location=device)
        model.load_state_dict(resume_ckpt["model_state_dict"])
        optimizer.load_state_dict(resume_ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(resume_ckpt["scheduler_state_dict"])
        best_psnr = float(resume_ckpt.get("best_psnr", best_psnr))
        start_epoch = int(resume_ckpt["epoch"]) + 1
        # Approximate global step from resumed epoch.
        global_step = int(resume_ckpt["epoch"]) * len(train_loader)
        print(f"Resumed from {resume_path} at epoch {resume_ckpt['epoch']}")

    for epoch in range(start_epoch, train_cfg["epochs"] + 1):
        model.train()
        running_loss = 0.0
        epoch_start = time.time()
        max_steps = train_cfg.get("max_train_steps_per_epoch")

        for step, batch in enumerate(train_loader, start=1):
            lr = batch["lr"].to(device, non_blocking=True)
            hr = batch["hr"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(device_type=device.type, enabled=scaler.is_enabled()):
                pred = _match_size(model(lr), hr)
                loss = criterion(pred, hr)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            global_step += 1

            if step % train_cfg["log_every"] == 0:
                print(
                    f"epoch={epoch} step={step} "
                    f"loss={running_loss / step:.6f} lr={scheduler.get_last_lr()[0]:.6e}"
                )
            if max_steps is not None and step >= max_steps:
                break

        train_loss = running_loss / max(step, 1)
        metrics = evaluate(
            model=model,
            loader=valid_loader,
            criterion=criterion,
            device=device,
            max_images=val_cfg.get("max_images"),
        )
        elapsed = time.time() - epoch_start
        scheduler.step()

        log_row = {
            "epoch": epoch,
            "global_step": global_step,
            "train_loss": train_loss,
            "val_loss": metrics["val_loss"],
            "val_psnr": metrics["val_psnr"],
            "lr": scheduler.get_last_lr()[0],
            "elapsed_sec": elapsed,
            "loss_type": loss_type,
        }
        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(log_row) + "\n")

        print(
            f"[epoch {epoch:03d}] train_loss={train_loss:.6f} "
            f"val_loss={metrics['val_loss']:.6f} val_psnr={metrics['val_psnr']:.4f}"
        )

        latest_path = ckpt_dir / ckpt_cfg["latest_name"]
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_psnr": best_psnr,
                "config": cfg,
            },
            latest_path,
        )

        if metrics["val_psnr"] > best_psnr:
            best_psnr = metrics["val_psnr"]
            best_path = ckpt_dir / ckpt_cfg["best_name"]
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "best_psnr": best_psnr,
                    "config": cfg,
                },
                best_path,
            )
            print(f"Saved new best checkpoint to {best_path} (PSNR={best_psnr:.4f})")


if __name__ == "__main__":
    main()
