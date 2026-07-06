#!/usr/bin/env python3
"""Train MobileSRNet with frozen SwinIR teacher and configurable KD loss."""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Callable, Dict

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
from utils.losses import CharbonnierLoss  # noqa: E402
from utils.model_loader import build_model_from_config  # noqa: E402
from utils.swinir_loader import load_swinir_classical_x4, swinir_forward  # noqa: E402
from utils.vgg_features import VGGFeatureExtractor  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train_mobile_srnet_kd.yaml")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--max-train-steps-per-epoch", type=int, default=None)
    parser.add_argument("--val-max-images", type=int, default=None)
    parser.add_argument("--resume-from", default=None)
    parser.add_argument(
        "--init-from",
        default=None,
        help="Initialize student weights from another checkpoint (e.g. MobileSRNet baseline).",
    )
    parser.add_argument("--lambda-kd", type=float, default=None)
    parser.add_argument(
        "--kd-method",
        default=None,
        help="KD loss: pixel_charbonnier (default) or vgg_relu3 / vgg_relu4",
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
            losses.append(criterion(pred, hr).item())
            psnrs.append(psnr_from_tensors(pred, hr))
    return {
        "val_loss": float(np.mean(losses)) if losses else float("nan"),
        "val_psnr": float(np.mean(psnrs)) if psnrs else float("nan"),
    }


def build_kd_loss_fn(
    method: str,
    char_loss: CharbonnierLoss,
    device: torch.device,
) -> tuple[Callable[[torch.Tensor, torch.Tensor], torch.Tensor], nn.Module | None]:
    """Return (kd_loss_fn, optional auxiliary module kept on device)."""
    if method in ("pixel_charbonnier", "pixel", "charbonnier"):
        def pixel_kd(student_out: torch.Tensor, teacher_out: torch.Tensor) -> torch.Tensor:
            return char_loss(student_out, teacher_out)

        return pixel_kd, None

    if method in ("vgg_relu3", "vgg_relu4"):
        layer = "relu3_3" if method == "vgg_relu3" else "relu4_3"
        vgg = VGGFeatureExtractor(layer).to(device)

        def vgg_kd(student_out: torch.Tensor, teacher_out: torch.Tensor) -> torch.Tensor:
            return F.l1_loss(vgg(student_out), vgg(teacher_out.detach()))

        return vgg_kd, vgg

    raise ValueError(
        f"Unknown kd_method {method!r}. Use pixel_charbonnier, vgg_relu3, or vgg_relu4."
    )


def main() -> None:
    args = parse_args()
    cfg_path = PROJECT_ROOT / args.config
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    train_cfg = cfg["train"]
    data_cfg = cfg["dataset"]
    val_cfg = cfg["validation"]
    ckpt_cfg = cfg["checkpoint"]
    kd_cfg = cfg["distillation"]

    if args.epochs is not None:
        train_cfg["epochs"] = args.epochs
    if args.max_train_steps_per_epoch is not None:
        train_cfg["max_train_steps_per_epoch"] = args.max_train_steps_per_epoch
    if args.val_max_images is not None:
        val_cfg["max_images"] = args.val_max_images
    if args.lambda_kd is not None:
        kd_cfg["lambda_kd"] = args.lambda_kd
    kd_method = args.kd_method or kd_cfg.get("kd_method", "pixel_charbonnier")
    kd_cfg["kd_method"] = kd_method

    set_seed(train_cfg["seed"])
    device = torch.device(
        args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu"
    )
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

    student = build_model_from_config(cfg).to(device)
    teacher = load_swinir_classical_x4(device)
    for param in teacher.parameters():
        param.requires_grad = False
    teacher.eval()

    char_loss = CharbonnierLoss()
    kd_loss_fn, _ = build_kd_loss_fn(kd_method, char_loss, device)
    val_criterion = nn.MSELoss()
    lambda_kd = float(kd_cfg["lambda_kd"])
    print(f"KD method={kd_method} lambda_kd={lambda_kd}")

    optimizer = Adam(
        student.parameters(),
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
        student.load_state_dict(resume_ckpt["model_state_dict"])
        optimizer.load_state_dict(resume_ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(resume_ckpt["scheduler_state_dict"])
        best_psnr = float(resume_ckpt.get("best_psnr", best_psnr))
        start_epoch = int(resume_ckpt["epoch"]) + 1
        global_step = int(resume_ckpt["epoch"]) * len(train_loader)
    elif args.init_from:
        init_path = PROJECT_ROOT / args.init_from
        init_ckpt = torch.load(init_path, map_location=device)
        student.load_state_dict(init_ckpt["model_state_dict"])
        print(f"Initialized student from {init_path}")

    for epoch in range(start_epoch, train_cfg["epochs"] + 1):
        student.train()
        running_loss = 0.0
        running_gt = 0.0
        running_kd = 0.0
        epoch_start = time.time()
        max_steps = train_cfg.get("max_train_steps_per_epoch")

        for step, batch in enumerate(train_loader, start=1):
            lr = batch["lr"].to(device, non_blocking=True)
            hr = batch["hr"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            with torch.no_grad():
                teacher_out = swinir_forward(teacher, lr.float())
            teacher_out = teacher_out.clamp(0.0, 1.0)

            with torch.autocast(device_type=device.type, enabled=scaler.is_enabled()):
                student_out = _match_size(student(lr), hr)
                student_out = student_out.clamp(0.0, 1.0)
                teacher_out = _match_size(teacher_out, hr)
                loss_gt = char_loss(student_out, hr)
                loss_kd = kd_loss_fn(student_out, teacher_out)
                loss = loss_gt + lambda_kd * loss_kd

            if not torch.isfinite(loss):
                print(f"Skipping non-finite loss at epoch={epoch} step={step}")
                optimizer.zero_grad(set_to_none=True)
                continue

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            running_gt += loss_gt.item()
            running_kd += loss_kd.item()
            global_step += 1

            if step % train_cfg["log_every"] == 0:
                print(
                    f"epoch={epoch} step={step} loss={running_loss / step:.6f} "
                    f"gt={running_gt / step:.6f} kd={running_kd / step:.6f}"
                )
            if max_steps is not None and step >= max_steps:
                break

        train_loss = running_loss / max(step, 1)
        metrics = evaluate(student, valid_loader, val_criterion, device, val_cfg.get("max_images"))
        elapsed = time.time() - epoch_start
        scheduler.step()

        log_row = {
            "epoch": epoch,
            "global_step": global_step,
            "train_loss": train_loss,
            "train_loss_gt": running_gt / max(step, 1),
            "train_loss_kd": running_kd / max(step, 1),
            "val_loss": metrics["val_loss"],
            "val_psnr": metrics["val_psnr"],
            "lr": scheduler.get_last_lr()[0],
            "elapsed_sec": elapsed,
            "lambda_kd": lambda_kd,
            "kd_method": kd_method,
        }
        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(log_row) + "\n")

        print(
            f"[epoch {epoch:03d}] train_loss={train_loss:.6f} "
            f"val_psnr={metrics['val_psnr']:.4f}"
        )

        latest_path = ckpt_dir / ckpt_cfg["latest_name"]
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": student.state_dict(),
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
                    "model_state_dict": student.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "best_psnr": best_psnr,
                    "config": cfg,
                },
                best_path,
            )
            print(f"Saved new best checkpoint (PSNR={best_psnr:.4f})")


if __name__ == "__main__":
    main()
