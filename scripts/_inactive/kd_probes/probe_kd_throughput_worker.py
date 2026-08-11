#!/usr/bin/env python3
"""Worker: timed KD training steps for solo vs parallel throughput probe."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from data.div2k_dataset import DIV2KPatchDataset  # noqa: E402
from models.mobile_srnet import MobileSRNet  # noqa: E402
from utils.losses import CharbonnierLoss  # noqa: E402
from utils.swinir_loader import load_swinir_classical_x4, swinir_forward  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--worker-id", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--warmup-steps", type=int, default=5)
    p.add_argument("--timed-steps", type=int, default=30)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--lambda-kd", type=float, default=0.2)
    p.add_argument("--out", required=True)
    return p.parse_args()


def _match_size(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if pred.shape[-2:] == target.shape[-2:]:
        return pred
    h, w = target.shape[-2:]
    return pred[..., :h, :w]


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for throughput probe")

    device = torch.device("cuda")
    torch.manual_seed(args.seed + args.worker_id)

    ds = DIV2KPatchDataset(
        hr_dir=PROJECT_ROOT / "data/div2k/DIV2K_train_HR",
        scale=4,
        hr_patch_size=256,
        augment=True,
        seed=args.seed + args.worker_id,
    )
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=args.workers > 0,
    )
    it = iter(loader)

    student = MobileSRNet(scale_factor=4, feat=40, num_blocks=6).to(device)
    teacher = load_swinir_classical_x4(device)
    for p in teacher.parameters():
        p.requires_grad = False
    teacher.eval()

    char = CharbonnierLoss()
    optimizer = Adam(student.parameters(), lr=1e-3)
    scaler = torch.GradScaler(enabled=True)
    student.train()

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    def train_step() -> None:
        batch = next(it)
        lr = batch["lr"].to(device, non_blocking=True)
        hr = batch["hr"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.no_grad():
            with torch.autocast(device_type="cuda", enabled=True):
                teacher_out = swinir_forward(teacher, lr.float()).clamp(0.0, 1.0)
        with torch.autocast(device_type="cuda", enabled=True):
            student_out = _match_size(student(lr), hr).clamp(0.0, 1.0)
            teacher_out = _match_size(teacher_out, hr)
            loss_gt = char(student_out, hr)
            loss_kd = char(student_out, teacher_out)
            loss = loss_gt + args.lambda_kd * loss_kd
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

    for _ in range(args.warmup_steps):
        train_step()
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(args.timed_steps):
        train_step()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    peak_gb = torch.cuda.max_memory_allocated() / (1024**3)
    result = {
        "worker_id": args.worker_id,
        "batch_size": args.batch_size,
        "workers": args.workers,
        "warmup_steps": args.warmup_steps,
        "timed_steps": args.timed_steps,
        "elapsed_sec": elapsed,
        "steps_per_sec": args.timed_steps / elapsed if elapsed > 0 else 0.0,
        "sec_per_step": elapsed / args.timed_steps if args.timed_steps > 0 else 0.0,
        "peak_mem_gb": peak_gb,
        "lambda_kd": args.lambda_kd,
        "oom": False,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as exc:
        if "out of memory" in str(exc).lower():
            out = Path(sys.argv[sys.argv.index("--out") + 1])
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(
                json.dumps({"oom": True, "error": str(exc)[:200]}, indent=2),
                encoding="utf-8",
            )
            sys.exit(1)
        raise
