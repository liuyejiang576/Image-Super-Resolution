#!/usr/bin/env python3
"""Profile FSRCNN model size and single-image latency."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from models import FSRCNN  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Model/training yaml config path.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument("--lr-size", type=int, default=180, help="LR H/W input size.")
    return parser.parse_args()


def count_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def main() -> None:
    args = parse_args()
    cfg_path = PROJECT_ROOT / args.config
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    data_cfg = cfg["dataset"]
    model_cfg = cfg["model"]

    device = torch.device(
        args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu"
    )
    model = FSRCNN(
        scale_factor=int(data_cfg["scale"]),
        num_channels=int(model_cfg["num_channels"]),
        d=int(model_cfg["d"]),
        s=int(model_cfg["s"]),
        m=int(model_cfg["m"]),
    ).to(device)
    model.eval()

    dummy = torch.rand(
        1, int(model_cfg["num_channels"]), args.lr_size, args.lr_size, device=device
    )

    with torch.no_grad():
        for _ in range(args.warmup):
            _ = model(dummy)
        if device.type == "cuda":
            torch.cuda.synchronize()

        t0 = time.perf_counter()
        for _ in range(args.runs):
            _ = model(dummy)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()

    latency_ms = (t1 - t0) * 1000.0 / args.runs
    params = count_params(model)
    model_mb_fp32 = params * 4 / (1024 ** 2)

    print(f"config={args.config}")
    print(f"params={params}")
    print(f"model_size_fp32_mb={model_mb_fp32:.3f}")
    print(f"latency_ms_bs1_lr{args.lr_size}={latency_ms:.4f}")


if __name__ == "__main__":
    main()
