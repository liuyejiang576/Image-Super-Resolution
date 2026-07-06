#!/usr/bin/env python3
"""Audited CUDA latency benchmark with symmetric FP32/FP16 protocol."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from models import FSRCNN, MobileSRNet  # noqa: E402
from utils.model_loader import load_checkpoint_model  # noqa: E402
from utils.swinir_loader import build_swinir_classical_x4, swinir_forward  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--lr-h", type=int, default=180)
    p.add_argument("--lr-w", type=int, default=None)
    p.add_argument("--warmup", type=int, default=100)
    p.add_argument("--iters", type=int, default=500)
    p.add_argument("--output", default="results/latency_audit/latency_audit.json")
    return p.parse_args()


@torch.no_grad()
def benchmark_cuda(model, input_shape, dtype=torch.float32, warmup=100, iters=500):
    device = torch.device("cuda")
    model = model.to(device).eval()
    if dtype == torch.float16:
        model = model.half()
    else:
        model = model.float()

    x = torch.randn(*input_shape, device=device, dtype=dtype)

    for _ in range(warmup):
        _ = model(x)
    torch.cuda.synchronize()

    starter = torch.cuda.Event(enable_timing=True)
    ender = torch.cuda.Event(enable_timing=True)
    times = []
    for _ in range(iters):
        starter.record()
        _ = model(x)
        ender.record()
        torch.cuda.synchronize()
        times.append(starter.elapsed_time(ender))

    arr = np.array(times)
    return {
        "mean_ms": float(arr.mean()),
        "std_ms": float(arr.std()),
        "median_ms": float(np.median(arr)),
        "p90_ms": float(np.percentile(arr, 90)),
    }


@torch.no_grad()
def benchmark_swinir(model, input_shape, dtype=torch.float32, warmup=20, iters=50):
    device = torch.device("cuda")
    model = model.to(device).eval()
    if dtype == torch.float16:
        model = model.half()
    x = torch.randn(*input_shape, device=device, dtype=dtype)

    for _ in range(warmup):
        _ = swinir_forward(model, x)
    torch.cuda.synchronize()

    starter = torch.cuda.Event(enable_timing=True)
    ender = torch.cuda.Event(enable_timing=True)
    times = []
    for _ in range(iters):
        starter.record()
        _ = swinir_forward(model, x)
        ender.record()
        torch.cuda.synchronize()
        times.append(starter.elapsed_time(ender))

    arr = np.array(times)
    return {
        "mean_ms": float(arr.mean()),
        "std_ms": float(arr.std()),
        "median_ms": float(np.median(arr)),
        "p90_ms": float(np.percentile(arr, 90)),
    }


def load_models(device):
    models = {}
    ckpt_pairs = [
        ("FSRCNN", "results/exp_runs/fsrcnn_fix_clean_20k/checkpoints/best.pt"),
        ("FSRCNN-Small", "results/fsrcnn_small/checkpoints/best.pt"),
        ("MobileSRNet-Base", "results/exp_runs/mobile_srnet_20k/checkpoints/best.pt"),
        ("MobileSRNet-Plus", "results/exp_runs/mobile_srnet_plus_20k/checkpoints/best.pt"),
    ]
    for name, path in ckpt_pairs:
        p = PROJECT_ROOT / path
        if p.exists():
            m, _ = load_checkpoint_model(p, device)
            models[name] = m
        elif name == "FSRCNN":
            models[name] = FSRCNN(scale_factor=4, d=56, s=12, m=4).to(device)
        elif name == "FSRCNN-Small":
            models[name] = FSRCNN(scale_factor=4, d=32, s=8, m=2).to(device)
        elif name == "MobileSRNet-Base":
            models[name] = MobileSRNet(scale_factor=4, feat=40, num_blocks=6).to(device)
        elif name == "MobileSRNet-Plus":
            models[name] = MobileSRNet(scale_factor=4, feat=64, num_blocks=8).to(device)
    models["SwinIR"] = build_swinir_classical_x4().to(device)
    return models


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required for latency audit")

    torch.backends.cudnn.benchmark = True
    torch.set_grad_enabled(False)
    lr_w = args.lr_w if args.lr_w is not None else args.lr_h
    shape = (1, 3, args.lr_h, lr_w)

    device = torch.device("cuda")
    models = load_models(device)
    results = {"input_lr": [args.lr_h, lr_w], "protocol": {"warmup": args.warmup, "iters": args.iters}}

    for name, model in models.items():
        print(f"Benchmarking {name}...")
        entry = {}
        if name == "SwinIR":
            entry["fp32"] = benchmark_swinir(model, shape, torch.float32, warmup=20, iters=50)
            try:
                entry["fp16"] = benchmark_swinir(model, shape, torch.float16, warmup=20, iters=50)
            except RuntimeError as exc:
                entry["fp16"] = {"error": str(exc)}
        else:
            entry["fp32"] = benchmark_cuda(model, shape, torch.float32, args.warmup, args.iters)
            entry["fp16"] = benchmark_cuda(model, shape, torch.float16, args.warmup, args.iters)
        results[name] = entry

    out = PROJECT_ROOT / args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
