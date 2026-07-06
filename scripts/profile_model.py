#!/usr/bin/env python3
"""Profile SR models: params, FLOPs, FP32/FP16 latency."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import yaml
from thop import profile

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from utils.model_loader import build_model_from_config, load_checkpoint_model  # noqa: E402
from utils.swinir_loader import build_swinir_classical_x4, swinir_forward  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-type", choices=["checkpoint", "swinir", "config"], required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--config", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--lr-size", type=int, default=180)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument("--save-json", default=None)
    return parser.parse_args()


def count_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def measure_latency(
    forward_fn,
    device: torch.device,
    warmup: int,
    runs: int,
) -> float:
    with torch.no_grad():
        for _ in range(warmup):
            forward_fn()
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(runs):
            forward_fn()
        if device.type == "cuda":
            torch.cuda.synchronize()
        return (time.perf_counter() - t0) * 1000.0 / runs


def main() -> None:
    args = parse_args()
    device = torch.device(
        args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu"
    )

    if args.model_type == "checkpoint":
        model, cfg = load_checkpoint_model(PROJECT_ROOT / args.checkpoint, device)
        name = Path(args.checkpoint).parent.parent.name
    elif args.model_type == "config":
        with (PROJECT_ROOT / args.config).open("r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        model = build_model_from_config(cfg).to(device)
        name = Path(args.config).stem
    else:
        model = build_swinir_classical_x4().to(device)
        cfg = {"model": {"type": "swinir"}}
        name = "swinir_classical_x4"

    model.eval()
    params = count_params(model)
    dummy = torch.rand(1, 3, args.lr_size, args.lr_size, device=device)

    if args.model_type == "swinir":
        flops, _ = profile(model, inputs=(dummy,), verbose=False)
        latency_fp32 = measure_latency(lambda: swinir_forward(model, dummy), device, args.warmup, args.runs)
        try:
            model_fp16 = model.half()
            dummy_fp16 = dummy.half()
            latency_fp16 = measure_latency(
                lambda: swinir_forward(model_fp16, dummy_fp16), device, args.warmup, args.runs
            )
        except RuntimeError:
            latency_fp16 = None
    else:
        flops, _ = profile(model, inputs=(dummy,), verbose=False)
        latency_fp32 = measure_latency(lambda: model(dummy), device, args.warmup, args.runs)
        model_fp16 = model.half()
        dummy_fp16 = dummy.half()
        latency_fp16 = measure_latency(lambda: model_fp16(dummy_fp16), device, args.warmup, args.runs)

    result = {
        "name": name,
        "params": params,
        "model_size_fp32_mb": params * 4 / (1024 ** 2),
        "flops_g": flops / 1e9,
        "latency_fp32_ms": latency_fp32,
        "latency_fp16_ms": latency_fp16,
        "lr_size": args.lr_size,
    }

    print(json.dumps(result, indent=2))
    if args.save_json:
        save_path = PROJECT_ROOT / args.save_json
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with save_path.open("w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
