#!/usr/bin/env python3
"""Post-training quantization benchmarks (FP16 + dynamic INT8)."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from utils.model_loader import load_checkpoint_model  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--lr-size", type=int, default=180)
    parser.add_argument("--save-json", required=True)
    return parser.parse_args()


def measure_latency(model, dummy, device, runs=50, warmup=10) -> float:
    model.eval()
    with torch.no_grad():
        for _ in range(warmup):
            model(dummy)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(runs):
            model(dummy)
        if device.type == "cuda":
            torch.cuda.synchronize()
        return (time.perf_counter() - t0) * 1000.0 / runs


def main() -> None:
    args = parse_args()
    device = torch.device(
        args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu"
    )
    model, _ = load_checkpoint_model(PROJECT_ROOT / args.checkpoint, device)
    model.eval()

    dummy = torch.rand(1, 3, args.lr_size, args.lr_size, device=device)
    fp32_latency = measure_latency(model, dummy, device)

    model_fp16 = model.half()
    dummy_fp16 = dummy.half()
    fp16_latency = measure_latency(model_fp16, dummy_fp16, device)

    # Dynamic INT8 quantization (CPU inference proxy for deployment size/latency).
    model_cpu = model.float().cpu()
    dummy_cpu = torch.rand(1, 3, args.lr_size, args.lr_size)
    model_int8 = torch.ao.quantization.quantize_dynamic(
        model_cpu, {torch.nn.Conv2d}, dtype=torch.qint8
    )
    int8_latency = measure_latency(model_int8, dummy_cpu, torch.device("cpu"))

    fp32_path = PROJECT_ROOT / "results" / "_tmp_fp32.pt"
    int8_path = PROJECT_ROOT / "results" / "_tmp_int8.pt"
    torch.save(model_cpu.state_dict(), fp32_path)
    torch.save(model_int8.state_dict(), int8_path)
    fp32_size_mb = fp32_path.stat().st_size / (1024 ** 2)
    int8_size_mb = int8_path.stat().st_size / (1024 ** 2)
    fp32_path.unlink(missing_ok=True)
    int8_path.unlink(missing_ok=True)

    result = {
        "checkpoint": args.checkpoint,
        "latency_fp32_ms_cuda": fp32_latency,
        "latency_fp16_ms_cuda": fp16_latency,
        "latency_int8_ms_cpu": int8_latency,
        "checkpoint_size_fp32_mb": fp32_size_mb,
        "checkpoint_size_int8_mb": int8_size_mb,
        "lr_size": args.lr_size,
        "notes": "INT8 uses torch dynamic quantization on Conv2d; CPU latency is a deployment proxy.",
    }

    save_path = PROJECT_ROOT / args.save_json
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with save_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
