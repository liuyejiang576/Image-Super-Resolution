#!/usr/bin/env python3
"""Mini-probe GPU throughput and memory for training configs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from copy import deepcopy
from pathlib import Path

import torch
import yaml
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from data.div2k_dataset import DIV2KPatchDataset  # noqa: E402
from models import FSRCNN, MobileSRNet  # noqa: E402
from utils.losses import CharbonnierLoss  # noqa: E402
from utils.swinir_loader import load_swinir_classical_x4, swinir_forward  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--probe-steps", type=int, default=30)
    parser.add_argument("--output", default="results/exp_runs/gpu_probe.csv")
    return parser.parse_args()


def _match_size(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if pred.shape[-2:] == target.shape[-2:]:
        return pred
    h, w = target.shape[-2:]
    return pred[..., :h, :w]


def probe_mobile(batch_size: int, amp: bool, workers: int, steps: int, device: torch.device) -> dict:
    ds = DIV2KPatchDataset(
        hr_dir=PROJECT_ROOT / "data/div2k/DIV2K_train_HR",
        scale=4,
        hr_patch_size=256,
        augment=True,
        seed=42,
    )
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=workers > 0,
    )
    model = MobileSRNet(scale_factor=4, feat=40, num_blocks=6).to(device)
    criterion = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=1e-3)
    scaler = torch.GradScaler(enabled=amp and device.type == "cuda")
    model.train()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    step = 0
    for batch in loader:
        lr = batch["lr"].to(device, non_blocking=True)
        hr = batch["hr"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=scaler.is_enabled()):
            pred = _match_size(model(lr), hr)
            loss = criterion(pred, hr)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        step += 1
        if step >= steps:
            break
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    peak_gb = torch.cuda.max_memory_allocated() / (1024**3) if device.type == "cuda" else 0.0
    return {
        "model": "mobile_srnet",
        "batch_size": batch_size,
        "amp": amp,
        "workers": workers,
        "steps_per_sec": step / elapsed if elapsed > 0 else 0.0,
        "peak_mem_gb": peak_gb,
        "oom": False,
        "notes": "",
    }


def probe_fsrcnn(batch_size: int, amp: bool, workers: int, steps: int, device: torch.device) -> dict:
    ds = DIV2KPatchDataset(
        hr_dir=PROJECT_ROOT / "data/div2k/DIV2K_train_HR",
        scale=4,
        hr_patch_size=256,
        augment=True,
        seed=42,
    )
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=workers > 0,
    )
    model = FSRCNN(scale_factor=4, d=56, s=12, m=4).to(device)
    criterion = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=1e-3)
    scaler = torch.GradScaler(enabled=amp and device.type == "cuda")
    model.train()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    step = 0
    for batch in loader:
        lr = batch["lr"].to(device, non_blocking=True)
        hr = batch["hr"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=scaler.is_enabled()):
            pred = _match_size(model(lr), hr)
            loss = criterion(pred, hr)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        step += 1
        if step >= steps:
            break
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    peak_gb = torch.cuda.max_memory_allocated() / (1024**3) if device.type == "cuda" else 0.0
    return {
        "model": "fsrcnn_fix_clean",
        "batch_size": batch_size,
        "amp": amp,
        "workers": workers,
        "steps_per_sec": step / elapsed if elapsed > 0 else 0.0,
        "peak_mem_gb": peak_gb,
        "oom": False,
        "notes": "",
    }


def probe_kd(batch_size: int, amp: bool, workers: int, steps: int, device: torch.device) -> dict:
    ds = DIV2KPatchDataset(
        hr_dir=PROJECT_ROOT / "data/div2k/DIV2K_train_HR",
        scale=4,
        hr_patch_size=256,
        augment=True,
        seed=42,
    )
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=workers > 0,
    )
    student = MobileSRNet(scale_factor=4, feat=40, num_blocks=6).to(device)
    teacher = load_swinir_classical_x4(device)
    for p in teacher.parameters():
        p.requires_grad = False
    teacher.eval()
    char = CharbonnierLoss()
    optimizer = Adam(student.parameters(), lr=1e-3)
    scaler = torch.GradScaler(enabled=amp and device.type == "cuda")
    student.train()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    step = 0
    for batch in loader:
        lr = batch["lr"].to(device, non_blocking=True)
        hr = batch["hr"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.no_grad():
            with torch.autocast(device_type=device.type, enabled=amp and device.type == "cuda"):
                teacher_out = swinir_forward(teacher, lr.float()).clamp(0.0, 1.0)
        with torch.autocast(device_type=device.type, enabled=scaler.is_enabled()):
            student_out = _match_size(student(lr), hr).clamp(0.0, 1.0)
            teacher_out = _match_size(teacher_out, hr)
            loss = char(student_out, hr) + 0.2 * char(student_out, teacher_out)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        step += 1
        if step >= steps:
            break
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    peak_gb = torch.cuda.max_memory_allocated() / (1024**3) if device.type == "cuda" else 0.0
    return {
        "model": "mobile_srnet_kd",
        "batch_size": batch_size,
        "amp": amp,
        "workers": workers,
        "steps_per_sec": step / elapsed if elapsed > 0 else 0.0,
        "peak_mem_gb": peak_gb,
        "oom": False,
        "notes": "",
    }


def run_probe(fn, **kwargs) -> dict:
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return fn(**kwargs)
    except RuntimeError as exc:
        if "out of memory" in str(exc).lower():
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return {
                "model": kwargs.get("model", "unknown"),
                "batch_size": kwargs.get("batch_size"),
                "amp": kwargs.get("amp"),
                "workers": kwargs.get("workers"),
                "steps_per_sec": 0.0,
                "peak_mem_gb": 0.0,
                "oom": True,
                "notes": str(exc)[:120],
            }
        raise


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    rows = []
    run_id = 0

    probes = [
        ("mobile_srnet", probe_mobile, [(24, True, 8), (28, True, 8), (32, True, 8), (32, True, 10)]),
        ("fsrcnn_fix_clean", probe_fsrcnn, [(8, False, 4), (10, False, 4), (12, False, 4), (16, True, 4), (12, True, 4)]),
        ("mobile_srnet_kd", probe_kd, [(16, False, 8), (16, True, 8), (20, True, 8), (24, True, 8)]),
    ]

    for model_name, fn, grid in probes:
        for batch_size, amp, workers in grid:
            run_id += 1
            print(f"[{run_id}] probing {model_name} bs={batch_size} amp={amp} workers={workers}")
            row = run_probe(
                fn,
                batch_size=batch_size,
                amp=amp,
                workers=workers,
                steps=args.probe_steps,
                device=device,
            )
            row["run_id"] = run_id
            rows.append(row)
            print(json.dumps(row))

    out_path = PROJECT_ROOT / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "run_id", "model", "batch_size", "amp", "workers",
        "steps_per_sec", "peak_mem_gb", "oom", "notes",
    ]
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    # Pick best non-OOM configs in 6.8-7.2 GB window (or closest below 7.2)
    best = {}
    for row in rows:
        if row["oom"]:
            continue
        key = row["model"]
        mem = row["peak_mem_gb"]
        if mem > 7.2:
            continue
        prev = best.get(key)
        if prev is None or row["steps_per_sec"] > prev["steps_per_sec"]:
            best[key] = row

    rec_path = out_path.parent / "gpu_probe_recommendations.json"
    with rec_path.open("w", encoding="utf-8") as f:
        json.dump(best, f, indent=2)
    print(f"Wrote {out_path}")
    print(f"Wrote {rec_path}")


if __name__ == "__main__":
    main()
