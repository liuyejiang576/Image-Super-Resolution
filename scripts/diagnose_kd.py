#!/usr/bin/env python3
"""Diagnose why KD is a no-op in the fair-budget setup.

D1: Teacher vs student patch-level quality on DIV2K-valid patches (the exact
    distribution KD trains on). If the teacher is barely better than the
    student on patches, KD cannot help.
D2: Gradient alignment between loss_gt and loss_kd. If the two gradients are
    highly aligned (cosine ~ 1), KD is redundant with the GT term and no
    lambda will rescue it. If they diverge, the KD signal is novel and a
    larger lambda is the fix.
D3: Loss-curve snapshot: report loss_gt and loss_kd for the trained kd0/kd02
    checkpoints on fresh patches so we can reason about the optimum.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from data.div2k_dataset import DIV2KPatchDataset  # noqa: E402
from utils.losses import CharbonnierLoss  # noqa: E402
from utils.model_loader import build_model_from_config, load_checkpoint_model  # noqa: E402
from utils.swinir_loader import load_swinir_classical_x4, swinir_forward  # noqa: E402


def _match_size(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if pred.shape[-2:] == target.shape[-2:]:
        return pred
    h, w = target.shape[-2:]
    return pred[..., :h, :w]


def psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    pred = torch.clamp(pred, 0.0, 1.0)
    target = torch.clamp(target, 0.0, 1.0)
    mse = F.mse_loss(pred, target).item()
    if mse == 0:
        return float("inf")
    return -10.0 * np.log10(mse)


def load_student(ckpt_path: Path, device: torch.device):
    model, cfg = load_checkpoint_model(ckpt_path, device, scale=4)
    model.eval()
    return model, cfg


@torch.no_grad()
def d1_patch_quality(
    teacher: nn.Module,
    students: dict,
    hr_dir: Path,
    device: torch.device,
    n_patches: int = 200,
    hr_patch_size: int = 256,
    scale: int = 4,
    seed: int = 42,
) -> dict:
    """Compare teacher and student patch-level PSNR vs HR."""
    ds = DIV2KPatchDataset(
        hr_dir=hr_dir, scale=scale, hr_patch_size=hr_patch_size, augment=False, seed=seed
    )
    torch.manual_seed(seed)
    # Sample n_patches random indices.
    idxs = torch.randperm(len(ds))[:n_patches].tolist()

    results = {name: [] for name in students}
    results["teacher"] = []
    teacher_vs_students = {name: [] for name in students}

    for i in idxs:
        batch = ds[i]
        lr = batch["lr"].unsqueeze(0).to(device)
        hr = batch["hr"].unsqueeze(0).to(device)
        t_out = torch.clamp(swinir_forward(teacher, lr.float()), 0.0, 1.0)
        t_out = _match_size(t_out, hr)
        results["teacher"].append(psnr(t_out, hr))
        for name, model in students.items():
            s_out = torch.clamp(_match_size(model(lr), hr), 0.0, 1.0)
            results[name].append(psnr(s_out, hr))
            # teacher vs student output distance (in PSNR terms against each other)
            teacher_vs_students[name].append(psnr(s_out, t_out))

    summary = {}
    for name, vals in results.items():
        summary[name] = {
            "mean_psnr_vs_hr": float(np.mean(vals)),
            "median_psnr_vs_hr": float(np.median(vals)),
            "std_psnr": float(np.std(vals)),
            "n": len(vals),
        }
    for name, vals in teacher_vs_students.items():
        summary[f"student_vs_teacher_psnr::{name}"] = {
            "mean": float(np.mean(vals)),
            "median": float(np.median(vals)),
        }
    return summary


def d2_gradient_alignment(
    student: nn.Module,
    teacher: nn.Module,
    hr_dir: Path,
    device: torch.device,
    n_batches: int = 8,
    batch_size: int = 16,
    hr_patch_size: int = 256,
    scale: int = 4,
    seed: int = 42,
) -> dict:
    """Compute cosine similarity between grad(loss_gt) and grad(loss_kd)."""
    ds = DIV2KPatchDataset(
        hr_dir=hr_dir, scale=scale, hr_patch_size=hr_patch_size, augment=True, seed=seed
    )
    loader = torch.utils.data.DataLoader(
        ds, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True
    )
    char = CharbonnierLoss()

    cosines = []
    gt_norms = []
    kd_norms = []
    gt_losses = []
    kd_losses = []

    for b, batch in enumerate(loader):
        if b >= n_batches:
            break
        lr = batch["lr"].to(device)
        hr = batch["hr"].to(device)

        with torch.no_grad():
            t_out = torch.clamp(swinir_forward(teacher, lr.float()), 0.0, 1.0)
            t_out = _match_size(t_out, hr)

        student.zero_grad(set_to_none=True)
        s_out = torch.clamp(_match_size(student(lr), hr), 0.0, 1.0)
        loss_gt = char(s_out, hr)
        loss_gt.backward()
        grad_gt = torch.cat([p.grad.detach().flatten() for p in student.parameters() if p.grad is not None])

        student.zero_grad(set_to_none=True)
        s_out2 = torch.clamp(_match_size(student(lr), hr), 0.0, 1.0)
        loss_kd = char(s_out2, t_out)
        loss_kd.backward()
        grad_kd = torch.cat([p.grad.detach().flatten() for p in student.parameters() if p.grad is not None])

        cos = F.cosine_similarity(grad_gt.unsqueeze(0), grad_kd.unsqueeze(0)).item()
        cosines.append(cos)
        gt_norms.append(grad_gt.norm().item())
        kd_norms.append(grad_kd.norm().item())
        gt_losses.append(loss_gt.item())
        kd_losses.append(loss_kd.item())

    return {
        "mean_cosine_grad_gt_vs_kd": float(np.mean(cosines)),
        "median_cosine": float(np.median(cosines)),
        "std_cosine": float(np.std(cosines)),
        "mean_grad_gt_norm": float(np.mean(gt_norms)),
        "mean_grad_kd_norm": float(np.mean(kd_norms)),
        "ratio_kd_over_gt_norm": float(np.mean(kd_norms) / np.mean(gt_norms)),
        "mean_loss_gt": float(np.mean(gt_losses)),
        "mean_loss_kd": float(np.mean(kd_losses)),
        "lambda_for_equal_gradient": float(np.mean(gt_norms) / np.mean(kd_norms)),
        "n_batches": n_batches,
        "n_params": int(grad_gt.numel()),
    }


def main() -> None:
    device = torch.device("cuda")
    hr_dir = PROJECT_ROOT / "data/div2k/DIV2K_valid_HR"

    print("Loading teacher (SwinIR)...")
    teacher = load_swinir_classical_x4(device)

    ckpts = {
        "kd0_20k": PROJECT_ROOT / "results/exp_runs/mobile_srnet_kd0_20k/checkpoints/best.pt",
        "kd02_20k": PROJECT_ROOT / "results/exp_runs/mobile_srnet_kd02_20k/checkpoints/best.pt",
        "mobile_srnet_20k": PROJECT_ROOT / "results/exp_runs/mobile_srnet_20k/checkpoints/best.pt",
    }
    students = {}
    for name, p in ckpts.items():
        print(f"Loading student {name} from {p}")
        m, _ = load_student(p, device)
        students[name] = m

    print("\n=== D1: patch-level PSNR vs HR (DIV2K-valid patches, 64x64 LR) ===")
    d1 = d1_patch_quality(teacher, students, hr_dir, device, n_patches=200)
    print(json.dumps(d1, indent=2))

    print("\n=== D2: gradient alignment (loss_gt vs loss_kd) on kd0_20k student ===")
    d2 = d2_gradient_alignment(
        students["kd0_20k"], teacher, hr_dir, device, n_batches=16, batch_size=16
    )
    print(json.dumps(d2, indent=2))

    print("\n=== D2b: gradient alignment on kd02_20k student ===")
    d2b = d2_gradient_alignment(
        students["kd02_20k"], teacher, hr_dir, device, n_batches=16, batch_size=16
    )
    print(json.dumps(d2b, indent=2))

    out = {"d1_patch_quality": d1, "d2_grad_alignment_kd0": d2, "d2_grad_alignment_kd02": d2b}
    out_path = PROJECT_ROOT / "results/exp_runs/kd_diagnostic.json"
    with out_path.open("w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
