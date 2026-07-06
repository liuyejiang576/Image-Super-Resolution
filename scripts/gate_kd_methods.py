#!/usr/bin/env python3
"""Cheap pre-training gates for KD method selection.

Gate 1 — signal availability: is there a teacher–student gap in the KD target space?
Gate 2 — gradient non-redundancy: is grad(L_kd) non-collinear with grad(L_gt)?

Run this before any full-budget KD training. Methods that fail either gate should be
rejected without a 13h run.

Example:
  python scripts/gate_kd_methods.py
  python scripts/gate_kd_methods.py --student-checkpoint results/exp_runs/mobile_srnet_20k/checkpoints/best.pt
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from data.div2k_dataset import DIV2KPatchDataset  # noqa: E402
from utils.losses import CharbonnierLoss  # noqa: E402
from utils.model_loader import build_model_from_config, load_checkpoint_model  # noqa: E402
from utils.swinir_loader import load_swinir_classical_x4, swinir_forward  # noqa: E402

# --- Gate thresholds (tune if needed; document in report) ---
GATE1_MIN_TEACHER_ADVANTAGE = 0.001  # student_hr_gap - teacher_hr_gap in KD space
GATE1_MIN_STUDENT_TEACHER_GAP = 0.001  # must be room to move toward teacher
GATE2_COSINE_MAX = 0.85  # pass if |cosine| < this (lower = more novel direction)
GATE2_MIN_NORM_RATIO = 0.05  # pass if mean ||grad_kd|| / ||grad_gt|| >= this


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pre-training gates for KD methods")
    parser.add_argument(
        "--student-checkpoint",
        default=None,
        help="Trained student checkpoint (recommended). Tries common paths if omitted.",
    )
    parser.add_argument(
        "--hr-dir",
        default="data/div2k/DIV2K_valid_HR",
        help="HR images for patch sampling",
    )
    parser.add_argument("--n-patches", type=int, default=64, help="Gate 1 patch count")
    parser.add_argument("--n-batches", type=int, default=8, help="Gate 2 batch count")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--hr-patch-size", type=int, default=256)
    parser.add_argument("--scale", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--methods",
        nargs="*",
        default=None,
        help="Subset of method ids (default: all)",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="cuda or cpu",
    )
    parser.add_argument(
        "--out",
        default="results/exp_runs/kd_method_gates.json",
        help="Output JSON path",
    )
    return parser.parse_args()


def _match_size(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if pred.shape[-2:] == target.shape[-2:]:
        return pred
    h, w = target.shape[-2:]
    return pred[..., :h, :w]


def bicubic_upsample(lr: torch.Tensor, scale: int) -> torch.Tensor:
    return F.interpolate(lr, scale_factor=scale, mode="bicubic", align_corners=False)


def haar_dwt2(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """One-level Haar DWT. Returns LL, LH, HL, HH."""
    x00 = x[:, :, 0::2, 0::2]
    x01 = x[:, :, 0::2, 1::2]
    x10 = x[:, :, 1::2, 0::2]
    x11 = x[:, :, 1::2, 1::2]
    ll = (x00 + x01 + x10 + x11) * 0.5
    lh = (x00 + x01 - x10 - x11) * 0.5
    hl = (x00 - x01 + x10 - x11) * 0.5
    hh = (x00 - x01 - x10 + x11) * 0.5
    return ll, lh, hl, hh


class VGGFeatureExtractor(nn.Module):
    """Frozen VGG16 features up to relu3_3 or relu4_3."""

    SLICES = {
        "relu3_3": (0, 16),
        "relu4_3": (0, 23),
    }

    def __init__(self, layer: str = "relu3_3") -> None:
        super().__init__()
        from torchvision.models import VGG16_Weights, vgg16

        vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features
        start, end = self.SLICES[layer]
        self.slice = vgg[start:end].eval()
        for p in self.slice.parameters():
            p.requires_grad = False
        self.layer = layer
        self.register_buffer(
            "mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1),
        )
        self.register_buffer(
            "std",
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = (x - self.mean) / self.std
        return self.slice(x)


@dataclass
class KDMethodSpec:
    id: str
    description: str
    # kd_loss(student_out, teacher_out, lr, hr, ctx) -> scalar tensor with grad
    kd_loss_fn: Callable[..., torch.Tensor]
    # signal metric on detached tensors (lower = closer); lr optional for residual methods
    signal_fn: Callable[..., torch.Tensor]


def _resolve_student(args: argparse.Namespace, device: torch.device) -> tuple[nn.Module, str]:
    candidates = []
    if args.student_checkpoint:
        candidates.append(PROJECT_ROOT / args.student_checkpoint)
    candidates.extend(
        [
            PROJECT_ROOT / "results/exp_runs/mobile_srnet_20k/checkpoints/best.pt",
            PROJECT_ROOT / "results/exp_runs/mobile_srnet_kd0_20k/checkpoints/best.pt",
            PROJECT_ROOT / "results/mobile_srnet/checkpoints/best.pt",
        ]
    )
    for path in candidates:
        if path.exists():
            model, _ = load_checkpoint_model(path, device, scale=args.scale)
            model.train()  # gradients needed for Gate 2
            return model, str(path)

    cfg_path = PROJECT_ROOT / "configs/train_mobile_srnet.yaml"
    import yaml

    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    model = build_model_from_config(cfg).to(device)
    model.train()
    print(
        "WARNING: no student checkpoint found; using randomly initialized MobileSRNet.\n"
        "         Gate rankings are still comparable, but absolute cosine values may differ."
    )
    return model, "random_init"


def _build_methods(device: torch.device) -> dict[str, KDMethodSpec]:
    char = CharbonnierLoss()
    vgg3 = VGGFeatureExtractor("relu3_3").to(device)
    vgg4 = VGGFeatureExtractor("relu4_3").to(device)

    def pixel_kd(s_out, t_out, lr, hr, ctx):
        return char(s_out, t_out)

    def pixel_signal(s_out, t_out, hr, lr=None):
        return char(s_out, t_out)

    def residual_kd(s_out, t_out, lr, hr, ctx):
        scale = ctx["scale"]
        bic = bicubic_upsample(lr, scale)
        bic = _match_size(bic, hr)
        return char(s_out - bic, t_out - bic)

    def residual_signal(s_out, t_out, hr, lr=None):
        assert lr is not None
        scale = hr.shape[-1] // lr.shape[-1]
        bic = _match_size(bicubic_upsample(lr, scale), hr)
        return char(s_out - bic, t_out - bic)

    def vgg_kd_factory(vgg: VGGFeatureExtractor):
        def kd(s_out, t_out, lr, hr, ctx):
            return F.l1_loss(vgg(s_out), vgg(t_out.detach()))

        def signal(s_out, t_out, hr, lr=None):
            return F.l1_loss(vgg(s_out), vgg(t_out))

        return kd, signal

    vgg3_kd, vgg3_sig = vgg_kd_factory(vgg3)
    vgg4_kd, vgg4_sig = vgg_kd_factory(vgg4)

    def wavelet_hf_kd(s_out, t_out, lr, hr, ctx):
        _, lh_s, hl_s, hh_s = haar_dwt2(s_out)
        _, lh_t, hl_t, hh_t = haar_dwt2(t_out.detach())
        return (
            F.l1_loss(lh_s, lh_t)
            + F.l1_loss(hl_s, hl_t)
            + F.l1_loss(hh_s, hh_t)
        ) / 3.0

    def wavelet_hf_signal(s_out, t_out, hr, lr=None):
        _, lh_s, hl_s, hh_s = haar_dwt2(s_out)
        _, lh_t, hl_t, hh_t = haar_dwt2(t_out)
        return (
            F.l1_loss(lh_s, lh_t)
            + F.l1_loss(hl_s, hl_t)
            + F.l1_loss(hh_s, hh_t)
        ) / 3.0

    def wavelet_lf_kd(s_out, t_out, lr, hr, ctx):
        ll_s, _, _, _ = haar_dwt2(s_out)
        ll_t, _, _, _ = haar_dwt2(t_out.detach())
        return F.l1_loss(ll_s, ll_t)

    def wavelet_lf_signal(s_out, t_out, hr, lr=None):
        ll_s, _, _, _ = haar_dwt2(s_out)
        ll_t, _, _, _ = haar_dwt2(t_out)
        return F.l1_loss(ll_s, ll_t)

    return {
        "pixel_charbonnier": KDMethodSpec(
            id="pixel_charbonnier",
            description="Charbonnier(student, teacher) — current baseline (expected fail Gate 2)",
            kd_loss_fn=pixel_kd,
            signal_fn=pixel_signal,
        ),
        "residual_charbonnier": KDMethodSpec(
            id="residual_charbonnier",
            description="Charbonnier on (output − bicubic) residual",
            kd_loss_fn=residual_kd,
            signal_fn=residual_signal,
        ),
        "vgg_relu3": KDMethodSpec(
            id="vgg_relu3",
            description="L1 on frozen VGG16 relu3_3 features",
            kd_loss_fn=vgg3_kd,
            signal_fn=vgg3_sig,
        ),
        "vgg_relu4": KDMethodSpec(
            id="vgg_relu4",
            description="L1 on frozen VGG16 relu4_3 features",
            kd_loss_fn=vgg4_kd,
            signal_fn=vgg4_sig,
        ),
        "wavelet_hf": KDMethodSpec(
            id="wavelet_hf",
            description="L1 on Haar LH/HL/HH high-frequency subbands",
            kd_loss_fn=wavelet_hf_kd,
            signal_fn=wavelet_hf_signal,
        ),
        "wavelet_lf": KDMethodSpec(
            id="wavelet_lf",
            description="L1 on Haar LL low-frequency subband (sanity / likely redundant)",
            kd_loss_fn=wavelet_lf_kd,
            signal_fn=wavelet_lf_signal,
        ),
    }


@torch.no_grad()
def gate1_signal_availability(
    student: nn.Module,
    teacher: nn.Module,
    spec: KDMethodSpec,
    hr_dir: Path,
    device: torch.device,
    n_patches: int,
    hr_patch_size: int,
    scale: int,
    seed: int,
) -> dict:
    """Measure teacher–student vs teacher–HR gaps in the KD target space."""
    ds = DIV2KPatchDataset(
        hr_dir=hr_dir,
        scale=scale,
        hr_patch_size=hr_patch_size,
        augment=False,
        seed=seed,
    )
    rng = torch.Generator().manual_seed(seed)
    idxs = torch.randperm(len(ds), generator=rng)[:n_patches].tolist()

    st_teacher = []
    st_hr = []
    teacher_hr = []

    for i in idxs:
        batch = ds[i]
        lr = batch["lr"].unsqueeze(0).to(device)
        hr = batch["hr"].unsqueeze(0).to(device)
        t_out = torch.clamp(swinir_forward(teacher, lr.float()), 0.0, 1.0)
        t_out = _match_size(t_out, hr)
        s_out = torch.clamp(_match_size(student(lr), hr), 0.0, 1.0)

        st_teacher.append(spec.signal_fn(s_out, t_out, hr, lr).item())
        st_hr.append(spec.signal_fn(s_out, hr, hr, lr).item())
        teacher_hr.append(spec.signal_fn(t_out, hr, hr, lr).item())

    st_teacher_m = float(np.mean(st_teacher))
    st_hr_m = float(np.mean(st_hr))
    teacher_hr_m = float(np.mean(teacher_hr))
    teacher_advantage = st_hr_m - teacher_hr_m
    # Teacher must beat the student vs HR, and student must not already match teacher.
    pass_g1 = (
        teacher_advantage > GATE1_MIN_TEACHER_ADVANTAGE
        and st_teacher_m > GATE1_MIN_STUDENT_TEACHER_GAP
    )

    return {
        "student_teacher_gap": st_teacher_m,
        "student_hr_gap": st_hr_m,
        "teacher_hr_gap": teacher_hr_m,
        "teacher_advantage": teacher_advantage,
        "gate1_pass": pass_g1,
        "gate1_reason": (
            "signal present"
            if pass_g1
            else (
                f"no teacher advantage ({teacher_advantage:.4f} <= {GATE1_MIN_TEACHER_ADVANTAGE})"
                if teacher_advantage <= GATE1_MIN_TEACHER_ADVANTAGE
                else f"student already near teacher ({st_teacher_m:.4f})"
            )
        ),
        "n_patches": n_patches,
    }


def gate2_gradient_alignment(
    student: nn.Module,
    teacher: nn.Module,
    spec: KDMethodSpec,
    hr_dir: Path,
    device: torch.device,
    n_batches: int,
    batch_size: int,
    hr_patch_size: int,
    scale: int,
    seed: int,
) -> dict:
    """Compare grad(L_gt) vs grad(L_kd) for a given KD formulation."""
    ds = DIV2KPatchDataset(
        hr_dir=hr_dir,
        scale=scale,
        hr_patch_size=hr_patch_size,
        augment=True,
        seed=seed,
    )
    loader = DataLoader(
        ds, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True
    )
    char = CharbonnierLoss()
    ctx = {"scale": scale}

    cosines = []
    gt_norms = []
    kd_norms = []

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
        grad_gt = torch.cat(
            [p.grad.detach().flatten() for p in student.parameters() if p.grad is not None]
        )

        student.zero_grad(set_to_none=True)
        s_out2 = torch.clamp(_match_size(student(lr), hr), 0.0, 1.0)
        loss_kd = spec.kd_loss_fn(s_out2, t_out, lr, hr, ctx)
        loss_kd.backward()
        grad_kd = torch.cat(
            [p.grad.detach().flatten() for p in student.parameters() if p.grad is not None]
        )

        cos = F.cosine_similarity(grad_gt.unsqueeze(0), grad_kd.unsqueeze(0)).item()
        cosines.append(cos)
        gt_norms.append(grad_gt.norm().item())
        kd_norms.append(grad_kd.norm().item())

    mean_cos = float(np.mean(cosines))
    mean_gt_norm = float(np.mean(gt_norms))
    mean_kd_norm = float(np.mean(kd_norms))
    norm_ratio = mean_kd_norm / max(mean_gt_norm, 1e-8)

    pass_g2 = abs(mean_cos) < GATE2_COSINE_MAX and norm_ratio >= GATE2_MIN_NORM_RATIO
    if abs(mean_cos) >= GATE2_COSINE_MAX:
        reason = f"gradient redundant (cosine={mean_cos:.3f} >= {GATE2_COSINE_MAX})"
    elif norm_ratio < GATE2_MIN_NORM_RATIO:
        reason = f"KD gradient too weak (norm_ratio={norm_ratio:.3f} < {GATE2_MIN_NORM_RATIO})"
    else:
        reason = "non-redundant gradient"

    return {
        "mean_cosine_grad_gt_vs_kd": mean_cos,
        "median_cosine": float(np.median(cosines)),
        "std_cosine": float(np.std(cosines)),
        "mean_grad_gt_norm": mean_gt_norm,
        "mean_grad_kd_norm": mean_kd_norm,
        "norm_ratio_kd_over_gt": norm_ratio,
        "lambda_for_equal_gradient": float(mean_gt_norm / max(mean_kd_norm, 1e-8)),
        "gate2_pass": pass_g2,
        "gate2_reason": reason,
        "n_batches": n_batches,
    }


def verdict(g1: dict, g2: dict) -> tuple[str, str]:
    if g1["gate1_pass"] and g2["gate2_pass"]:
        return "PROCEED", "passes both gates — eligible for Stage B (1–2k update probe)"
    if not g1["gate1_pass"] and not g2["gate2_pass"]:
        return "REJECT", f"Gate1: {g1['gate1_reason']}; Gate2: {g2['gate2_reason']}"
    if not g1["gate1_pass"]:
        return "REJECT", f"Gate1 fail: {g1['gate1_reason']}"
    return "REJECT", f"Gate2 fail: {g2['gate2_reason']}"


def rank_score(g1: dict, g2: dict) -> float:
    """Higher = better candidate. Used only among PROCEED methods."""
    if not (g1["gate1_pass"] and g2["gate2_pass"]):
        return -1.0
    novelty = 1.0 - abs(g2["mean_cosine_grad_gt_vs_kd"])
    signal = g1["teacher_advantage"]
    strength = g2["norm_ratio_kd_over_gt"]
    return novelty * signal * strength


def print_summary_table(rows: list[dict]) -> None:
    print("\n" + "=" * 100)
    print("KD METHOD GATE SUMMARY")
    print("=" * 100)
    header = (
        f"{'Method':<22} {'Verdict':<8} {'G1':^4} {'G2':^4} "
        f"{'cosine':>8} {'norm_r':>8} {'te_adv':>10} {'st-te gap':>10} {'λ_eq':>8}"
    )
    print(header)
    print("-" * len(header))
    for r in rows:
        g1 = "Y" if r["gate1"]["gate1_pass"] else "N"
        g2 = "Y" if r["gate2"]["gate2_pass"] else "N"
        print(
            f"{r['method_id']:<22} {r['verdict']:<8} {g1:^4} {g2:^4} "
            f"{r['gate2']['mean_cosine_grad_gt_vs_kd']:>8.3f} "
            f"{r['gate2']['norm_ratio_kd_over_gt']:>8.3f} "
            f"{r['gate1']['teacher_advantage']:>10.4f} "
            f"{r['gate1']['student_teacher_gap']:>10.4f} "
            f"{r['gate2']['lambda_for_equal_gradient']:>8.3f}"
        )
    print("-" * len(header))
    proceed = [r for r in rows if r["verdict"] == "PROCEED"]
    if proceed:
        best = max(proceed, key=lambda r: r["rank_score"])
        print(f"\nRecommended for Stage B probe: {best['method_id']} (rank_score={best['rank_score']:.4f})")
        print(f"  Suggested starting λ ≈ {best['gate2']['lambda_for_equal_gradient']:.2f} (equal-gradient heuristic)")
    else:
        print("\nNo method passed both gates. Do not start full KD training.")
        print("Consider architecture changes or a different teacher signal before another 13h run.")
    print()


def main() -> None:
    args = parse_args()
    t0 = time.time()

    device = torch.device(
        args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu"
    )
    hr_dir = PROJECT_ROOT / args.hr_dir
    if not hr_dir.exists():
        raise FileNotFoundError(f"HR dir not found: {hr_dir}")

    print(f"Device: {device}")
    student, student_src = _resolve_student(args, device)
    print(f"Student: {student_src}")

    print("Loading teacher (SwinIR)...")
    teacher = load_swinir_classical_x4(device)
    for p in teacher.parameters():
        p.requires_grad = False
    teacher.eval()

    all_methods = _build_methods(device)
    method_ids = args.methods or list(all_methods.keys())
    unknown = [m for m in method_ids if m not in all_methods]
    if unknown:
        raise ValueError(f"Unknown methods: {unknown}. Available: {list(all_methods.keys())}")

    rows = []
    for mid in method_ids:
        spec = all_methods[mid]
        print(f"\n--- {mid}: {spec.description} ---")
        g1 = gate1_signal_availability(
            student,
            teacher,
            spec,
            hr_dir,
            device,
            args.n_patches,
            args.hr_patch_size,
            args.scale,
            args.seed,
        )
        g2 = gate2_gradient_alignment(
            student,
            teacher,
            spec,
            hr_dir,
            device,
            args.n_batches,
            args.batch_size,
            args.hr_patch_size,
            args.scale,
            args.seed,
        )
        v, reason = verdict(g1, g2)
        row = {
            "method_id": mid,
            "description": spec.description,
            "gate1": g1,
            "gate2": g2,
            "verdict": v,
            "verdict_reason": reason,
            "rank_score": rank_score(g1, g2),
        }
        rows.append(row)
        print(f"  Gate1: {'PASS' if g1['gate1_pass'] else 'FAIL'} — {g1['gate1_reason']}")
        print(f"  Gate2: {'PASS' if g2['gate2_pass'] else 'FAIL'} — {g2['gate2_reason']}")
        print(f"  Verdict: {v}")

    rows.sort(key=lambda r: r["rank_score"], reverse=True)
    print_summary_table(rows)

    out_path = PROJECT_ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "thresholds": {
            "gate1_min_teacher_advantage": GATE1_MIN_TEACHER_ADVANTAGE,
            "gate1_min_student_teacher_gap": GATE1_MIN_STUDENT_TEACHER_GAP,
            "gate2_cosine_max": GATE2_COSINE_MAX,
            "gate2_min_norm_ratio": GATE2_MIN_NORM_RATIO,
        },
        "student_source": student_src,
        "device": str(device),
        "elapsed_sec": time.time() - t0,
        "methods": rows,
        "recommended": next((r["method_id"] for r in rows if r["verdict"] == "PROCEED"), None),
    }
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"Wrote {out_path} ({payload['elapsed_sec']:.1f}s)")


if __name__ == "__main__":
    main()
