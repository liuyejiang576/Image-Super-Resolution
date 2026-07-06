"""Load pretrained SwinIR classical SR model."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import torch
import torch.nn as nn

SWINIR_ROOT = Path(__file__).resolve().parents[2] / "third_party" / "SwinIR"
NETWORK_SWINIR = SWINIR_ROOT / "models" / "network_swinir.py"
DEFAULT_WEIGHT = SWINIR_ROOT / "weights" / "001_classicalSR_DIV2K_s48w8_SwinIR-M_x4.pth"


def _load_swinir_class() -> type[nn.Module]:
    spec = importlib.util.spec_from_file_location("network_swinir", NETWORK_SWINIR)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load SwinIR module from {NETWORK_SWINIR}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.SwinIR


def build_swinir_classical_x4(training_patch_size: int = 48) -> nn.Module:
    SwinIR = _load_swinir_class()
    return SwinIR(
        upscale=4,
        in_chans=3,
        img_size=training_patch_size,
        window_size=8,
        img_range=1.0,
        depths=[6, 6, 6, 6, 6, 6],
        embed_dim=180,
        num_heads=[6, 6, 6, 6, 6, 6],
        mlp_ratio=2,
        upsampler="pixelshuffle",
        resi_connection="1conv",
    )


def load_swinir_classical_x4(
    device: torch.device,
    weight_path: Path | None = None,
    training_patch_size: int = 48,
) -> nn.Module:
    weight_path = weight_path or DEFAULT_WEIGHT
    if not weight_path.exists():
        raise FileNotFoundError(f"SwinIR weights not found: {weight_path}")

    model = build_swinir_classical_x4(training_patch_size=training_patch_size)
    checkpoint = torch.load(weight_path, map_location=device)
    state = checkpoint["params"] if "params" in checkpoint else checkpoint
    model.load_state_dict(state, strict=True)
    model.eval().to(device)
    return model


@torch.no_grad()
def swinir_forward(model: nn.Module, lr: torch.Tensor, window_size: int = 8) -> torch.Tensor:
    """Pad LR input to window multiple, run SwinIR, crop back to scaled size."""
    _, _, h_old, w_old = lr.size()
    h_pad = (h_old // window_size + 1) * window_size - h_old
    w_pad = (w_old // window_size + 1) * window_size - w_old
    padded = torch.cat([lr, torch.flip(lr, [2])], 2)[:, :, : h_old + h_pad, :]
    padded = torch.cat([padded, torch.flip(padded, [3])], 3)[:, :, :, : w_old + w_pad]
    output = model(padded)
    scale = output.shape[-1] // lr.shape[-1]
    return output[..., : h_old * scale, : w_old * scale]
