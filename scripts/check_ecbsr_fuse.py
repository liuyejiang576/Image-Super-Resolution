#!/usr/bin/env python3
"""Smoke-test ECBSR-M10C16: forward shapes, train/eval fuse parity, fused module."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from models.ecbsr import ECBSR, fuse_ecbsr  # noqa: E402


def main() -> None:
    torch.manual_seed(0)
    model = ECBSR(
        scale_factor=4,
        num_channels=3,
        num_block=10,
        num_channel=16,
        with_idt=True,
        act_type="prelu",
    )
    x = torch.randn(2, 3, 48, 48)

    model.train()
    y_train = model(x)
    assert y_train.shape == (2, 3, 192, 192), y_train.shape

    model.eval()
    with torch.no_grad():
        y_eval = model(x)
        fused = fuse_ecbsr(model)
        y_fused = fused(x)

    abs_eval = (y_train - y_eval).abs().max().item()  # train≠eval expected
    abs_fuse = (y_eval - y_fused).abs().max().item()
    print(f"params={sum(p.numel() for p in model.parameters())}")
    print(f"train_vs_eval max_abs={abs_eval:.3e} (multi-branch vs rep; not required equal)")
    print(f"eval_vs_fused_module max_abs={abs_fuse:.3e}")
    if abs_fuse > 1e-5:
        raise SystemExit(f"fuse mismatch too large: {abs_fuse}")
    print("ECBSR-M10C16 smoke OK")


if __name__ == "__main__":
    main()
