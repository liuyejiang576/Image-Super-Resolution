"""Model loading helpers for SR evaluation and training."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from models import FSRCNN, MobileSRNet


def build_model_from_config(cfg: dict) -> nn.Module:
    data_cfg = cfg["dataset"]
    model_cfg = cfg["model"]
    model_type = str(model_cfg.get("type", "fsrcnn")).lower()

    if model_type == "mobile_srnet":
        return MobileSRNet(
            scale_factor=int(data_cfg["scale"]),
            num_channels=int(model_cfg["num_channels"]),
            feat=int(model_cfg["feat"]),
            num_blocks=int(model_cfg["num_blocks"]),
        )
    if model_type == "fsrcnn" or "d" in model_cfg:
        return FSRCNN(
            scale_factor=int(data_cfg["scale"]),
            num_channels=int(model_cfg.get("num_channels", 3)),
            d=int(model_cfg.get("d", 56)),
            s=int(model_cfg.get("s", 12)),
            m=int(model_cfg.get("m", 4)),
        )
    raise ValueError(f"Unsupported model type: {model_type}")


def load_checkpoint_model(
    checkpoint_path: Path,
    device: torch.device,
    scale: int = 4,
) -> tuple[nn.Module, dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    cfg = checkpoint.get("config", {})
    if not cfg:
        raise ValueError(f"Checkpoint missing config: {checkpoint_path}")
    data_cfg = cfg.setdefault("dataset", {"scale": scale})
    data_cfg.setdefault("scale", scale)
    model = build_model_from_config(cfg).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, cfg
