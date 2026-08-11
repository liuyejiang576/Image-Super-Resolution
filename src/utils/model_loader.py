"""Model loading helpers for SR evaluation and training."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from models import DualStreamSR, ECBSR, FSRCNN, MobileSRNet, PlainSR, SepResV2


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
    if model_type == "ecbsr":
        return ECBSR(
            scale_factor=int(data_cfg["scale"]),
            num_channels=int(model_cfg.get("num_channels", 3)),
            num_block=int(model_cfg.get("num_block", 10)),
            num_channel=int(model_cfg.get("num_channel", 16)),
            with_idt=bool(model_cfg.get("with_idt", True)),
            act_type=str(model_cfg.get("act_type", "prelu")),
            depth_multiplier=float(model_cfg.get("depth_multiplier", 2.0)),
        )
    if model_type == "sepres_v2":
        # Accept feat/num_blocks aliases; Spec fields are num_channel/num_block.
        num_channel = model_cfg.get("num_channel", model_cfg.get("feat"))
        num_block = model_cfg.get("num_block", model_cfg.get("num_blocks"))
        if num_channel is None or num_block is None:
            raise ValueError(
                "sepres_v2 requires num_channel (or feat) and num_block (or num_blocks)"
            )
        return SepResV2(
            scale_factor=int(data_cfg["scale"]),
            num_channels=int(model_cfg.get("num_channels", 3)),
            num_channel=int(num_channel),
            num_block=int(num_block),
            with_idt=bool(model_cfg.get("with_idt", True)),
            act_type=str(model_cfg.get("act_type", "prelu")),
            depth_multiplier=float(model_cfg.get("depth_multiplier", 2.0)),
            body_kind=str(model_cfg.get("body_kind", "ecb")),
        )
    if model_type in {"dual_stream_sr", "dualstream", "etds_dual"}:
        return DualStreamSR(
            scale_factor=int(data_cfg["scale"]),
            num_channels=int(model_cfg.get("num_channels", 3)),
            detail_channels=int(model_cfg.get("detail_channels", 17)),
            low_channels=int(model_cfg.get("low_channels", 3)),
            num_mid=int(model_cfg.get("num_mid", 5)),
        )
    if model_type in {"plain_sr", "plain_c20n5"}:
        return PlainSR(
            scale_factor=int(data_cfg["scale"]),
            num_channels=int(model_cfg.get("num_channels", 3)),
            num_channel=int(model_cfg.get("num_channel", 20)),
            num_mid=int(model_cfg.get("num_mid", 5)),
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
