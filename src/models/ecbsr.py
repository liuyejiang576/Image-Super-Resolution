"""ECBSR: Edge-oriented Convolution Block Super-Resolution (Zhang et al., MM'21).

Default lab variant: **M10C16** — 10 trunk ECBs @ 16 channels, RGB ×4, PReLU, with_idt.
"""

from __future__ import annotations

import copy

import torch
import torch.nn as nn

from .ecb import ECB


class ECBSR(nn.Module):
    """LR-space ECB trunk + PixelShuffle upsampler (RGB residual via channel repeat)."""

    def __init__(
        self,
        scale_factor: int = 4,
        num_channels: int = 3,
        num_block: int = 10,
        num_channel: int = 16,
        with_idt: bool = True,
        act_type: str = "prelu",
        depth_multiplier: float = 2.0,
    ) -> None:
        super().__init__()
        self.scale_factor = scale_factor
        self.num_channels = num_channels
        self.num_block = num_block
        self.num_channel = num_channel
        self.with_idt = with_idt
        self.act_type = act_type

        backbone: list[nn.Module] = [
            ECB(
                num_channels,
                num_channel,
                depth_multiplier=depth_multiplier,
                act_type=act_type,
                with_idt=with_idt,
            )
        ]
        for _ in range(num_block):
            backbone.append(
                ECB(
                    num_channel,
                    num_channel,
                    depth_multiplier=depth_multiplier,
                    act_type=act_type,
                    with_idt=with_idt,
                )
            )
        backbone.append(
            ECB(
                num_channel,
                num_channels * scale_factor * scale_factor,
                depth_multiplier=depth_multiplier,
                act_type="linear",
                with_idt=with_idt,
            )
        )
        self.backbone = nn.Sequential(*backbone)
        self.upsampler = nn.PixelShuffle(scale_factor)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # repeat_interleave is dropped by PNNX; expand+reshape is equivalent and exportable.
        b, c, h, w = x.shape
        r2 = self.scale_factor * self.scale_factor
        shortcut = x.unsqueeze(2).expand(b, c, r2, h, w).reshape(b, c * r2, h, w)
        y = self.backbone(x) + shortcut
        return self.upsampler(y)


class FusedECB(nn.Module):
    """Single dense 3×3 (+ optional act) materialized from ECB.rep_params()."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        weight: torch.Tensor,
        bias: torch.Tensor,
        act: nn.Module | None,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        with torch.no_grad():
            self.conv.weight.copy_(weight)
            self.conv.bias.copy_(bias)
        self.act = copy.deepcopy(act) if act is not None else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv(x)
        if self.act is not None:
            y = self.act(y)
        return y


class FusedECBSR(nn.Module):
    """Deploy graph: plain Conv3×3 stack + residual + PixelShuffle (NCNN-friendly)."""

    def __init__(self, scale_factor: int, num_channels: int, backbone: nn.Sequential) -> None:
        super().__init__()
        self.scale_factor = scale_factor
        self.num_channels = num_channels
        self.backbone = backbone
        self.upsampler = nn.PixelShuffle(scale_factor)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        r2 = self.scale_factor * self.scale_factor
        shortcut = x.unsqueeze(2).expand(b, c, r2, h, w).reshape(b, c * r2, h, w)
        return self.upsampler(self.backbone(x) + shortcut)


def fuse_ecbsr(model: ECBSR) -> FusedECBSR:
    """Fold every ECB into a dense 3×3 for export / phone bench."""
    model.eval()
    fused_layers: list[nn.Module] = []
    for block in model.backbone:
        if not isinstance(block, ECB):
            raise TypeError(f"expected ECB, got {type(block)}")
        weight, bias = block.rep_params()
        fused_layers.append(
            FusedECB(
                block.in_channels,
                block.out_channels,
                weight.detach(),
                bias.detach(),
                block.act,
            )
        )
    return FusedECBSR(
        scale_factor=model.scale_factor,
        num_channels=model.num_channels,
        backbone=nn.Sequential(*fused_layers),
    )
