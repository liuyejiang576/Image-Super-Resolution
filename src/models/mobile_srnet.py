"""MobileSRNet: lightweight depthwise-separable SR model with PixelShuffle upsampling."""

from __future__ import annotations

import copy

import torch
import torch.nn as nn


class DepthwiseSeparableBlock(nn.Module):
    """Residual depthwise-separable block with ReLU6 for quantization-friendly activations."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False),
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.ReLU6(inplace=True),
        )

    def forward(self, x):
        return x + self.conv(x)


class FusedResidualBlock(nn.Module):
    """Algebraically fused DW+PW → dense 3×3, then ReLU6 (same residual form)."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.ReLU6(inplace=True),
        )

    def forward(self, x):
        return x + self.conv(x)


def fuse_dw_pw_to_dense(dw: nn.Conv2d, pw: nn.Conv2d) -> nn.Conv2d:
    """Fold depthwise 3×3 + pointwise 1×1 (no mid-activation) into one dense 3×3.

    K[o, i, u, v] = P[o, i] * D[i, u, v]
    """
    if dw.groups != dw.in_channels:
        raise ValueError("dw must be depthwise (groups == in_channels)")
    if dw.bias is not None or pw.bias is not None:
        raise ValueError("fuse assumes bias-free DW and PW")
    d = dw.weight.detach()  # [C, 1, 3, 3]
    p = pw.weight.detach()  # [C, C, 1, 1]
    # [O, I, 1, 1] * [1, I, 3, 3] -> [O, I, 3, 3]
    k = p * d.squeeze(1).unsqueeze(0)
    fused = nn.Conv2d(
        dw.in_channels,
        pw.out_channels,
        kernel_size=dw.kernel_size,
        padding=dw.padding,
        bias=False,
    )
    fused.weight.data.copy_(k)
    return fused


def fuse_mobile_srnet(model: MobileSRNet) -> MobileSRNet:
    """Return a copy of MobileSRNet with body blocks folded to dense 3×3."""
    fused = copy.deepcopy(model)
    new_blocks: list[nn.Module] = []
    for block in fused.body:
        if not isinstance(block, DepthwiseSeparableBlock):
            raise TypeError(f"expected DepthwiseSeparableBlock, got {type(block)}")
        dw, pw, _relu = block.conv[0], block.conv[1], block.conv[2]
        channels = dw.in_channels
        fb = FusedResidualBlock(channels)
        fb.conv[0] = fuse_dw_pw_to_dense(dw, pw)
        new_blocks.append(fb)
    fused.body = nn.Sequential(*new_blocks)
    return fused


class MobileSRNet(nn.Module):
    """Simple mobile-oriented SR network operating in LR feature space."""

    def __init__(
        self,
        scale_factor: int = 4,
        num_channels: int = 3,
        feat: int = 40,
        num_blocks: int = 6,
    ) -> None:
        super().__init__()
        self.head = nn.Conv2d(num_channels, feat, kernel_size=3, padding=1)
        self.body = nn.Sequential(
            *[DepthwiseSeparableBlock(feat) for _ in range(num_blocks)]
        )
        self.tail = nn.Sequential(
            nn.Conv2d(feat, num_channels * (scale_factor ** 2), kernel_size=3, padding=1),
            nn.PixelShuffle(scale_factor),
        )
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                n = module.in_channels * module.kernel_size[0] * module.kernel_size[1]
                if module.groups == module.in_channels:
                    n = module.in_channels * module.kernel_size[0] * module.kernel_size[1]
                else:
                    n = module.out_channels * module.kernel_size[0] * module.kernel_size[1]
                std = (2.0 / n) ** 0.5
                nn.init.normal_(module.weight, mean=0.0, std=std)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x):
        x = self.head(x)
        x = self.body(x)
        x = self.tail(x)
        return x
