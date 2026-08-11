"""Edge-oriented Convolution Block (ECB) with train-time multi-branch / infer fuse.

Adapted from ECBSR (Zhang et al., ACM MM 2021) / BasicSR `ecbsr_arch.py`
for this lab's RGB DIV2K pipeline.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SeqConv3x3(nn.Module):
    """Re-parameterizable 1×1 → 3×3 sequence (plain / Sobel / Laplacian)."""

    def __init__(
        self,
        seq_type: str,
        in_channels: int,
        out_channels: int,
        depth_multiplier: float = 1.0,
    ) -> None:
        super().__init__()
        self.seq_type = seq_type
        self.in_channels = in_channels
        self.out_channels = out_channels

        if seq_type == "conv1x1-conv3x3":
            self.mid_planes = int(out_channels * depth_multiplier)
            conv0 = nn.Conv2d(in_channels, self.mid_planes, kernel_size=1, padding=0)
            self.k0 = conv0.weight
            self.b0 = conv0.bias
            conv1 = nn.Conv2d(self.mid_planes, out_channels, kernel_size=3)
            self.k1 = conv1.weight
            self.b1 = conv1.bias
        elif seq_type in ("conv1x1-sobelx", "conv1x1-sobely", "conv1x1-laplacian"):
            conv0 = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
            self.k0 = conv0.weight
            self.b0 = conv0.bias
            self.scale = nn.Parameter(torch.randn(out_channels, 1, 1, 1) * 1e-3)
            self.bias = nn.Parameter(torch.randn(out_channels) * 1e-3)
            mask = torch.zeros(out_channels, 1, 3, 3)
            for i in range(out_channels):
                if seq_type == "conv1x1-sobelx":
                    mask[i, 0, 0, 0] = 1.0
                    mask[i, 0, 1, 0] = 2.0
                    mask[i, 0, 2, 0] = 1.0
                    mask[i, 0, 0, 2] = -1.0
                    mask[i, 0, 1, 2] = -2.0
                    mask[i, 0, 2, 2] = -1.0
                elif seq_type == "conv1x1-sobely":
                    mask[i, 0, 0, 0] = 1.0
                    mask[i, 0, 0, 1] = 2.0
                    mask[i, 0, 0, 2] = 1.0
                    mask[i, 0, 2, 0] = -1.0
                    mask[i, 0, 2, 1] = -2.0
                    mask[i, 0, 2, 2] = -1.0
                else:  # laplacian
                    mask[i, 0, 0, 1] = 1.0
                    mask[i, 0, 1, 0] = 1.0
                    mask[i, 0, 1, 2] = 1.0
                    mask[i, 0, 2, 1] = 1.0
                    mask[i, 0, 1, 1] = -4.0
            self.mask = nn.Parameter(mask, requires_grad=False)
        else:
            raise ValueError(f"Unsupported SeqConv3x3 type: {seq_type}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.seq_type == "conv1x1-conv3x3":
            y0 = F.conv2d(x, weight=self.k0, bias=self.b0, stride=1)
            y0 = F.pad(y0, (1, 1, 1, 1), "constant", 0)
            b0_pad = self.b0.view(1, -1, 1, 1)
            y0[:, :, 0:1, :] = b0_pad
            y0[:, :, -1:, :] = b0_pad
            y0[:, :, :, 0:1] = b0_pad
            y0[:, :, :, -1:] = b0_pad
            return F.conv2d(y0, weight=self.k1, bias=self.b1, stride=1)

        y0 = F.conv2d(x, weight=self.k0, bias=self.b0, stride=1)
        y0 = F.pad(y0, (1, 1, 1, 1), "constant", 0)
        b0_pad = self.b0.view(1, -1, 1, 1)
        y0[:, :, 0:1, :] = b0_pad
        y0[:, :, -1:, :] = b0_pad
        y0[:, :, :, 0:1] = b0_pad
        y0[:, :, :, -1:] = b0_pad
        return F.conv2d(
            y0,
            weight=self.scale * self.mask,
            bias=self.bias,
            stride=1,
            groups=self.out_channels,
        )

    def rep_params(self) -> tuple[torch.Tensor, torch.Tensor]:
        device = self.k0.device
        if self.seq_type == "conv1x1-conv3x3":
            weight = F.conv2d(self.k1, weight=self.k0.permute(1, 0, 2, 3))
            bias = torch.ones(1, self.mid_planes, 3, 3, device=device) * self.b0.view(
                1, -1, 1, 1
            )
            bias = F.conv2d(bias, weight=self.k1).view(-1) + self.b1
            return weight, bias

        tmp = self.scale * self.mask
        k1 = torch.zeros(self.out_channels, self.out_channels, 3, 3, device=device)
        for i in range(self.out_channels):
            k1[i, i, :, :] = tmp[i, 0, :, :]
        weight = F.conv2d(k1, weight=self.k0.permute(1, 0, 2, 3))
        bias = torch.ones(1, self.out_channels, 3, 3, device=device) * self.b0.view(
            1, -1, 1, 1
        )
        bias = F.conv2d(bias, weight=k1).view(-1) + self.bias
        return weight, bias


class ECB(nn.Module):
    """Edge-oriented Convolution Block (multi-branch train, single 3×3 infer)."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        depth_multiplier: float = 2.0,
        act_type: str = "prelu",
        with_idt: bool = False,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.act_type = act_type
        self.with_idt = bool(with_idt and in_channels == out_channels)

        self.conv3x3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv1x1_3x3 = SeqConv3x3(
            "conv1x1-conv3x3", in_channels, out_channels, depth_multiplier
        )
        self.conv1x1_sbx = SeqConv3x3("conv1x1-sobelx", in_channels, out_channels)
        self.conv1x1_sby = SeqConv3x3("conv1x1-sobely", in_channels, out_channels)
        self.conv1x1_lpl = SeqConv3x3("conv1x1-laplacian", in_channels, out_channels)

        if act_type == "prelu":
            self.act: nn.Module | None = nn.PReLU(num_parameters=out_channels)
        elif act_type == "relu":
            self.act = nn.ReLU(inplace=True)
        elif act_type == "linear":
            self.act = None
        else:
            raise ValueError(f"Unsupported act_type: {act_type}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            y = (
                self.conv3x3(x)
                + self.conv1x1_3x3(x)
                + self.conv1x1_sbx(x)
                + self.conv1x1_sby(x)
                + self.conv1x1_lpl(x)
            )
            if self.with_idt:
                y = y + x
        else:
            weight, bias = self.rep_params()
            y = F.conv2d(x, weight=weight, bias=bias, stride=1, padding=1)
        if self.act is not None:
            y = self.act(y)
        return y

    def rep_params(self) -> tuple[torch.Tensor, torch.Tensor]:
        w0, b0 = self.conv3x3.weight, self.conv3x3.bias
        w1, b1 = self.conv1x1_3x3.rep_params()
        w2, b2 = self.conv1x1_sbx.rep_params()
        w3, b3 = self.conv1x1_sby.rep_params()
        w4, b4 = self.conv1x1_lpl.rep_params()
        weight = w0 + w1 + w2 + w3 + w4
        bias = b0 + b1 + b2 + b3 + b4
        if self.with_idt:
            eye = torch.zeros(
                self.out_channels, self.out_channels, 3, 3, device=weight.device
            )
            for i in range(self.out_channels):
                eye[i, i, 1, 1] = 1.0
            weight = weight + eye
        return weight, bias
