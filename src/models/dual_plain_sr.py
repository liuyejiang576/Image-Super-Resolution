"""DualStream / Plain C20N5 (ETDS-style train–deploy decoupling).

Train DualStream (detail 17 + low 3), deploy via exact merge to Plain:

    Conv 3→20 + ReLU
    → 5 × (Conv 20→20 + ReLU)
    → Conv 20→48
    → PixelShuffle×4

Cite: Chao et al., ETDS (CVPR 2023). This file is a budget-scaled adaptation
for FP16/NCNN probes, not a claim of a new backbone family.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class _DualMidBlock(nn.Module):
    """One dual-stream stage: detail gets low injection; low updates alone."""

    def __init__(self, detail_ch: int, low_ch: int) -> None:
        super().__init__()
        self.conv_xx = nn.Conv2d(detail_ch, detail_ch, 3, padding=1)
        self.conv_rx = nn.Conv2d(low_ch, detail_ch, 3, padding=1)
        self.conv_rr = nn.Conv2d(low_ch, low_ch, 3, padding=1)

    def forward(self, x: torch.Tensor, r: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = F.relu(self.conv_xx(x) + self.conv_rx(r), inplace=False)
        r = F.relu(self.conv_rr(r), inplace=False)
        return x, r


class DualStreamSR(nn.Module):
    """ETDS-style dual stream. ``forward`` returns SR; pass ``return_aux=True`` for aux."""

    supports_aux_loss = True

    def __init__(
        self,
        scale_factor: int = 4,
        num_channels: int = 3,
        detail_channels: int = 17,
        low_channels: int = 3,
        num_mid: int = 5,
    ) -> None:
        super().__init__()
        if scale_factor < 2:
            raise ValueError(f"scale_factor must be >= 2, got {scale_factor}")
        if detail_channels < 1 or low_channels < 1 or num_mid < 1:
            raise ValueError("detail_channels, low_channels, num_mid must be positive")

        self.scale_factor = scale_factor
        self.num_channels = num_channels
        self.detail_channels = detail_channels
        self.low_channels = low_channels
        self.num_mid = num_mid
        self.deploy_channels = detail_channels + low_channels
        out_ch = num_channels * scale_factor * scale_factor

        self.head_x = nn.Conv2d(num_channels, detail_channels, 3, padding=1)
        self.head_r = nn.Conv2d(num_channels, low_channels, 3, padding=1)
        self.mids = nn.ModuleList(
            [_DualMidBlock(detail_channels, low_channels) for _ in range(num_mid)]
        )
        self.tail_x = nn.Conv2d(detail_channels, out_ch, 3, padding=1)
        self.tail_r = nn.Conv2d(low_channels, out_ch, 3, padding=1)
        self.upsampler = nn.PixelShuffle(scale_factor)

    def forward(
        self, x: torch.Tensor, return_aux: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        detail = F.relu(self.head_x(x), inplace=False)
        low = F.relu(self.head_r(x), inplace=False)
        for block in self.mids:
            detail, low = block(detail, low)
        pre_x = self.tail_x(detail)
        pre_r = self.tail_r(low)
        sr = self.upsampler(pre_x + pre_r)
        if return_aux:
            sr_aux = self.upsampler(pre_r)
            return sr, sr_aux
        return sr


class PlainSR(nn.Module):
    """Single-stream plain deploy graph (also the Plain-C20N5 train model)."""

    supports_aux_loss = False

    def __init__(
        self,
        scale_factor: int = 4,
        num_channels: int = 3,
        num_channel: int = 20,
        num_mid: int = 5,
    ) -> None:
        super().__init__()
        if scale_factor < 2:
            raise ValueError(f"scale_factor must be >= 2, got {scale_factor}")
        if num_channel < 1 or num_mid < 1:
            raise ValueError("num_channel and num_mid must be positive")

        self.scale_factor = scale_factor
        self.num_channels = num_channels
        self.num_channel = num_channel
        self.num_mid = num_mid
        out_ch = num_channels * scale_factor * scale_factor

        layers: list[nn.Module] = [
            nn.Conv2d(num_channels, num_channel, 3, padding=1),
            nn.ReLU(inplace=True),
        ]
        for _ in range(num_mid):
            layers.append(nn.Conv2d(num_channel, num_channel, 3, padding=1))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(num_channel, out_ch, 3, padding=1))
        self.backbone = nn.Sequential(*layers)
        self.upsampler = nn.PixelShuffle(scale_factor)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.upsampler(self.backbone(x))


def fuse_dual_stream_sr(model: DualStreamSR) -> PlainSR:
    """Exact ETDS-style merge of dual stream into Plain deploy graph."""
    model.eval()
    d = model.detail_channels
    r = model.low_channels
    c = model.deploy_channels
    assert c == d + r

    plain = PlainSR(
        scale_factor=model.scale_factor,
        num_channels=model.num_channels,
        num_channel=c,
        num_mid=model.num_mid,
    )

    with torch.no_grad():
        # Head: stack detail/low along output channels → Conv 3→20
        head = plain.backbone[0]
        assert isinstance(head, nn.Conv2d)
        head.weight.zero_()
        head.bias.zero_()
        head.weight[:d].copy_(model.head_x.weight)
        head.weight[d:].copy_(model.head_r.weight)
        head.bias[:d].copy_(model.head_x.bias)
        head.bias[d:].copy_(model.head_r.bias)

        # Mid i sits at backbone indices 2, 4, ... (conv, relu pairs after head)
        for i, block in enumerate(model.mids):
            conv_idx = 2 + 2 * i
            mid = plain.backbone[conv_idx]
            assert isinstance(mid, nn.Conv2d)
            # z' = [xx  rx;  0  rr] @ [x; r]
            mid.weight.zero_()
            mid.bias.zero_()
            mid.weight[:d, :d].copy_(block.conv_xx.weight)
            mid.weight[:d, d:].copy_(block.conv_rx.weight)
            mid.weight[d:, :d].zero_()
            mid.weight[d:, d:].copy_(block.conv_rr.weight)
            mid.bias[:d].copy_(block.conv_xx.bias + block.conv_rx.bias)
            mid.bias[d:].copy_(block.conv_rr.bias)

        # Tail: pre = Wx @ x + Wr @ r → single Conv 20→48
        tail = plain.backbone[-1]
        assert isinstance(tail, nn.Conv2d)
        tail.weight.zero_()
        tail.bias.zero_()
        tail.weight[:, :d].copy_(model.tail_x.weight)
        tail.weight[:, d:].copy_(model.tail_r.weight)
        tail.bias.copy_(model.tail_x.bias + model.tail_r.bias)

    return plain


def count_plain_convs(model: PlainSR) -> int:
    return sum(1 for m in model.modules() if isinstance(m, nn.Conv2d))


def plain_param_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def conv_macs_at_lr(model: nn.Module, lr_h: int = 180, lr_w: int = 180) -> int:
    total = 0
    for m in model.modules():
        if not isinstance(m, nn.Conv2d):
            continue
        kh, kw = m.kernel_size
        cout = m.out_channels
        cin_g = m.in_channels // m.groups
        total += cout * cin_g * kh * kw * lr_h * lr_w
    return total


def expected_plain_budget(num_channel: int = 20, num_mid: int = 5) -> dict[str, int]:
    """Analytic deploy budget for Plain / fused DualStream C20N5."""
    # head 3→C + num_mid C→C + tail C→48
    params = (
        (3 * num_channel * 9 + num_channel)
        + num_mid * (num_channel * num_channel * 9 + num_channel)
        + (num_channel * 48 * 9 + 48)
    )
    convs = num_mid + 2
    macs_180 = (
        (3 * num_channel * 9 * 180 * 180)
        + num_mid * (num_channel * num_channel * 9 * 180 * 180)
        + (num_channel * 48 * 9 * 180 * 180)
    )
    return {
        "fused_convs": convs,
        "params": params,
        "macs_180": macs_180,
    }
