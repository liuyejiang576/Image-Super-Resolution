"""SepResSR-v2 / PECSR: MobileSRNet outer shell + ECB (or plain) body.

Canonical contract (progress/track_b.md, IMPLEMENTATION §11):

    Conv3x3(3→C, no act)
    → N × ECB(C→C, PReLU, with_idt=True, depth_multiplier=2.0)
    → Conv3x3(C→48, no act)
    → PixelShuffle(4)

- ``N`` counts body blocks only (fused deploy = N+2 dense 3×3).
- No global LR RGB channel-repeat / pre-shuffle shortcut.
- Head and tail are plain convolutions, not ECBs.
- B5a P0: ``body_kind=plain3x3`` trains dense Conv+act matching fused ECB
  deploy (ECBSR paper plain-3×3 baseline); fuse is a weight copy.
- Do not wrap or subclass ``ECBSR``.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .ecb import ECB
from .ecbsr import FusedECB


class PlainBodyBlock(nn.Module):
    """Trainable dense 3×3 + act; same deploy shape as ``FusedECB``."""

    def __init__(self, channels: int, act_type: str = "prelu") -> None:
        super().__init__()
        self.in_channels = channels
        self.out_channels = channels
        self.act_type = act_type
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        if act_type == "prelu":
            self.act: nn.Module | None = nn.PReLU(num_parameters=channels)
        elif act_type == "relu":
            self.act = nn.ReLU(inplace=True)
        elif act_type == "linear":
            self.act = None
        else:
            raise ValueError(f"unsupported act_type for PlainBodyBlock: {act_type}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv(x)
        if self.act is not None:
            y = self.act(y)
        return y


class SepResV2(nn.Module):
    """LR-space SepRes shell with ECB or plain body; deploy via ``fuse_sepres_v2``."""

    def __init__(
        self,
        scale_factor: int = 4,
        num_channels: int = 3,
        num_channel: int = 16,
        num_block: int = 10,
        with_idt: bool = True,
        act_type: str = "prelu",
        depth_multiplier: float = 2.0,
        body_kind: str = "ecb",
    ) -> None:
        super().__init__()
        if scale_factor < 2:
            raise ValueError(f"scale_factor must be >= 2, got {scale_factor}")
        if num_channel < 1 or num_block < 1:
            raise ValueError("num_channel and num_block must be positive")
        body_kind = str(body_kind).lower()
        if body_kind not in {"ecb", "plain3x3"}:
            raise ValueError(f"body_kind must be 'ecb' or 'plain3x3', got {body_kind}")

        self.scale_factor = scale_factor
        self.num_channels = num_channels
        self.num_channel = num_channel
        self.num_block = num_block
        self.with_idt = with_idt
        self.act_type = act_type
        self.depth_multiplier = depth_multiplier
        self.body_kind = body_kind

        self.head = nn.Conv2d(num_channels, num_channel, kernel_size=3, padding=1)
        if body_kind == "ecb":
            self.body = nn.Sequential(
                *[
                    ECB(
                        num_channel,
                        num_channel,
                        depth_multiplier=depth_multiplier,
                        act_type=act_type,
                        with_idt=with_idt,
                    )
                    for _ in range(num_block)
                ]
            )
        else:
            # P0: no train-time multi-branch / explicit idt; matches fused deploy.
            self.body = nn.Sequential(
                *[PlainBodyBlock(num_channel, act_type=act_type) for _ in range(num_block)]
            )
        self.tail = nn.Conv2d(
            num_channel,
            num_channels * scale_factor * scale_factor,
            kernel_size=3,
            padding=1,
        )
        self.upsampler = nn.PixelShuffle(scale_factor)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # No global LR shortcut — that is the ECBSR difference.
        y = self.head(x)
        y = self.body(y)
        y = self.tail(y)
        return self.upsampler(y)


class FusedSepResV2(nn.Module):
    """Deploy graph: plain head + fused dense body + plain tail + PixelShuffle."""

    def __init__(
        self,
        scale_factor: int,
        num_channels: int,
        num_channel: int,
        num_block: int,
        head: nn.Conv2d,
        body: nn.Sequential,
        tail: nn.Conv2d,
    ) -> None:
        super().__init__()
        self.scale_factor = scale_factor
        self.num_channels = num_channels
        self.num_channel = num_channel
        self.num_block = num_block
        self.head = head
        self.body = body
        self.tail = tail
        self.upsampler = nn.PixelShuffle(scale_factor)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.head(x)
        y = self.body(y)
        y = self.tail(y)
        return self.upsampler(y)


def fuse_sepres_v2(model: SepResV2) -> FusedSepResV2:
    """Fold body to dense 3×3 stack; keep plain head/tail as-is.

    ECB body: ``rep_params()``. Plain body (B5a P0): copy Conv+act (already dense).
    """
    model.eval()
    fused_body: list[nn.Module] = []
    for block in model.body:
        if isinstance(block, ECB):
            weight, bias = block.rep_params()
            fused_body.append(
                FusedECB(
                    block.in_channels,
                    block.out_channels,
                    weight.detach(),
                    bias.detach(),
                    block.act,
                )
            )
        elif isinstance(block, PlainBodyBlock):
            fused_body.append(
                FusedECB(
                    block.in_channels,
                    block.out_channels,
                    block.conv.weight.detach(),
                    block.conv.bias.detach(),
                    block.act,
                )
            )
        else:
            raise TypeError(f"expected ECB or PlainBodyBlock in body, got {type(block)}")

    head = nn.Conv2d(
        model.head.in_channels,
        model.head.out_channels,
        kernel_size=3,
        padding=1,
    )
    tail = nn.Conv2d(
        model.tail.in_channels,
        model.tail.out_channels,
        kernel_size=3,
        padding=1,
    )
    with torch.no_grad():
        head.weight.copy_(model.head.weight)
        head.bias.copy_(model.head.bias)
        tail.weight.copy_(model.tail.weight)
        tail.bias.copy_(model.tail.bias)

    return FusedSepResV2(
        scale_factor=model.scale_factor,
        num_channels=model.num_channels,
        num_channel=model.num_channel,
        num_block=model.num_block,
        head=head,
        body=nn.Sequential(*fused_body),
        tail=tail,
    )


def count_fused_convs(model: nn.Module) -> int:
    return sum(1 for m in model.modules() if isinstance(m, nn.Conv2d))


def fused_param_count(model: nn.Module) -> int:
    """Params of fused deploy module (Conv weight/bias + body PReLU)."""
    return sum(p.numel() for p in model.parameters())


def conv_macs_at_lr(model: nn.Module, lr_h: int = 180, lr_w: int = 180) -> int:
    """Conv-only MACs at LR spatial size (PixelShuffle / act / add excluded)."""
    total = 0
    for m in model.modules():
        if not isinstance(m, nn.Conv2d):
            continue
        kh, kw = m.kernel_size
        cout = m.out_channels
        cin_g = m.in_channels // m.groups
        total += cout * cin_g * kh * kw * lr_h * lr_w
    return total


def expected_fused_budget(num_channel: int, num_block: int) -> dict[str, int]:
    """Analytic fused budget matching IMPLEMENTATION §11."""
    c, n = num_channel, num_block
    params = 9 * n * c * c + (460 + 2 * n) * c + 48
    macs = 180 * 180 * (27 * c + 9 * n * c * c + 432 * c)
    return {
        "fused_convs": n + 2,
        "params": params,
        "conv_macs_lr180": macs,
    }
