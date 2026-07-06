"""MobileSRNet: lightweight depthwise-separable SR model with PixelShuffle upsampling."""

from __future__ import annotations

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
