"""FSRCNN model implementation for x4 super-resolution."""

from __future__ import annotations

import torch.nn as nn


class FSRCNN(nn.Module):
    """Fast Super-Resolution Convolutional Neural Network."""

    def __init__(
        self,
        scale_factor: int = 4,
        num_channels: int = 3,
        d: int = 56,
        s: int = 12,
        m: int = 4,
    ) -> None:
        super().__init__()
        self.first_part = nn.Sequential(
            nn.Conv2d(num_channels, d, kernel_size=5, padding=2),
            nn.PReLU(d),
        )
        mid_layers = [nn.Conv2d(d, s, kernel_size=1), nn.PReLU(s)]
        for _ in range(m):
            mid_layers.extend(
                [
                    nn.Conv2d(s, s, kernel_size=3, padding=1),
                    nn.PReLU(s),
                ]
            )
        mid_layers.extend([nn.Conv2d(s, d, kernel_size=1), nn.PReLU(d)])
        self.mid_part = nn.Sequential(*mid_layers)
        self.last_part = nn.ConvTranspose2d(
            d,
            num_channels,
            kernel_size=9,
            stride=scale_factor,
            padding=4,
            output_padding=scale_factor - 1,
        )
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.normal_(module.weight, mean=0.0, std=0.001)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.ConvTranspose2d):
                nn.init.normal_(module.weight, mean=0.0, std=0.001)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x):
        x = self.first_part(x)
        x = self.mid_part(x)
        x = self.last_part(x)
        return x
