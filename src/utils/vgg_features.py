"""Frozen VGG16 feature extractor for perceptual KD."""

from __future__ import annotations

import torch
import torch.nn as nn


class VGGFeatureExtractor(nn.Module):
    """Frozen VGG16 features up to relu3_3 or relu4_3."""

    SLICES = {
        "relu3_3": (0, 16),
        "relu4_3": (0, 23),
    }

    def __init__(self, layer: str = "relu3_3") -> None:
        super().__init__()
        if layer not in self.SLICES:
            raise ValueError(f"Unknown VGG layer {layer!r}; choose from {list(self.SLICES)}")
        from torchvision.models import VGG16_Weights, vgg16

        vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features
        start, end = self.SLICES[layer]
        self.slice = vgg[start:end].eval()
        for p in self.slice.parameters():
            p.requires_grad = False
        self.layer = layer
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = (x - self.mean) / self.std
        return self.slice(x)
