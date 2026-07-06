"""SR metric helpers for PSNR/SSIM evaluation on Y channel."""

from __future__ import annotations

import math

import numpy as np
from skimage.metrics import structural_similarity


def rgb_to_y_channel(rgb: np.ndarray) -> np.ndarray:
    """Convert RGB uint8 image to Y channel in YCbCr space."""
    rgb = rgb.astype(np.float64)
    return (65.738 * rgb[..., 0] + 129.057 * rgb[..., 1] + 25.064 * rgb[..., 2]) / 256.0 + 16.0


def crop_border(img: np.ndarray, border: int) -> np.ndarray:
    if border <= 0:
        return img
    return img[border:-border, border:-border]


def compute_psnr(img1: np.ndarray, img2: np.ndarray) -> float:
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float("inf")
    return 20.0 * math.log10(255.0 / math.sqrt(mse))


def compute_ssim(img1: np.ndarray, img2: np.ndarray) -> float:
    return float(
        structural_similarity(
            img1,
            img2,
            data_range=255.0,
            gaussian_weights=True,
            sigma=1.5,
            use_sample_covariance=False,
        )
    )
