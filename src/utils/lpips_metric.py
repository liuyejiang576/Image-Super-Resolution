"""Perceptual LPIPS metric helper."""

from __future__ import annotations

import numpy as np
import torch


class LPIPSMetric:
    def __init__(self, device: torch.device) -> None:
        import lpips

        self.model = lpips.LPIPS(net="alex").to(device)
        self.model.eval()
        self.device = device

    @torch.no_grad()
    def compute(self, pred_rgb: np.ndarray, true_rgb: np.ndarray) -> float:
        pred = torch.from_numpy(pred_rgb).permute(2, 0, 1).unsqueeze(0).float()
        true = torch.from_numpy(true_rgb).permute(2, 0, 1).unsqueeze(0).float()
        pred = pred.to(self.device) * 2.0 - 1.0
        true = true.to(self.device) * 2.0 - 1.0
        return float(self.model(pred, true).item())
