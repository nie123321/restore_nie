"""Pixel L1 plus complex-spectrum L1, following StarIR's loss implementation.

StarIR uses fft2; UHDRes uses rfft2. Both average absolute errors of the
real/imaginary components, with the default (backward) FFT normalization.
This implementation fixes the full fft2 convention for the loss comparison.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F


FFT_LOSS_RECIPE = {
    "version": 1,
    "transform": "torch.fft.fft2",
    "dimensions": [-2, -1],
    "normalization": "backward",
    "representation": "real and imaginary components; equivalent to stack(..., dim=-1)",
    "reduction": "mean over batch, RGB, both spatial frequency axes, real/imaginary axis",
    "pixel_weight": 1.0,
    "precision": "FP32 with autocast disabled",
    "prediction": "raw unclamped encoded RGB; no GT-Mean alignment",
    "frequency_scope": "all frequencies, including DC; no low-frequency mask",
    "implementation_source": "https://github.com/c-yn/StarIR/blob/main/basicsr/models/losses/losses.py",
    "weight_source": "https://github.com/c-yn/StarIR/blob/main/Low_Light_Enhancement/Options/lol-v2s.yml",
    "related_formulation": "https://arxiv.org/html/2511.05009v1#S3.SS5",
}


def complex_fft_l1(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Compare raw RGB full spectra in FP32, with backward normalization."""
    if output.ndim != 4 or output.shape != target.shape:
        raise ValueError("FFT loss expects matching NCHW tensors.")
    with torch.autocast(device_type=output.device.type, enabled=False):
        output_fft = torch.fft.fft2(output.float(), dim=(-2, -1), norm="backward")
        target_fft = torch.fft.fft2(target.float(), dim=(-2, -1), norm="backward")
        return F.l1_loss(torch.view_as_real(output_fft), torch.view_as_real(target_fft))


def pixel_fft_l1(output: torch.Tensor, target: torch.Tensor,
                 fft_weight: float = 0.1) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Return pixel L1 + weight * FFT L1 and detached logging components.

The frequency error is component-wise complex L1, not magnitude-only L1 or
the complex modulus of the difference. FFT normalization is independent of
the model's internal amplitude module (which uses ortho).
"""
    if output.ndim != 4 or output.shape != target.shape:
        raise ValueError("Pixel/FFT loss expects matching NCHW tensors.")
    if not math.isfinite(fft_weight) or fft_weight < 0:
        raise ValueError("FFT loss weight must be finite and nonnegative.")
    with torch.autocast(device_type=output.device.type, enabled=False):
        output, target = output.float(), target.float()
        pixel = F.l1_loss(output, target)
        frequency = complex_fft_l1(output, target)
        weighted_frequency = fft_weight * frequency
        total = pixel + weighted_frequency
    return total, {
        "loss_pixel_l1": pixel.detach(),
        "loss_fft": frequency.detach(),
        "loss_fft_weighted": weighted_frequency.detach(),
    }


def gt_mean_fft_l1(output: torch.Tensor, target: torch.Tensor,
                   sigma: float = 0.1, fft_weight: float = 0.1
                   ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """GT-Mean L1 plus FFT L1 on raw output; only the pixel term aligns brightness."""
    from gt_mean_loss import gt_mean_l1

    if not math.isfinite(fft_weight) or fft_weight < 0:
        raise ValueError("FFT loss weight must be finite and nonnegative.")
    base, parts = gt_mean_l1(output, target, sigma=sigma)
    frequency = complex_fft_l1(output, target)
    weighted_frequency = fft_weight * frequency
    return base + weighted_frequency, {
        **parts,
        "loss_gt_mean": base.detach(),
        "loss_fft": frequency.detach(),
        "loss_fft_weighted": weighted_frequency.detach(),
    }
