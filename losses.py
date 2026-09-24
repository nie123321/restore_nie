"""First-version joint loss and a numerically safe sRGB→CIE Lab (D65) map.

Lab conversion follows IEC 61966-2-1 sRGB decoding, D65 XYZ, and the CIE
piecewise f(t). Encoded RGB is not treated as linear light.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


# IEC 61966-2-1 / Bruce Lindbloom sRGB to XYZ, D65, 2°.
_SRGB_TO_XYZ = torch.tensor(
    [[0.4124564, 0.3575761, 0.1804375],
     [0.2126729, 0.7151522, 0.0721750],
     [0.0193339, 0.1191920, 0.9503041]],
    dtype=torch.float32,
)
_XN, _YN, _ZN = 0.95047, 1.00000, 1.08883
_DELTA = 6.0 / 29.0
_DELTA_CUBE = _DELTA ** 3
_DELTA_SCALE = 1.0 / (3.0 * _DELTA ** 2)


def _srgb_to_linear(encoded: Tensor) -> Tensor:
    """Both branches stay defined; unused pow is evaluated at the 0.04045 knot."""
    low = encoded / 12.92
    high = ((encoded.clamp_min(0.04045) + 0.055) / 1.055).pow(2.4)
    return torch.where(encoded <= 0.04045, low, high)


def _lab_f(ratio: Tensor) -> Tensor:
    """Cube root is never evaluated at 0, so black does not produce inf/NaN grads.

    d(t^{1/3})/dt diverges as t→0. torch.where still builds that backward for
    the unused branch, so the high branch is evaluated at t ≥ (6/29)^3.
    """
    low = _DELTA_SCALE * ratio + 4.0 / 29.0
    high = ratio.clamp_min(_DELTA_CUBE).pow(1.0 / 3.0)
    return torch.where(ratio > _DELTA_CUBE, high, low)


def srgb_to_cielab_d65(encoded_rgb: Tensor) -> Tensor:
    """encoded_rgb [B,3,H,W] in [0,1] → Lab [B,3,H,W]. Always FP32."""
    encoded_rgb = encoded_rgb.float()
    linear = _srgb_to_linear(encoded_rgb)
    matrix = _SRGB_TO_XYZ.to(device=encoded_rgb.device, dtype=torch.float32)
    xyz = torch.einsum("ij,bjhw->bihw", matrix, linear)
    fx = _lab_f(xyz[:, 0] / _XN)
    fy = _lab_f(xyz[:, 1] / _YN)
    fz = _lab_f(xyz[:, 2] / _ZN)
    lightness = 116.0 * fy - 16.0
    a_star = 500.0 * (fx - fy)
    b_star = 200.0 * (fy - fz)
    return torch.stack((lightness, a_star, b_star), dim=1)


def lut_smoothness(luts: Tensor) -> Tensor:
    """Mean squared adjacent differences along the three LUT grid axes, summed."""
    axis_b = (luts[:, :, 1:] - luts[:, :, :-1]).square().mean()
    axis_g = (luts[:, :, :, 1:] - luts[:, :, :, :-1]).square().mean()
    axis_r = (luts[:, :, :, :, 1:] - luts[:, :, :, :, :-1]).square().mean()
    return axis_b + axis_g + axis_r


def pyramid(image: Tensor) -> Tensor:
    height, width = image.shape[-2:]
    return F.interpolate(
        image, size=(max(1, height // 8), max(1, width // 8)), mode="area",
    )


def color_wavelet_losses(
    output: Tensor,
    coarse: Tensor,
    target: Tensor,
    luts: Tensor,
    lambda_coarse: float,
    lambda_ab: float,
    lambda_lut: float,
) -> tuple[Tensor, dict[str, float]]:
    device_type = output.device.type if output.device.type in {"cpu", "cuda"} else "cpu"
    with torch.amp.autocast(device_type=device_type, enabled=False):
        output = output.float()
        coarse = coarse.float()
        target = target.float()
        luts = luts.float()
        rgb = (output - target).abs().mean()
        coarse_term = (pyramid(coarse) - pyramid(target)).abs().mean()
        lab_y = srgb_to_cielab_d65(output.clamp(0, 1))
        lab_t = srgb_to_cielab_d65(target)
        ab = ((lab_y[:, 1:3] - lab_t[:, 1:3]) / 128.0).abs().mean()
        smooth = lut_smoothness(luts)
        total = rgb + lambda_coarse * coarse_term + lambda_ab * ab + lambda_lut * smooth
    parts = {
        "loss": float(total.detach()),
        "loss_rgb": float(rgb.detach()),
        "loss_coarse": float(coarse_term.detach()),
        "loss_ab": float(ab.detach()),
        "loss_lut": float(smooth.detach()),
    }
    return total, parts
