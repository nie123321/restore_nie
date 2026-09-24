"""Shared residual 3D LUTs mixed by predicted spatial weights.

Storage: delta_luts[k, output_channel, b_index, g_index, r_index]
grid last dim: (2*R-1, 2*G-1, 2*B-1). 5D grid_sample bilinear = trilinear.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def init_residual_luts(count: int, size: int, seed: int, amplitude: float = 1e-3) -> Tensor:
    generator = torch.Generator().manual_seed(seed)
    small = torch.randn(count, 3, 4, 4, 4, generator=generator) * amplitude
    return F.interpolate(small, size=(size, size, size), mode="trilinear", align_corners=True)


def sample_one_lut(lut: Tensor, rgb: Tensor) -> Tensor:
    """lut [3,D,D,D], rgb [B,3,H,W] in encoded [0,1] -> residual [B,3,H,W]."""
    batch, _, height, width = rgb.shape
    table = lut.unsqueeze(0).expand(batch, -1, -1, -1, -1)
    red, green, blue = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    grid = torch.stack((2 * red - 1, 2 * green - 1, 2 * blue - 1), dim=-1).unsqueeze(1)
    sampled = F.grid_sample(table, grid, mode="bilinear", padding_mode="border", align_corners=True)
    return sampled.squeeze(2)


def mix_residual_luts(luts: Tensor, rgb: Tensor, weights: Tensor) -> Tensor:
    """luts [K,3,D,D,D], weights [B,K,H,W] nonnegative and sum to 1."""
    residual = rgb.new_zeros(rgb.shape)
    for index in range(luts.shape[0]):
        residual = residual + weights[:, index : index + 1] * sample_one_lut(luts[index], rgb)
    return residual
