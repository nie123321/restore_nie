"""Full Star Block used by Star-A.

Adapted from c-yn/StarIR `basicsr/models/archs/StarIR_arch.py`
(retrieved 2026-09-23, SHA256 9ee5e3644df5ebb65b1ddf9b488d2a0832c00c879ffcac75e9144d519ad5d6e0).
Parameter names follow that file so the same state dict can be compared.
basesize=None. Sizes that are not multiples of 8 are replicate-padded only
inside the window FFT and cropped before the following LayerNorm or pooling.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class WithBiasLayerNormBody(nn.Module):
    """Official WithBias_LayerNorm. Input is [B, H*W, C]."""

    def __init__(self, dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))

    def forward(self, x: Tensor) -> Tensor:
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, keepdim=True, unbiased=False)
        return (x - mean) / torch.sqrt(var + 1e-5) * self.weight + self.bias


class WithBiasLayerNorm(nn.Module):
    """Official LayerNorm(WithBias) on NCHW features. eps=1e-5."""

    def __init__(self, dim: int):
        super().__init__()
        self.body = WithBiasLayerNormBody(dim)

    def forward(self, x: Tensor) -> Tensor:
        height, width = x.shape[-2:]
        flat = x.permute(0, 2, 3, 1).reshape(x.shape[0], height * width, x.shape[1])
        normalized = self.body(flat)
        return normalized.view(x.shape[0], height, width, x.shape[1]).permute(0, 3, 1, 2)


def window_fft_filter(x: Tensor, weight: Tensor) -> Tensor:
    """weight: [C, 1, 1, 8, 5]. FFT stays in FP32 with autocast disabled."""
    device_type = x.device.type if x.device.type in {"cpu", "cuda"} else "cpu"
    with torch.amp.autocast(device_type=device_type, enabled=False):
        value = x.float()
        height, width = value.shape[-2:]
        pad_h = (8 - height % 8) % 8
        pad_w = (8 - width % 8) % 8
        if pad_h or pad_w:
            value = F.pad(value, (0, pad_w, 0, pad_h), mode="replicate")
        padded_h, padded_w = value.shape[-2:]
        windows_h, windows_w = padded_h // 8, padded_w // 8
        patches = value.view(
            value.shape[0], value.shape[1], windows_h, 8, windows_w, 8,
        ).permute(0, 1, 2, 4, 3, 5).contiguous()
        spectrum = torch.fft.rfft2(patches) * weight.float()
        restored = torch.fft.irfft2(spectrum, s=(8, 8))
        merged = restored.permute(0, 1, 2, 4, 3, 5).contiguous().view(
            value.shape[0], value.shape[1], padded_h, padded_w,
        )
        return merged[..., :height, :width]


class SpatialOperation(nn.Module):
    """Official SMB. Depthwise convolution keeps bias=True."""

    def __init__(self, dim: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1, groups=dim),
            nn.Sigmoid(),
        )

    def forward(self, x: Tensor) -> Tensor:
        return x * self.block(x)


class StarModule(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.patch_size = 8
        self.dim = dim
        self.to_hidden = nn.Conv2d(dim, dim * 2, kernel_size=1, bias=False)
        self.to_hidden_dw = nn.Conv2d(
            dim * 2, dim * 2, kernel_size=3, stride=1, padding=1, groups=dim * 2, bias=False,
        )
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=False)
        self.norm = WithBiasLayerNorm(dim)
        self.fft_FSAS = nn.Parameter(torch.ones(dim, 1, 1, 8, 8 // 2 + 1))
        self.spatial = SpatialOperation(dim)
        self.plain_channel = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(dim, dim, kernel_size=1, bias=False),
        )

    def forward(self, x: Tensor) -> Tensor:
        hidden = self.to_hidden(x)
        query, value = self.to_hidden_dw(hidden).split([self.dim, self.dim], dim=1)
        filtered = self.norm(window_fft_filter(query, self.fft_FSAS))
        fused = self.spatial(value) * filtered
        mixed = self.plain_channel(fused) * fused
        return self.project_out(mixed)


class DFFN(nn.Module):
    def __init__(self, dim: int, ffn_expansion_factor: float = 3.0):
        super().__init__()
        hidden = int(dim * ffn_expansion_factor)
        self.patch_size = 8
        self.dim = dim
        self.project_in = nn.Conv2d(dim, hidden * 2, kernel_size=1, bias=False)
        self.dwconv = nn.Conv2d(
            hidden * 2, hidden * 2, kernel_size=3, stride=1, padding=1,
            groups=hidden * 2, bias=False,
        )
        self.fft = nn.Parameter(torch.ones(dim, 1, 1, 8, 8 // 2 + 1))
        self.project_out = nn.Conv2d(hidden, dim, kernel_size=1, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        projected = self.project_in(x)
        first, second = self.dwconv(projected).chunk(2, dim=1)
        gated = F.gelu(first) * second
        return window_fft_filter(self.project_out(gated), self.fft)


class StarBlock(nn.Module):
    def __init__(self, dim: int, ffn_expansion_factor: float = 3.0):
        super().__init__()
        self.dim = dim
        self.ffn_expansion_factor = ffn_expansion_factor
        self.norm1 = WithBiasLayerNorm(dim)
        self.attn = StarModule(dim)
        self.norm2 = WithBiasLayerNorm(dim)
        self.ffn = DFFN(dim, ffn_expansion_factor)

    def forward(self, x: Tensor) -> Tensor:
        updated = x + self.attn(self.norm1(x))
        return updated + self.ffn(self.norm2(updated))
