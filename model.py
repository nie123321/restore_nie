"""Deep-A: A's gated U-Net with one extra H/8 scale.

LayerNorm2d and GatedConvBlock are copied from
`endo_enhancement_demo/model.py`. SpatialFusion is the spatial path of
ConditionalSpectralBlock with mode=off (no Fourier / router).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class LayerNorm2d(nn.Module):
    """Copied from endo_enhancement_demo/model.py."""

    def __init__(self, channels: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x: Tensor) -> Tensor:
        variance, mean = torch.var_mean(x, dim=1, keepdim=True, unbiased=False)
        return (x - mean) * torch.rsqrt(variance + 1e-6) * self.weight + self.bias


class GatedConvBlock(nn.Module):
    """Copied from endo_enhancement_demo/model.py."""

    def __init__(self, channels: int):
        super().__init__()
        self.norm1 = LayerNorm2d(channels)
        self.expand = nn.Conv2d(channels, 2 * channels, 1)
        self.depthwise = nn.Conv2d(2 * channels, 2 * channels, 3,
                                   padding=1, groups=2 * channels)
        self.channel_scale = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                           nn.Conv2d(channels, channels, 1))
        self.project = nn.Conv2d(channels, channels, 1)
        self.norm2 = LayerNorm2d(channels)
        self.ffn_in = nn.Conv2d(channels, 2 * channels, 1)
        self.ffn_out = nn.Conv2d(channels, channels, 1)
        self.scale1 = nn.Parameter(torch.full((1, channels, 1, 1), 0.1))
        self.scale2 = nn.Parameter(torch.full((1, channels, 1, 1), 0.1))

    def forward(self, x: Tensor) -> Tensor:
        a, b = self.depthwise(self.expand(self.norm1(x))).chunk(2, dim=1)
        feature = a * b
        x = x + self.scale1 * self.project(feature * self.channel_scale(feature))
        a, b = self.ffn_in(self.norm2(x)).chunk(2, dim=1)
        return x + self.scale2 * self.ffn_out(a * b)


class SpatialFusion(nn.Module):
    """A's spatial fusion with Fourier filtering removed.

    Parameter names match ConditionalSpectralBlock's spatial subset so the
    same weights produce the same forward and backward values.
    """

    def __init__(self, channels: int):
        super().__init__()
        self.channels = channels
        self.norm = LayerNorm2d(channels)
        self.project_in = nn.Conv2d(channels, 2 * channels, 1)
        self.depthwise = nn.Conv2d(2 * channels, 2 * channels, 3,
                                   padding=1, groups=2 * channels)
        self.frequency_norm = LayerNorm2d(channels)
        self.spatial_gate = nn.Conv2d(channels, channels, 3, padding=1, groups=channels)
        self.project_out = nn.Conv2d(channels, channels, 1)
        self.residual_scale = nn.Parameter(torch.full((1, channels, 1, 1), 0.1))

    def forward(self, x: Tensor) -> Tensor:
        q, v = self.depthwise(self.project_in(self.norm(x))).chunk(2, dim=1)
        mixed = self.frequency_norm(q) * (v * torch.sigmoid(self.spatial_gate(v)))
        return x + self.residual_scale * self.project_out(mixed)


def _stack(channels: int, count: int) -> nn.Sequential:
    return nn.Sequential(*[GatedConvBlock(channels) for _ in range(count)])


def resize(tensor: Tensor, size: tuple[int, int]) -> Tensor:
    if tensor.shape[-2:] == size:
        return tensor
    return F.interpolate(tensor, size=size, mode="bilinear", align_corners=False)


class DeepA(nn.Module):
    """Direct residual U-Net: widths 24/48/96/192, one SpatialFusion at H/8."""

    def __init__(
        self,
        arch_version: str = "deep_a_v1",
        widths: tuple[int, int, int, int] = (24, 48, 96, 192),
        encoder_blocks: tuple[int, int, int, int] = (1, 2, 2, 2),
        decoder_blocks: tuple[int, int, int] = (2, 2, 1),
        output_mode: str = "direct",
        spectral_mode: str = "off",
    ):
        super().__init__()
        if arch_version != "deep_a_v1":
            raise ValueError(f"Unsupported architecture: {arch_version}")
        if output_mode != "direct" or spectral_mode != "off":
            raise ValueError("Deep-A is fixed to output_mode=direct and spectral_mode=off")
        if list(encoder_blocks) != [1, 2, 2, 2] or list(decoder_blocks) != [2, 2, 1]:
            raise ValueError("Deep-A v1 block counts are fixed")
        if list(widths) != [24, 48, 96, 192]:
            raise ValueError("Deep-A v1 widths are fixed to 24/48/96/192")
        self.arch_version = arch_version
        self.widths = tuple(widths)
        self.encoder_blocks = tuple(encoder_blocks)
        self.decoder_blocks = tuple(decoder_blocks)
        self.output_mode = output_mode
        self.spectral_mode = spectral_mode
        w0, w1, w2, w3 = self.widths
        self.stem = nn.Conv2d(3, w0, 3, padding=1)
        self.enc0 = _stack(w0, encoder_blocks[0])
        self.down1 = nn.Conv2d(w0, w1, 3, stride=2, padding=1)
        self.enc1 = _stack(w1, encoder_blocks[1])
        self.down2 = nn.Conv2d(w1, w2, 3, stride=2, padding=1)
        self.enc2 = _stack(w2, encoder_blocks[2])
        self.down3 = nn.Conv2d(w2, w3, 3, stride=2, padding=1)
        self.enc3 = _stack(w3, encoder_blocks[3])
        self.fusion = SpatialFusion(w3)
        self.fuse2 = nn.Conv2d(w3 + w2, w2, 1)
        self.dec2 = _stack(w2, decoder_blocks[0])
        self.fuse1 = nn.Conv2d(w2 + w1, w1, 1)
        self.dec1 = _stack(w1, decoder_blocks[1])
        self.fuse0 = nn.Conv2d(w1 + w0, w0, 1)
        self.dec0 = _stack(w0, decoder_blocks[2])
        self.head = nn.Conv2d(w0, 3, 3, padding=1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def config(self) -> dict:
        return {
            "arch_version": self.arch_version,
            "widths": list(self.widths),
            "encoder_blocks": list(self.encoder_blocks),
            "decoder_blocks": list(self.decoder_blocks),
            "output_mode": self.output_mode,
            "spectral_mode": self.spectral_mode,
        }

    def forward(self, x: Tensor, return_debug: bool = False):
        if x.ndim != 4 or x.shape[1] != 3 or min(x.shape[-2:]) < 1:
            raise ValueError("Expected nonempty NCHW RGB tensor")
        e0 = self.enc0(self.stem(x))
        e1 = self.enc1(self.down1(e0))
        e2 = self.enc2(self.down2(e1))
        e3 = self.enc3(self.down3(e2))
        bottleneck = self.fusion(e3)
        d2 = self.dec2(self.fuse2(torch.cat((resize(bottleneck, e2.shape[-2:]), e2), dim=1)))
        d1 = self.dec1(self.fuse1(torch.cat((resize(d2, e1.shape[-2:]), e1), dim=1)))
        d0 = self.dec0(self.fuse0(torch.cat((resize(d1, e0.shape[-2:]), e0), dim=1)))
        residual = self.head(d0).float()
        output = x.float() + residual
        if not return_debug:
            return output
        return {
            "output": output,
            "e0": e0, "e1": e1, "e2": e2, "e3": e3,
            "bottleneck": bottleneck, "d2": d2, "d1": d1, "d0": d0,
        }
