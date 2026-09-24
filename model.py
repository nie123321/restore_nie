"""Star-A: A's three-scale U-Net with two full Star Blocks.

LayerNorm2d and GatedConvBlock are copied from
`endo_enhancement_demo/model.py`. Star blocks live in star_blocks.py.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from star_blocks import StarBlock


class LayerNorm2d(nn.Module):
    """Copied from endo_enhancement_demo/model.py. eps=1e-6, not Star's 1e-5."""

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


def resize(tensor: Tensor, size: tuple[int, int]) -> Tensor:
    if tensor.shape[-2:] == size:
        return tensor
    return F.interpolate(tensor, size=size, mode="bilinear", align_corners=False)


class StarA(nn.Module):
    """Direct residual U-Net. Star Blocks replace A's encoder2+spectral and decoder1."""

    def __init__(
        self,
        arch_version: str = "star_a_v1",
        widths: tuple[int, int, int] = (24, 48, 96),
        star_locations: tuple[str, str] = ("bottleneck", "decoder1"),
        ffn_expansion_factor: float = 3.0,
        output_mode: str = "direct",
    ):
        super().__init__()
        if arch_version != "star_a_v1":
            raise ValueError(f"Unsupported architecture: {arch_version}")
        if list(widths) != [24, 48, 96]:
            raise ValueError("Star-A v1 widths are fixed to 24/48/96")
        if list(star_locations) != ["bottleneck", "decoder1"]:
            raise ValueError("Star-A v1 places Star Blocks only at bottleneck and decoder1")
        if output_mode != "direct":
            raise ValueError("Star-A is fixed to output_mode=direct")
        self.arch_version = arch_version
        self.widths = tuple(widths)
        self.star_locations = tuple(star_locations)
        self.ffn_expansion_factor = ffn_expansion_factor
        self.output_mode = output_mode
        w0, w1, w2 = self.widths
        self.stem = nn.Conv2d(3, w0, 3, padding=1)
        self.enc0 = GatedConvBlock(w0)
        self.down1 = nn.Conv2d(w0, w1, 3, stride=2, padding=1)
        self.enc1 = GatedConvBlock(w1)
        self.down2 = nn.Conv2d(w1, w2, 3, stride=2, padding=1)
        self.bottleneck = StarBlock(w2, ffn_expansion_factor)
        self.fuse1 = nn.Conv2d(w2 + w1, w1, 1)
        self.dec1 = StarBlock(w1, ffn_expansion_factor)
        self.fuse0 = nn.Conv2d(w1 + w0, w0, 1)
        self.dec0 = GatedConvBlock(w0)
        self.head = nn.Conv2d(w0, 3, 3, padding=1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def config(self) -> dict:
        star = {
            "ffn_expansion_factor": self.ffn_expansion_factor,
            "window": 8,
            "bias": False,
            "spatial_depthwise_bias": True,
            "basesize": None,
            "layernorm": "WithBias",
            "layernorm_eps": 1e-5,
        }
        return {
            "arch_version": self.arch_version,
            "widths": list(self.widths),
            "star_locations": list(self.star_locations),
            "star_block": star,
            "output_mode": self.output_mode,
            "gated_locations": ["encoder0", "encoder1", "decoder0"],
        }

    def forward(self, x: Tensor, return_debug: bool = False):
        if x.ndim != 4 or x.shape[1] != 3 or min(x.shape[-2:]) < 1:
            raise ValueError("Expected nonempty NCHW RGB tensor")
        e0 = self.enc0(self.stem(x))
        e1 = self.enc1(self.down1(e0))
        bottleneck = self.bottleneck(self.down2(e1))
        d1 = self.dec1(self.fuse1(torch.cat((resize(bottleneck, e1.shape[-2:]), e1), dim=1)))
        d0 = self.dec0(self.fuse0(torch.cat((resize(d1, e0.shape[-2:]), e0), dim=1)))
        residual = self.head(d0).float()
        output = x.float() + residual
        if not return_debug:
            return output
        return {"output": output, "e0": e0, "e1": e1, "bottleneck": bottleneck, "d1": d1, "d0": d0}
