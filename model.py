"""Prior-A: frozen A (direct, spectral off, gated fusion) plus two priors.

LayerNorm2d, GatedConvBlock and ConditionalSpectralBlock follow the frozen A
in endo_enhancement_demo/.../A_spatial_fusion_on_off_10k_seed100_20260924/code/model.py.
Shared modules are constructed first so their initialization matches that A.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from priors import CIConvW, PriorAdapter, six_luminance


class LayerNorm2d(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x: Tensor) -> Tensor:
        variance, mean = torch.var_mean(x, dim=1, keepdim=True, unbiased=False)
        return (x - mean) * torch.rsqrt(variance + 1e-6) * self.weight + self.bias


class GatedConvBlock(nn.Module):
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

    def forward(self, x: Tensor, gamma_l: Tensor | None = None) -> Tensor:
        a, b = self.depthwise(self.expand(self.norm1(x))).chunk(2, dim=1)
        feature = a * b
        residual = self.project(feature * self.channel_scale(feature))
        if gamma_l is None:
            x = x + self.scale1 * residual
        else:
            gain = (1.0 + gamma_l).to(dtype=residual.dtype, device=residual.device)
            x = x + self.scale1 * residual * gain
        a, b = self.ffn_in(self.norm2(x)).chunk(2, dim=1)
        return x + self.scale2 * self.ffn_out(a * b)


class ConditionalSpectralBlock(nn.Module):
    def __init__(self, channels: int, mode: str = "off", experts: int = 3,
                 patch_size: int = 8, fusion_mode: str = "gated"):
        super().__init__()
        if mode not in {"off", "static", "conditional"}:
            raise ValueError(f"Unsupported spectral mode: {mode}")
        if fusion_mode not in {"gated", "additive"}:
            raise ValueError(f"Unsupported fusion mode: {fusion_mode}")
        if experts < 2 or patch_size < 2:
            raise ValueError("experts and patch_size must be at least 2")
        self.mode, self.channels = mode, channels
        self.fusion_mode = fusion_mode
        self.experts, self.patch_size = experts, patch_size
        self.norm = LayerNorm2d(channels)
        self.project_in = nn.Conv2d(channels, 2 * channels, 1)
        self.depthwise = nn.Conv2d(2 * channels, 2 * channels, 3,
                                   padding=1, groups=2 * channels)
        self.frequency_norm = LayerNorm2d(channels)
        self.spatial_gate = nn.Conv2d(channels, channels, 3, padding=1, groups=channels)
        self.project_out = nn.Conv2d(channels, channels, 1)
        self.residual_scale = nn.Parameter(torch.full((1, channels, 1, 1), 0.1))
        if mode != "off":
            raise ValueError("Prior-A keeps spectral_mode=off and does not create Fourier parameters")

    def forward(self, x: Tensor, condition: Tensor, structure_bias: Tensor | None = None):
        del condition
        q, v = self.depthwise(self.project_in(self.norm(x))).chunk(2, dim=1)
        logits = self.spatial_gate(v)
        if structure_bias is not None:
            logits = logits + structure_bias.to(dtype=logits.dtype, device=logits.device)
        if self.fusion_mode == "gated":
            fused = self.frequency_norm(q) * (v * torch.sigmoid(logits))
        else:
            fused = self.frequency_norm(q) + logits
        return x + self.residual_scale * self.project_out(fused), None


class PriorA(nn.Module):
    def __init__(self, width: int = 24, spectral_mode: str = "off",
                 output_mode: str = "direct", fusion_mode: str = "gated"):
        super().__init__()
        if width != 24 or spectral_mode != "off" or output_mode != "direct" or fusion_mode != "gated":
            raise ValueError("Prior-A v1 is fixed to width=24, direct, spectral off, gated fusion")
        self.output_mode = output_mode
        self.max_log_gain = math.log(32.0)
        self.max_log_color = math.log(2.0)
        self.stem = nn.Conv2d(3, width, 3, padding=1)
        self.encoder0 = GatedConvBlock(width)
        self.down1 = nn.Conv2d(width, 2 * width, 3, stride=2, padding=1)
        self.encoder1 = GatedConvBlock(2 * width)
        self.down2 = nn.Conv2d(2 * width, 4 * width, 3, stride=2, padding=1)
        self.encoder2 = GatedConvBlock(4 * width)
        self.spectral = ConditionalSpectralBlock(4 * width, mode="off", fusion_mode="gated")
        self.fuse1 = nn.Conv2d(6 * width, 2 * width, 1)
        self.decoder1 = GatedConvBlock(2 * width)
        self.fuse0 = nn.Conv2d(3 * width, width, 1)
        self.decoder0 = GatedConvBlock(width)
        luminance = torch.tensor([0.2126, 0.7152, 0.0722])
        axis1 = torch.tensor([luminance[1], -luminance[0], 0.0])
        axis1 = F.normalize(axis1, dim=0)
        axis2 = F.normalize(torch.linalg.cross(luminance, axis1), dim=0)
        self.register_buffer("luminance_weights", luminance.view(1, 3, 1, 1))
        self.register_buffer("chroma_basis", torch.stack((axis1, axis2), dim=1))
        self.direct_head = nn.Conv2d(width, 3, 3, padding=1)
        nn.init.zeros_(self.direct_head.weight)
        nn.init.zeros_(self.direct_head.bias)
        self.structure_extractor = CIConvW()
        self.luma_adapter = PriorAdapter(6, 8, 4 * width)
        self.structure_adapter = PriorAdapter(1, 8, 4 * width)
        self.settings = {
            "arch_version": "prior_a_v1",
            "width": width,
            "output_mode": output_mode,
            "spectral_mode": spectral_mode,
            "fusion_mode": fusion_mode,
            "luma_names": ["mean", "rec709", "vmax", "lightness", "ycgco", "l2norm_scaled"],
            "eps_L": 1e-6,
            "l2_scale": "1/sqrt(3)",
            "gamma_limit": 0.5,
            "structure_type": "ciconv_w_rgb_float_v1",
            "structure_scale": 0.9,
            "structure_k": 3,
            "structure_eps": 1e-5,
            "structure_padding": "zero",
            "adapter_width": 8,
            "luma_site": "encoder2_first_residual",
            "structure_site": "spatial_fusion_gate_logits",
            "resize": "bilinear_align_corners_false",
        }

    def config(self) -> dict:
        return dict(self.settings)

    def _priors(self, rgb: Tensor, size: tuple[int, int]) -> tuple[Tensor, Tensor]:
        device_type = rgb.device.type if rgb.device.type in {"cpu", "cuda"} else "cpu"
        with torch.amp.autocast(device_type=device_type, enabled=False):
            luma = six_luminance(rgb.float())
            structure = self.structure_extractor(rgb.float())
            luma = F.interpolate(luma, size=size, mode="bilinear", align_corners=False)
            structure = F.interpolate(structure, size=size, mode="bilinear", align_corners=False)
            gamma = 0.5 * torch.tanh(self.luma_adapter(luma))
            bias = self.structure_adapter(structure)
        return gamma, bias

    def forward(self, x: Tensor, return_debug: bool = False):
        if x.ndim != 4 or x.shape[1] != 3 or min(x.shape[-2:]) < 1:
            raise ValueError("Expected nonempty NCHW RGB tensor")
        skip0 = self.encoder0(self.stem(x))
        skip1 = self.encoder1(self.down1(skip0))
        down = self.down2(skip1)
        gamma, structure_bias = self._priors(x, down.shape[-2:])
        encoded = self.encoder2(down, gamma)
        pooled = encoded.mean(dim=(-2, -1))
        color_gain = x.new_ones(x.shape[0], 3, 1, 1)
        log_gain_small = x.new_zeros(x.shape[0], 1, *encoded.shape[-2:])
        gain_mean = log_gain_small.mean(dim=(1, 2, 3))
        gain_rms = torch.sqrt(log_gain_small.square().mean(dim=(1, 2, 3)) + 1e-6)
        condition = torch.cat((pooled, gain_mean[:, None], gain_rms[:, None]), dim=1)
        bottom, spectral_weights = self.spectral(encoded, condition, structure_bias)
        feature = self.fuse1(torch.cat((F.interpolate(bottom, size=skip1.shape[-2:], mode="bilinear", align_corners=False), skip1), dim=1))
        feature = self.decoder1(feature)
        feature = self.fuse0(torch.cat((F.interpolate(feature, size=skip0.shape[-2:], mode="bilinear", align_corners=False), skip0), dim=1))
        decoded = self.decoder0(feature)
        residual = self.direct_head(decoded).float()
        output = x.float() + residual
        if not return_debug:
            return output
        return {
            "output": output,
            "encoder2": encoded,
            "spectral": bottom,
            "decoder0": decoded,
            "gamma_l": gamma,
            "structure_bias": structure_bias,
            "spectral_weights": spectral_weights,
        }
