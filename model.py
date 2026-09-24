"""ColorWaveletNet: spatial residual LUTs plus a color-guided Haar U-Net.

GatedConvBlock and LayerNorm2d are copied from
`endo_enhancement_demo/model.py` so this project does not import that file.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from lut import init_residual_luts, mix_residual_luts
from wavelet import haar_dwt2d, haar_iwt2d


class LayerNorm2d(nn.Module):
    """Copied from endo_enhancement_demo/model.py: per-location channel norm."""

    def __init__(self, channels: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x: Tensor) -> Tensor:
        variance, mean = torch.var_mean(x, dim=1, keepdim=True, unbiased=False)
        return (x - mean) * torch.rsqrt(variance + 1e-6) * self.weight + self.bias


class GatedConvBlock(nn.Module):
    """Copied from endo_enhancement_demo/model.py: NAFNet-style gated block."""

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


def _stack(block: type[GatedConvBlock], channels: int, count: int) -> nn.Sequential:
    return nn.Sequential(*[block(channels) for _ in range(count)])


def _small_conv(conv: nn.Conv2d) -> nn.Conv2d:
    nn.init.normal_(conv.weight, mean=0.0, std=1e-3)
    if conv.bias is not None:
        nn.init.zeros_(conv.bias)
    return conv


def _zero_conv(conv: nn.Conv2d) -> nn.Conv2d:
    nn.init.zeros_(conv.weight)
    if conv.bias is not None:
        nn.init.zeros_(conv.bias)
    return conv


def pad_to_multiple(x: Tensor, multiple: int = 4) -> tuple[Tensor, int, int]:
    height, width = x.shape[-2:]
    pad_h = (multiple - height % multiple) % multiple
    pad_w = (multiple - width % multiple) % multiple
    if pad_h or pad_w:
        x = F.pad(x, (0, pad_w, 0, pad_h), mode="replicate")
    return x, height, width


def resize(tensor: Tensor, size: tuple[int, int]) -> Tensor:
    if tensor.shape[-2:] == size:
        return tensor
    return F.interpolate(tensor, size=size, mode="bilinear", align_corners=False)


class ContextEncoder(nn.Module):
    def __init__(self, context_width: int = 32, lut_count: int = 4):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, context_width, 3, padding=1),
            _stack(GatedConvBlock, context_width, 2),
        )
        self.down1 = nn.Sequential(
            nn.Conv2d(context_width, 2 * context_width, 3, stride=2, padding=1),
            _stack(GatedConvBlock, 2 * context_width, 2),
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(2 * context_width, 4 * context_width, 3, stride=2, padding=1),
            _stack(GatedConvBlock, 4 * context_width, 2),
        )
        self.proj1 = nn.Conv2d(2 * context_width, context_width, 1)
        self.proj2 = nn.Conv2d(4 * context_width, context_width, 1)
        self.local = nn.Sequential(
            nn.GELU(),
            nn.Conv2d(context_width, lut_count, 3, padding=1),
        )
        self.global_logits = nn.Linear(4 * context_width, lut_count)

    def forward(self, low_res: Tensor) -> tuple[Tensor, Tensor]:
        c0 = self.stem(low_res)
        c1 = self.down1(c0)
        c2 = self.down2(c1)
        mixed = c0 + resize(self.proj1(c1), c0.shape[-2:]) + resize(self.proj2(c2), c0.shape[-2:])
        local = self.local(mixed)
        pooled = F.adaptive_avg_pool2d(c2, 1).flatten(1)
        logits = local + self.global_logits(pooled)[:, :, None, None]
        return mixed, logits


class GuidedLowBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.condition = nn.Conv2d(35, channels, 3, padding=1)
        self.fuse = nn.Conv2d(2 * channels, channels, 1)
        self.affine = _zero_conv(nn.Conv2d(channels, 2 * channels, 1))
        self.body = GatedConvBlock(channels)

    def forward(self, projected: Tensor, original_low: Tensor, mixed: Tensor,
                residual: Tensor) -> tuple[Tensor, Tensor]:
        size = original_low.shape[-2:]
        query = F.gelu(self.condition(torch.cat((resize(mixed, size), resize(residual, size)), dim=1)))
        fused = self.fuse(torch.cat((projected, original_low), dim=1))
        scale, shift = self.affine(query).chunk(2, dim=1)
        fused = fused * (1.0 + 0.1 * torch.tanh(scale)) + 0.1 * torch.tanh(shift)
        return self.body(fused), query


class GuidedHighBlock(nn.Module):
    def __init__(self, channels: int, blocks: int = 1):
        super().__init__()
        if blocks < 1:
            raise ValueError("high-frequency branch needs at least one GatedConvBlock")
        high = 3 * channels
        self.prepare = nn.Conv2d(5 * channels, high, 1)
        self.body = _stack(GatedConvBlock, high, blocks)
        self.delta = _small_conv(nn.Conv2d(high, high, 3, padding=1))
        self.gate = nn.Conv2d(2 * channels, high, 1)

    def forward(self, high: Tensor, restored_low: Tensor, query: Tensor) -> Tensor:
        token = self.body(self.prepare(torch.cat((high, restored_low, query), dim=1)))
        gate = torch.sigmoid(self.gate(torch.cat((restored_low, query), dim=1)))
        return high + gate * self.delta(token)


class ColorWaveletNet(nn.Module):
    def __init__(
        self,
        arch_version: str = "color_wavelet_v1",
        widths: tuple[int, int, int] = (32, 64, 128),
        encoder_blocks: tuple[int, int] = (2, 2),
        bottleneck_blocks: int = 2,
        decoder_blocks: tuple[int, int] = (2, 2),
        high_blocks: int = 1,
        context_width: int = 32,
        lut_count: int = 4,
        lut_size: int = 17,
        lut_init_seed: int = 100,
        lut_init_amplitude: float = 1e-3,
    ):
        super().__init__()
        if arch_version != "color_wavelet_v1":
            raise ValueError(f"Unsupported architecture: {arch_version}")
        if bottleneck_blocks < 1 or high_blocks < 1:
            raise ValueError("bottleneck_blocks and high_blocks must be at least 1")
        self.arch_version = arch_version
        self.widths = tuple(widths)
        self.encoder_blocks = tuple(encoder_blocks)
        self.bottleneck_blocks = bottleneck_blocks
        self.decoder_blocks = tuple(decoder_blocks)
        self.high_blocks = high_blocks
        self.context_width = context_width
        self.lut_count = lut_count
        self.lut_size = lut_size
        self.lut_init_seed = lut_init_seed
        self.lut_init_amplitude = lut_init_amplitude
        w0, w1, w2 = self.widths
        self.context = ContextEncoder(context_width, lut_count)
        self.delta_luts = nn.Parameter(
            init_residual_luts(lut_count, lut_size, lut_init_seed, lut_init_amplitude)
        )
        self.in_conv = nn.Conv2d(6, w0, 3, padding=1)
        self.enc0 = _stack(GatedConvBlock, w0, encoder_blocks[0])
        self.to_w1 = nn.Conv2d(w0, w1, 1)
        self.enc1 = _stack(GatedConvBlock, w1, encoder_blocks[1])
        self.to_w2 = nn.Conv2d(w1, w2, 1)
        self.bottleneck = _stack(GatedConvBlock, w2, bottleneck_blocks)
        self.down_w1 = nn.Conv2d(w2, w1, 1)
        self.guide1 = GuidedLowBlock(w1)
        self.high1 = GuidedHighBlock(w1, high_blocks)
        self.dec1 = _stack(GatedConvBlock, w1, decoder_blocks[0])
        self.down_w0 = nn.Conv2d(w1, w0, 1)
        self.guide0 = GuidedLowBlock(w0)
        self.high0 = GuidedHighBlock(w0, high_blocks)
        self.dec0 = _stack(GatedConvBlock, w0, decoder_blocks[1])
        self.residual_head = _small_conv(nn.Conv2d(w0, 3, 3, padding=1))

    def config(self) -> dict:
        return {
            "arch_version": self.arch_version,
            "widths": list(self.widths),
            "encoder_blocks": list(self.encoder_blocks),
            "bottleneck_blocks": self.bottleneck_blocks,
            "decoder_blocks": list(self.decoder_blocks),
            "high_blocks": self.high_blocks,
            "context_width": self.context_width,
            "lut_count": self.lut_count,
            "lut_size": self.lut_size,
            "lut_init_seed": self.lut_init_seed,
            "lut_init_amplitude": self.lut_init_amplitude,
        }

    def _stage1(self, padded: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        low_res = F.interpolate(padded, scale_factor=0.25, mode="area")
        mixed, logits = self.context(low_res)
        device_type = padded.device.type if padded.device.type in {"cuda", "cpu"} else "cpu"
        with torch.amp.autocast(device_type=device_type, enabled=False):
            image = padded.float()
            weights = torch.softmax(resize(logits.float(), image.shape[-2:]), dim=1)
            coarse = image + mix_residual_luts(self.delta_luts.float(), image, weights)
        return coarse, weights, mixed

    def _stage2(self, padded: Tensor, coarse: Tensor, mixed: Tensor,
                return_debug: bool) -> tuple[Tensor, dict]:
        residual = coarse - padded
        f0 = self.enc0(self.in_conv(torch.cat((padded, coarse), dim=1)))
        low0, high0 = haar_dwt2d(f0)
        f1 = self.enc1(self.to_w1(low0))
        low1, high1 = haar_dwt2d(f1)
        f2 = self.bottleneck(self.to_w2(low1))
        low1_hat, query1 = self.guide1(self.down_w1(f2), low1, mixed, residual)
        high1_hat = self.high1(high1, low1_hat, query1)
        up1 = self.dec1(haar_iwt2d(low1_hat, high1_hat))
        low0_hat, query0 = self.guide0(self.down_w0(up1), low0, mixed, residual)
        high0_hat = self.high0(high0, low0_hat, query0)
        up0 = self.dec0(haar_iwt2d(low0_hat, high0_hat))
        detail = {"query0": query0, "query1": query1, "low0": low0, "low1": low1,
                  "high0": high0, "high1": high1, "low0_hat": low0_hat, "low1_hat": low1_hat,
                  "high0_hat": high0_hat, "high1_hat": high1_hat}
        if not return_debug:
            detail = {}
        return self.residual_head(up0), detail

    def forward(self, x: Tensor, return_aux: bool = False, return_debug: bool = False):
        padded, height, width = pad_to_multiple(x, 4)
        coarse, weights, mixed = self._stage1(padded)
        residual, debug = self._stage2(padded, coarse, mixed, return_debug)
        output = (coarse + residual)[..., :height, :width]
        coarse = coarse[..., :height, :width]
        weights = weights[..., :height, :width]
        if return_aux or return_debug:
            aux = {"output": output, "coarse": coarse, "weights": weights}
            aux.update(debug)
            return aux
        return output
