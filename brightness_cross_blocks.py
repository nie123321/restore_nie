"""Four fixed brightness maps and an independent bottom cross-attention block.

Adapted from SG-LLIE Cross_attention: main Q, prior K/V, channel attention.
The shared prior encoder, two-times FFN and 0.1 residual scales are adaptations.
"""
import math

import torch
from torch import nn
import torch.nn.functional as F


class FourBrightnessMaps(nn.Module):
    """Low RGB only; fixed [0,1] transforms, without per-image normalization."""

    def __init__(self):
        super().__init__()
        coordinates = torch.arange(-15, 16, dtype=torch.float32)
        kernel = torch.exp(-0.5 * (coordinates / 5.0).square())
        self.register_buffer("gaussian_kernel", kernel / kernel.sum())

    @torch.no_grad()
    def forward(self, image):
        with torch.autocast(device_type=image.device.type, enabled=False):
            rgb = image.float().clamp(0, 1)
            y = 0.299 * rgb[:, 0:1] + 0.587 * rgb[:, 1:2] + 0.114 * rgb[:, 2:3]
            kernel = self.gaussian_kernel.to(y)
            horizontal_mode = "reflect" if y.shape[-1] > 15 else "replicate"
            vertical_mode = "reflect" if y.shape[-2] > 15 else "replicate"
            smooth = F.conv2d(F.pad(y, (15, 15, 0, 0), mode=horizontal_mode),
                              kernel.view(1, 1, 1, 31))
            smooth = F.conv2d(F.pad(smooth, (0, 0, 15, 15), mode=vertical_mode),
                              kernel.view(1, 1, 31, 1))
            power = ((y + 0.02).pow(0.4) - 0.02**0.4) / (1.02**0.4 - 0.02**0.4)
            smooth_log = torch.log1p(20 * smooth) / math.log(21)
            nonlinear = torch.sin(0.5 * math.pi * y).clamp_min(0).pow(0.2)
            log = torch.log1p(20 * y) / math.log(21)
            return torch.cat((power, smooth_log, nonlinear, log), dim=1).clamp(0, 1)


class BrightnessCrossAttentionFFN(nn.Module):
    """Same channel cross-attention/FFN, accepting an already encoded prior."""

    def __init__(self, channels, norm_layer, heads=4):
        super().__init__()
        self._init_attention(channels, norm_layer, heads)

    def _init_attention(self, channels, norm_layer, heads):
        if channels % heads:
            raise ValueError("Cross-attention channels must be divisible by heads")
        self.heads = heads
        self.norm_main = norm_layer(channels)
        self.norm_prior = norm_layer(channels)
        self.q = nn.Conv2d(channels, channels, 1, bias=False)
        self.q_dw = nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False)
        self.kv = nn.Conv2d(channels, 2 * channels, 1, bias=False)
        self.kv_dw = nn.Conv2d(2 * channels, 2 * channels, 3,
                               padding=1, groups=2 * channels, bias=False)
        self.temperature = nn.Parameter(torch.ones(heads, 1, 1))
        self.project_out = nn.Conv2d(channels, channels, 1, bias=False)
        self.attention_scale = nn.Parameter(torch.full((1, channels, 1, 1), 0.1))
        self.norm_ffn = norm_layer(channels)
        self.ffn = nn.Sequential(
            nn.Conv2d(channels, 2 * channels, 1, bias=False), nn.GELU(),
            nn.Conv2d(2 * channels, 2 * channels, 3,
                      padding=1, groups=2 * channels, bias=False), nn.GELU(),
            nn.Conv2d(2 * channels, channels, 1, bias=False),
        )
        self.ffn_scale = nn.Parameter(torch.full((1, channels, 1, 1), 0.1))

    def forward(self, feature, prior):
        if prior.shape != feature.shape:
            raise ValueError("Prior and main bottom features must have matching B,C,H,W")
        batch, channels, height, width = feature.shape
        shape = (batch, self.heads, channels // self.heads, height * width)
        q = self.q_dw(self.q(self.norm_main(feature))).reshape(shape)
        k, v = self.kv_dw(self.kv(self.norm_prior(prior))).chunk(2, dim=1)
        k, v = k.reshape(shape), v.reshape(shape)
        with torch.autocast(device_type=feature.device.type, enabled=False):
            q = F.normalize(q.float(), dim=-1)
            k = F.normalize(k.float(), dim=-1)
            weights = ((q @ k.transpose(-2, -1)) * self.temperature.float()).softmax(dim=-1)
        attended = (weights.to(v.dtype) @ v).reshape(batch, channels, height, width)
        residual = self.project_out(attended)
        updated = feature + self.attention_scale.to(residual.dtype) * residual
        residual = self.ffn(self.norm_ffn(updated))
        return updated + self.ffn_scale.to(residual.dtype) * residual


class BottomBrightnessCrossAttention(BrightnessCrossAttentionFFN):
    """Preserve bottom-only parameter names, initialization and forward behavior."""

    def __init__(self, channels, norm_layer, heads=4):
        nn.Module.__init__(self)
        if channels % heads:
            raise ValueError("Cross-attention channels must be divisible by heads")
        self.heads = heads
        self.maps = FourBrightnessMaps()
        self.encoder = nn.Sequential(
            nn.Conv2d(4, 16, 3, stride=2, padding=1), nn.GELU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.GELU(),
            nn.Conv2d(32, channels, 1),
        )
        self._init_attention(channels, norm_layer, heads)

    def prior_pyramid(self, low_image):
        maps = self.maps(low_image)
        half = self.encoder[1](self.encoder[0](maps))
        quarter = self.encoder[4](self.encoder[3](self.encoder[2](half)))
        return {"maps": maps, "half": half, "bottom": quarter}

    def forward(self, feature, low_image, prior=None):
        if prior is None:
            prior = self.encoder(self.maps(low_image))
        return super().forward(feature, prior)
