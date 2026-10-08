"""Assign coarse LL and signed high subbands separate restoration roles."""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from model import LayerNorm2d


def check_bands(low: Tensor, high: tuple[Tensor, Tensor, Tensor], channels: int) -> None:
    if low.ndim != 4 or low.shape[1] != channels or len(high) != 3:
        raise ValueError("Expected one C-channel LL and three signed high subbands")
    if any(band.shape != low.shape for band in high):
        raise ValueError("All four subbands must have the same NCHW shape")


class SignedDetailUpdates(nn.Module):
    """Three independent DW3 residuals, with identity initialization."""

    def __init__(self, channels: int):
        super().__init__()
        self.filters = nn.ModuleList([
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False)
            for _ in range(3)
        ])
        for layer in self.filters:
            nn.init.zeros_(layer.weight)

    def forward(self, high: tuple[Tensor, Tensor, Tensor]) -> tuple[Tensor, Tensor, Tensor]:
        return tuple(band.float() + layer(band.float())
                     for band, layer in zip(high, self.filters))


class CoarseLowFrequencyAffineMixer(nn.Module):
    """LL2 predicts its own spatial/channel gain and signed bias.

    Statistics come from the raw LL2, before channel normalization. No GT,
    illumination label, high-band attention, or separately scaled LL branch.
    """

    def __init__(self, channels: int):
        super().__init__()
        self.channels = channels
        self.norm = LayerNorm2d(channels)
        hidden = 2 * channels
        self.context_pw = nn.Conv2d(3 * channels, hidden, 1, bias=True)
        self.context_dw = nn.Conv2d(hidden, hidden, 5, padding=2,
                                    groups=hidden, bias=True)
        self.affine_head = nn.Conv2d(hidden, 2 * channels, 1, bias=True)
        nn.init.zeros_(self.affine_head.weight)
        nn.init.zeros_(self.affine_head.bias)
        self.high = SignedDetailUpdates(channels)

    def affine_parameters(self, low: Tensor) -> tuple[Tensor, Tensor]:
        value = low.float()
        variance, mean = torch.var_mean(value, dim=(-2, -1), keepdim=True, unbiased=False)
        deviation = torch.sqrt(variance + 1e-6)
        context = torch.cat((self.norm(value), mean.expand_as(value),
                             deviation.expand_as(value)), dim=1)
        hidden = F.gelu(self.context_dw(self.context_pw(context)))
        log_gain, bias = self.affine_head(hidden).chunk(2, dim=1)
        # Positive, bounded gain; bias stays signed. Head zeros yield gain=1,bias=0.
        gain = torch.exp(torch.tanh(log_gain))
        return gain, bias

    def forward(self, low: Tensor, high: tuple[Tensor, Tensor, Tensor]
                ) -> tuple[Tensor, tuple[Tensor, Tensor, Tensor]]:
        check_bands(low, high, self.channels)
        with torch.autocast(device_type=low.device.type, enabled=False):
            gain, bias = self.affine_parameters(low)
            updated_low = gain * low.float() + bias
            updated_high = self.high(high)
        return updated_low, updated_high


class FineDetailMixer(nn.Module):
    """Pass reconstructed LL1 through; refine only LH1, HL1, HH1."""

    def __init__(self, channels: int):
        super().__init__()
        self.channels = channels
        self.high = SignedDetailUpdates(channels)

    def forward(self, low: Tensor, high: tuple[Tensor, Tensor, Tensor]
                ) -> tuple[Tensor, tuple[Tensor, Tensor, Tensor]]:
        check_bands(low, high, self.channels)
        with torch.autocast(device_type=low.device.type, enabled=False):
            updated_high = self.high(high)
        # No second LL head that could overwrite the coarse restoration.
        return low, updated_high
