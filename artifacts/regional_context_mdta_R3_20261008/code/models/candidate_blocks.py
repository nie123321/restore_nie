"""Regional LL2 conditioning and independent bottom-attention depth candidates."""
from __future__ import annotations

import math
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from model import LayerNorm2d
from mdta_blocks import MDTAResidual
from wavelet_blocks import WaveletMDTAResidual
from models.baseline_a_blocks import ALL_OWN_SITES
from models.low_frequency_blocks import CoarseLowFrequencyAffineMixer


class RegionalLLContext(nn.Module):
    """Pool raw LL2 to 16 spatial tokens, exchange regions, restore LL2 size.

    This is spatial attention (16 by 16), unlike the existing MDTA channel
    attention. The unpooled LL2 and its local processor stay in the affine path.
    """

    def __init__(self, channels: int, grid: int = 4, heads: int = 4):
        super().__init__()
        hidden = 2 * channels
        if grid != 4 or heads != 4 or hidden % heads:
            raise ValueError("This candidate fixes a 4x4 grid and four attention heads")
        self.channels, self.hidden = channels, hidden
        self.grid, self.heads = grid, heads
        self.pool = nn.AdaptiveAvgPool2d((grid, grid))
        self.input_project = nn.Linear(channels, hidden)
        self.position = nn.Parameter(torch.empty(1, grid * grid, hidden))
        nn.init.trunc_normal_(self.position, std=0.02)
        self.norm = nn.LayerNorm(hidden)
        self.qkv = nn.Linear(hidden, 3 * hidden)
        self.output_project = nn.Linear(hidden, hidden)

    def token_attention(self, low: Tensor) -> tuple[Tensor, Tensor]:
        with torch.autocast(device_type=low.device.type, enabled=False):
            pooled = self.pool(low.float())
            batch = pooled.shape[0]
            count = self.grid * self.grid
            regions = pooled.flatten(2).transpose(1, 2)
            tokens = self.input_project(regions) + self.position
            qkv = self.qkv(self.norm(tokens)).reshape(
                batch, count, 3, self.heads, self.hidden // self.heads)
            query, key, value = qkv.permute(2, 0, 3, 1, 4).unbind(0)
            weights = ((query @ key.transpose(-2, -1)) /
                       math.sqrt(self.hidden // self.heads)).softmax(dim=-1)
            exchanged = (weights @ value).transpose(1, 2).reshape(batch, count, self.hidden)
            # Preserve each region's representation while importing other regions.
            return tokens + self.output_project(exchanged), weights

    def forward(self, low: Tensor) -> Tensor:
        tokens, _ = self.token_attention(low)
        context = tokens.transpose(1, 2).reshape(
            low.shape[0], self.hidden, self.grid, self.grid)
        return F.interpolate(context, size=low.shape[-2:], mode="bilinear", align_corners=False)


class RegionalCoarseLowFrequencyAffineMixer(CoarseLowFrequencyAffineMixer):
    """Joint local/regional conditioning for the same LL2 gain and bias head."""

    def __init__(self, original: CoarseLowFrequencyAffineMixer):
        nn.Module.__init__(self)
        self.channels = original.channels
        # Reuse all existing V2 tensors. Additional weights are initialized last.
        self.norm = original.norm
        self.context_pw = original.context_pw
        self.context_dw = original.context_dw
        self.affine_head = original.affine_head
        self.high = original.high
        self.regional = RegionalLLContext(self.channels)
        self.context_fusion = nn.Conv2d(4 * self.channels, 2 * self.channels, 1)

    def affine_parameters(self, low: Tensor) -> tuple[Tensor, Tensor]:
        value = low.float()
        variance, mean = torch.var_mean(value, dim=(-2, -1), keepdim=True, unbiased=False)
        deviation = torch.sqrt(variance + 1e-6)
        context = torch.cat((self.norm(value), mean.expand_as(value),
                             deviation.expand_as(value)), dim=1)
        local = F.gelu(self.context_dw(self.context_pw(context)))
        regional = self.regional(value)
        joint = F.gelu(self.context_fusion(torch.cat((local, regional), dim=1)))
        log_gain, bias = self.affine_head(joint).chunk(2, dim=1)
        return torch.exp(torch.tanh(log_gain)), bias


def install_candidate(model: nn.Module, recipe: dict) -> None:
    candidate = recipe["candidate"]
    if candidate == "regional-ll2":
        for site in ALL_OWN_SITES:
            block = model.get_submodule(site)
            block.coarse = RegionalCoarseLowFrequencyAffineMixer(block.coarse)
        return
    if candidate not in {"bottom-stacked", "bottom-interleaved"}:
        raise ValueError(f"Unknown regional/depth candidate: {candidate}")
    channels, heads = model.mdta.channels, model.mdta.heads
    # A newly initialized MDTA, not a second call or copy of the first weights.
    model.mdta_second = WaveletMDTAResidual(MDTAResidual(channels, LayerNorm2d, heads=heads))
    model.experimental_bottom_order = candidate
