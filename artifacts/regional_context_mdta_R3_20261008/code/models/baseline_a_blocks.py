"""Eleven old-D style fusion blocks with dedicated coarse LL restoration."""
from __future__ import annotations

from torch import Tensor, nn

from model import GatedConvBlock
from multiscale_frequency_blocks import MultiscaleFrequencyFusionBlock
from models.low_frequency_blocks import CoarseLowFrequencyAffineMixer, FineDetailMixer

FUSION_SITES = ("encoder0", "encoder1", "encoder2", "decoder1", "decoder0")
SITE_CHANNELS = (24, 48, 96, 48, 24)
FORMER_STAR_SITES = ("star_encoder0", "star_encoder1", "spectral.block.0",
                     "spectral.block.1", "star_decoder1", "star_decoder0")
ALL_OWN_SITES = FUSION_SITES + FORMER_STAR_SITES


class LowFrequencyFusionBlock(MultiscaleFrequencyFusionBlock):
    """Keep old D's spatial gate and fusion exterior; replace its band mixers."""

    def __init__(self, channels: int, original: nn.Module | None = None):
        if original is None:
            super().__init__(GatedConvBlock(channels))
        else:
            nn.Module.__init__(self)
            if original.channels != channels:
                raise ValueError("Mismatched old-D block channels")
            self.channels = channels
            self.spatial = original.spatial
            self.fusion = original.fusion
            self.scale = original.scale
        self.coarse = CoarseLowFrequencyAffineMixer(channels)
        self.fine = FineDetailMixer(channels)


class OwnBottleneckFusion(nn.Module):
    """Forward adapter for two independent own blocks, without a StarBlock."""

    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(LowFrequencyFusionBlock(channels),
                                   LowFrequencyFusionBlock(channels))

    def forward(self, x: Tensor, condition: Tensor):
        return self.block(x), None


def install_eleven_own_blocks(model: nn.Module) -> None:
    # Full old D is initialized first. Reuse its five spatial/fusion exteriors.
    for site, channels in zip(FUSION_SITES, SITE_CHANNELS):
        setattr(model, site, LowFrequencyFusionBlock(channels, getattr(model, site)))
    model.star_encoder0 = LowFrequencyFusionBlock(24)
    model.star_encoder1 = LowFrequencyFusionBlock(48)
    model.spectral = OwnBottleneckFusion(96)
    model.star_decoder1 = LowFrequencyFusionBlock(48)
    model.star_decoder0 = LowFrequencyFusionBlock(24)
