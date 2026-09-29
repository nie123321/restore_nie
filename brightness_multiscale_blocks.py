"""Four extra guidance sites sharing the existing bottom prior pyramid."""
from torch import nn

from brightness_cross_blocks import BrightnessCrossAttentionFFN


class MultiScaleBrightnessGuidance(nn.Module):
    """Full/half priors reused by encoder and decoder; independent site attention."""

    def __init__(self, width, norm_layer):
        super().__init__()
        self.full_projection = nn.Sequential(
            nn.Conv2d(4, 16, 3, padding=1), nn.GELU(), nn.Conv2d(16, width, 1),
        )
        self.half_projection = nn.Conv2d(16, 2 * width, 1)
        self.encoder_full = BrightnessCrossAttentionFFN(width, norm_layer, heads=1)
        self.encoder_half = BrightnessCrossAttentionFFN(2 * width, norm_layer, heads=2)
        self.decoder_half = BrightnessCrossAttentionFFN(2 * width, norm_layer, heads=2)
        self.decoder_full = BrightnessCrossAttentionFFN(width, norm_layer, heads=1)

    def project_pyramid(self, pyramid):
        return {"full": self.full_projection(pyramid["maps"]),
                "half": self.half_projection(pyramid["half"]),
                "bottom": pyramid["bottom"]}
