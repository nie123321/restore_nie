"""Two-level joint Haar fusion with a preserved full-resolution spatial branch.

The LL path visits both levels; all four subbands are learned and mixed at each
level. This is a structural low-frequency bias, not a low-only correction.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class JointSubbandMixer(nn.Module):
    """Mix four signed subbands in a shared C-channel latent space, in FP32."""

    def __init__(self, channels: int):
        super().__init__()
        self.channels = channels
        self.reduce = nn.Conv2d(4 * channels, channels, 1, bias=True)
        self.spatial = nn.Conv2d(channels, channels, 3, padding=1,
                                 groups=channels, bias=True)
        self.expand = nn.Conv2d(channels, 4 * channels, 1, bias=True)
        self.scale = nn.Parameter(torch.full((1, 4 * channels, 1, 1), 0.1))

    def forward(self, low: Tensor, high: tuple[Tensor, Tensor, Tensor]
                ) -> tuple[Tensor, tuple[Tensor, Tensor, Tensor]]:
        if len(high) != 3 or low.ndim != 4 or low.shape[1] != self.channels:
            raise ValueError("Expected LL and three C-channel NCHW subbands")
        if any(value.shape != low.shape for value in high):
            raise ValueError("Joint fusion requires equally shaped subbands")
        with torch.autocast(device_type=low.device.type, enabled=False):
            bands = torch.cat((low.float(), *(value.float() for value in high)), dim=1)
            update = self.expand(F.gelu(self.spatial(self.reduce(bands))))
            ll, lh, hl, hh = (bands + self.scale * update).chunk(4, dim=1)
        return ll, (lh, hl, hh)


class MultiscaleFrequencyFusionBlock(nn.Module):
    """Ordinary spatial gate plus an input-conditioned two-level frequency residual."""

    def __init__(self, original: nn.Module):
        super().__init__()
        self.channels = original.expand.in_channels
        self.spatial = original
        self.coarse = JointSubbandMixer(self.channels)
        self.fine = JointSubbandMixer(self.channels)
        self.fusion = nn.Conv2d(2 * self.channels, self.channels, 1, bias=True)
        nn.init.zeros_(self.fusion.weight)
        nn.init.zeros_(self.fusion.bias)
        self.scale = nn.Parameter(torch.full((1, self.channels, 1, 1), 0.1))

    def frequency_residual(self, x: Tensor) -> Tensor:
        # Local import keeps the installer and Haar definitions acyclic.
        from wavelet_blocks import haar_dwt, haar_idwt

        low1, high1, size1 = haar_dwt(x)
        low2, high2, size2 = haar_dwt(low1)
        updated_low2, updated_high2 = self.coarse(low2, high2)
        refined_low1 = haar_idwt(updated_low2, updated_high2, size2)
        updated_low1, updated_high1 = self.fine(refined_low1, high1)
        frequency = haar_idwt(updated_low1, updated_high1, size1)
        # IDWT includes the input; subtract once before adding to spatial output.
        return frequency - x.float()

    def forward(self, x: Tensor) -> Tensor:
        spatial = self.spatial(x)
        delta = self.frequency_residual(x)
        with torch.autocast(device_type=x.device.type, enabled=False):
            gate = 2.0 * torch.sigmoid(self.fusion(torch.cat((spatial.float(), delta), dim=1)))
            output = spatial.float() + self.scale * gate * delta
        # Preserve the original spatial gate's output precision under AMP.
        return output.to(dtype=spatial.dtype)
