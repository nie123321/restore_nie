"""Three joint spatial/wavelet updates, retaining A's LL/H processors and FFN."""
from __future__ import annotations

import torch
from torch import Tensor, nn

from models.baseline_a_blocks import ALL_OWN_SITES
from wavelet_blocks import haar_dwt, haar_idwt

FUSION_MODES = ("wavelet-first", "spatial-first", "amplitude-gate")
EXECUTION_SITES = (
    "encoder0", "star_encoder0", "encoder1", "star_encoder1", "encoder2",
    "spectral.block.0", "spectral.block.1", "mdta", "decoder1",
    "star_decoder1", "decoder0", "star_decoder0",
)


class JointWaveletFusionBlock(nn.Module):
    """One joint residual followed by A's original pointwise gated FFN.

    Build the complete A first and reuse its modules. No new random draws,
    independent frequency-only scale, difference normalization, or DFFN.
    Spatial proposals are unscaled inside the joint transform; scale1 applies
    once to the entire resulting update, in every mode.
    """

    def __init__(self, original: nn.Module, mode: str):
        super().__init__()
        if mode not in FUSION_MODES:
            raise ValueError(f"Unknown joint fusion mode: {mode}")
        self.mode = mode
        self.channels = original.channels
        self.spatial = original.spatial
        self.coarse = original.coarse
        self.fine = original.fine

    def wavelet_transform(self, x: Tensor) -> Tensor:
        """Reconstruct the full feature, retaining the magnitude of its update."""
        low1, high1, size1 = haar_dwt(x)
        low2, high2, size2 = haar_dwt(low1)
        low2, high2 = self.coarse(low2, high2)
        low1 = haar_idwt(low2, high2, size2)
        low1, high1 = self.fine(low1, high1)
        return haar_idwt(low1, high1, size1)

    def spatial_update(self, x: Tensor) -> Tensor:
        local = self.spatial
        a, b = local.depthwise(local.expand(local.norm1(x))).chunk(2, dim=1)
        mixed = a * b
        return local.project(mixed * local.channel_scale(mixed))

    @staticmethod
    def amplitude_gate(source: Tensor, delta: Tensor) -> Tensor:
        """Normalize by input RMS, never by the wavelet difference's own RMS."""
        reference = torch.sqrt(source.square().mean(dim=1, keepdim=True) + 1e-6)
        return 1.0 + torch.tanh(delta / reference)

    def joint_update(self, source: Tensor) -> Tensor:
        if self.mode == "wavelet-first":
            frequency = self.wavelet_transform(source)
            return (frequency - source) + self.spatial_update(frequency)
        if self.mode == "spatial-first":
            spatial_proposal = source + self.spatial_update(source)
            return self.wavelet_transform(spatial_proposal) - source
        delta = self.wavelet_transform(source) - source
        return delta + self.spatial_update(source) * self.amplitude_gate(source, delta)

    def forward(self, x: Tensor) -> Tensor:
        with torch.autocast(device_type=x.device.type, enabled=False):
            source = x.float()
            local = self.spatial
            updated = source + local.scale1 * self.joint_update(source)
            a, b = local.ffn_in(local.norm2(updated)).chunk(2, dim=1)
            output = updated + local.scale2 * local.ffn_out(a * b)
        return output.to(dtype=x.dtype)


def install_joint_fusion(model: nn.Module, mode: str) -> None:
    """Replace A's exteriors at all eleven independent sites; preserve MDTA."""
    if mode not in FUSION_MODES:
        raise ValueError(mode)
    for site in ALL_OWN_SITES:
        parent_name, _, leaf = site.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        original = getattr(parent, leaf)
        setattr(parent, leaf, JointWaveletFusionBlock(original, mode))
