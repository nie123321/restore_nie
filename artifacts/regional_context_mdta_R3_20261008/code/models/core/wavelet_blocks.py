"""Haar subband gating and LL-guided channel attention for the original Star-A.

Haar LL is a feature subband, not a ground-truth illumination map. The six
original StarBlocks are preserved. New candidates have no measured quality yet.
"""
from __future__ import annotations

import copy

from frequency_candidate_blocks import CANDIDATE_VARIANTS, candidate_recipe

import torch
import torch.nn.functional as F
from torch import Tensor, nn


WAVELET_VARIANTS = ("none", "encoder", "encoder-mdta", "all", "mdta-only",
                    "encoder-mdta-low-scale", "encoder-context-mdta", "encoder-multiscale-mdta",
                    "all-multiscale-mdta") + CANDIDATE_VARIANTS
NEXT_VARIANTS = ("mdta-only", "encoder-mdta-low-scale", "encoder-context-mdta")
FREQUENCY_VARIANTS = ("encoder-multiscale-mdta", "all-multiscale-mdta") + CANDIDATE_VARIANTS
MDTA_VARIANTS = ("encoder-mdta",) + NEXT_VARIANTS + FREQUENCY_VARIANTS
ENCODER_SITES = ("encoder0", "encoder1", "encoder2")
ALL_SITES = ENCODER_SITES + ("decoder1", "decoder0")
WAVELET_RECIPE = {
    "version": 2,
    "transform": "fixed orthonormal one-level Haar; FP32 DWT/IDWT",
    "subband_order": ["LL", "LH (high along height)", "HL (high along width)", "HH"],
    "boundary": "replicate bottom/right to even dimensions; crop after IDWT",
    "low_processing": "two GatedConvBlocks; first reuses the replaced block's initialization",
    "high_processing": "separate DW3 residual update for each high subband",
    "gate": "processed LL -> groups=1 PW C-to-3C -> 2*sigmoid -> three C-channel spatial maps",
    "gate_initialization": "PW weight/bias zero: each gate starts at one",
    "high_scale": "learnable per-subband per-channel residual scale, initial 0.1",
    "star_blocks": "all six original StarModules and DFFNs retained",
    "supervision": "final-output GT-Mean L1(sigma=0.1) + 0.1 * complex FFT L1 on raw RGB; no LL/high auxiliary target",
    "mdta": "LL Q/K build one channel attention shared by LL/high V; high residual gated by updated LL",
    "mdta_location": "fresh DWT after both bottleneck StarBlocks, replacing the original MDTAResidual",
}


def wavelet_recipe(variant: str) -> dict:
    if variant not in WAVELET_VARIANTS or variant == "none":
        raise ValueError(f"Expected an enabled wavelet variant, got {variant}")
    if variant in CANDIDATE_VARIANTS:
        result = wavelet_recipe("all-multiscale-mdta")
        result.update(candidate_recipe(variant))
        return result
    result = copy.deepcopy(WAVELET_RECIPE)
    result["variant"] = variant
    result["gate_sites"] = list(gate_sites(variant))
    result["replace_mdta"] = variant in MDTA_VARIANTS
    if not result["replace_mdta"]:
        result["mdta"] = "original unchanged MDTAResidual"
        result["mdta_location"] = "after both original bottleneck StarBlocks"
    if variant in NEXT_VARIANTS:
        result["version"] = 3
        result["reference"] = "encoder-mdta (B); shared weights and common new weights retain seed-100 initialization"
        result["encoder_gate"] = "unchanged B pointwise LL gate"
        result["mdta_low_scale"] = "unchanged B: factor one"
        if variant == "mdta-only":
            result["encoder_gate"] = "absent; all five ordinary GatedConvBlocks retained"
            result["low_processing"] = "no encoder subband decomposition; ordinary full-band gates"
            result["high_processing"] = "no encoder subband paths; bottom MDTA still processes four subbands"
            result["intervention"] = "remove the three complete encoder WaveletGatedConvBlocks; retain B subband MDTA"
        elif variant == "encoder-mdta-low-scale":
            result["mdta_low_scale"] = "learnable C-channel scale, initialized 0.1, applied to LL attention residual before high gates"
            result["intervention"] = "B plus only bottom LL residual scaling"
        else:
            result["encoder_gate"] = "U=L_prime+GELU(DW3(L_prime)); gates=2*sigmoid(PW(U)); only encoder0/1/2"
            result["encoder_gate_precision"] = "FP32 context, pointwise projection and sigmoid to retain small context gradients under AMP"
            result["intervention"] = "B plus residual spatial LL context in the three encoder gate generators"
    if variant in FREQUENCY_VARIANTS:
        result.update(
            version=4,
            reference="mdta-only (R1); all shared original and bottom MDTA tensors retain seed-100 initialization",
            transform="fixed two-level orthonormal Haar on encoder input; FP32 DWT/IDWT",
            encoder_gate="preserved ordinary full-resolution spatial gate plus two-level joint frequency residual",
            low_processing="LL1 decomposed again; joint coarse mixing then fine mixing after coarse IDWT",
            high_processing="all three signed high subbands jointly mixed with LL at each level; residual coefficients retained",
            joint_mixer="concat 4C -> PW 4C-to-C -> DW3 C -> GELU -> PW C-to-4C; residual scale initial 0.1 for every band",
            gate="concat spatial output and reconstructed frequency delta -> PW 2C-to-C -> 2*sigmoid",
            gate_initialization="fusion PW weight/bias zero, gate starts at one; output residual scale initial 0.1",
            high_scale="no high-only scale; identical initial residual scale for all four bands",
            precision="new frequency mixers and final fusion gate FP32; spatial path retains original AMP behavior",
            low_bias="LL1 receives an additional coarse processing level; no fixed suppression of high bands",
            mdta_low_scale="unchanged R1/B: factor one",
            intervention="replace encoder0/1/2 with spatial plus two-level joint frequency fusion; R1 bottom MDTA and six StarBlocks unchanged",
        )
        if variant == "all-multiscale-mdta":
            result.update(
                version=5,
                reference="encoder-multiscale-mdta; all common tensors and three encoder fusion initializations preserved; independent new decoder fusion modules appended last",
                transform="fixed two-level orthonormal Haar inside all five gated-block sites; FP32 DWT/IDWT",
                decoder_gate="same spatial plus two-level frequency fusion architecture as encoder, with independent parameters",
                symmetry="encoder0/decoder0 at width, encoder1/decoder1 at 2*width; encoder2 is the central 4*width block",
                intervention="extend encoder-multiscale-mdta to decoder1/0: all five gated blocks use the same multi-scale frequency fusion module; bottom MDTA and six StarBlocks unchanged",
            )
    return result


def gate_sites(variant: str) -> tuple[str, ...]:
    if variant == "all-subband-depth7-mdta":
        return ALL_SITES + ("post_star_encoder1", "post_star_decoder1")
    if variant == "mdta-only":
        return ()
    return ALL_SITES if variant in {"all", "all-multiscale-mdta", *CANDIDATE_VARIANTS} else ENCODER_SITES


def haar_dwt(x: Tensor) -> tuple[Tensor, tuple[Tensor, Tensor, Tensor], tuple[int, int]]:
    """Return LL, (LH, HL, HH), original size. Signed coefficients are retained."""
    if x.ndim != 4 or min(x.shape[-2:]) < 1 or not x.is_floating_point():
        raise ValueError("Haar DWT expects a nonempty floating NCHW tensor")
    height, width = x.shape[-2:]
    with torch.autocast(device_type=x.device.type, enabled=False):
        value = x.float()
        if height % 2 or width % 2:
            value = F.pad(value, (0, width % 2, 0, height % 2), mode="replicate")
        a, b = value[..., 0::2, 0::2], value[..., 0::2, 1::2]
        c, d = value[..., 1::2, 0::2], value[..., 1::2, 1::2]
        ll = (a + b + c + d) * 0.5
        lh = (-a - b + c + d) * 0.5
        hl = (-a + b - c + d) * 0.5
        hh = (a - b - c + d) * 0.5
    return ll, (lh, hl, hh), (height, width)


def haar_idwt(ll: Tensor, high: tuple[Tensor, Tensor, Tensor],
              size: tuple[int, int]) -> Tensor:
    """Reconstruct in FP32. Adding another full-input residual would double it."""
    if len(high) != 3 or ll.ndim != 4 or any(h.shape != ll.shape for h in high):
        raise ValueError("Haar IDWT expects LL and three equally shaped NCHW subbands")
    height, width = size
    if height < 1 or width < 1 or ((height + 1) // 2, (width + 1) // 2) != ll.shape[-2:]:
        raise ValueError("Original size does not match the Haar subband dimensions")
    with torch.autocast(device_type=ll.device.type, enabled=False):
        ll = ll.float()
        lh, hl, hh = (h.float() for h in high)
        a = (ll - lh - hl + hh) * 0.5
        b = (ll - lh + hl - hh) * 0.5
        c = (ll + lh - hl - hh) * 0.5
        d = (ll + lh + hl + hh) * 0.5
        batch, channels, half_h, half_w = ll.shape
        cells = torch.stack((a, b, c, d), dim=-1).reshape(
            batch, channels, half_h, half_w, 2, 2)
        value = cells.permute(0, 1, 2, 4, 3, 5).reshape(
            batch, channels, half_h * 2, half_w * 2)
        return value[..., :height, :width]


class SubbandChannelGates(nn.Module):
    """Channel-specific spatial gates; the generating pointwise projection mixes LL channels."""

    def __init__(self, channels: int):
        super().__init__()
        self.project = nn.Conv2d(channels, 3 * channels, 1, groups=1, bias=True)
        nn.init.zeros_(self.project.weight)
        nn.init.zeros_(self.project.bias)

    def forward(self, low: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        return (2.0 * torch.sigmoid(self.project(low))).chunk(3, dim=1)


class ContextSubbandChannelGates(nn.Module):
    """Retain pointwise LL information and add neighboring context before gating."""

    def __init__(self, original: SubbandChannelGates):
        super().__init__()
        self.project = original.project
        channels = self.project.in_channels
        self.context = nn.Conv2d(channels, channels, 3, padding=1,
                                 groups=channels, bias=False)

    def forward(self, low: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        # With initially tiny PW weights, FP16 PW input gradients underflow
        # before reaching the context DW convolution. Keep this new path FP32.
        with torch.autocast(device_type=low.device.type, enabled=False):
            value = low.float()
            context = value + F.gelu(self.context(value))
            return (2.0 * torch.sigmoid(self.project(context))).chunk(3, dim=1)


class WaveletGatedConvBlock(nn.Module):
    """Two LL blocks and three gated high residuals, reconstructed at the input size."""

    def __init__(self, original: nn.Module):
        super().__init__()
        channels = original.expand.in_channels
        self.channels = channels
        # Same construction order for all variants preserves their common new weights.
        self.low_blocks = nn.Sequential(original, type(original)(channels))
        self.high_updates = nn.ModuleList([
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False)
            for _ in range(3)
        ])
        self.gates = SubbandChannelGates(channels)
        self.high_scale = nn.Parameter(torch.full((3, 1, channels, 1, 1), 0.1))

    def forward(self, x: Tensor) -> Tensor:
        low, high, size = haar_dwt(x)
        updated_low = self.low_blocks(low)
        gates = self.gates(updated_low)
        updated_high = tuple(
            value + self.high_scale[index] * gate * update(value)
            for index, (value, gate, update) in enumerate(zip(high, gates, self.high_updates))
        )
        # Both LL and high retain their residual; IDWT already contains the input.
        return haar_idwt(updated_low, updated_high, size).to(dtype=x.dtype)


class WaveletMDTAResidual(nn.Module):
    """LL-derived channel attention shared by four subband values.

No spatial token attention is introduced. For C=96, heads=4, each attention
matrix is 24x24. A new DWT is evaluated on the current post-Star feature.
"""

    def __init__(self, original: nn.Module, low_scale_init: float | None = None):
        super().__init__()
        self.norm = original.norm
        self.attn = original.attn
        self.channels = self.attn.qkv.in_channels
        self.heads = self.attn.heads
        channels = self.channels
        self.high_values = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(channels, channels, 1, bias=False),
                nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False),
            ) for _ in range(3)
        ])
        self.high_outputs = nn.ModuleList([
            nn.Conv2d(channels, channels, 1, bias=False) for _ in range(3)
        ])
        self.gates = SubbandChannelGates(channels)
        self.high_scale = nn.Parameter(torch.full((3, 1, channels, 1, 1), 0.1))
        if low_scale_init is None:
            self.register_parameter("low_scale", None)
        else:
            self.low_scale = nn.Parameter(torch.full((1, channels, 1, 1), low_scale_init))

    def forward(self, x: Tensor) -> Tensor:
        low, high, size = haar_dwt(x)
        batch, channels, height, width = low.shape
        shape = (batch, self.heads, channels // self.heads, height * width)
        query, key, value = self.attn.qkv_dwconv(self.attn.qkv(self.norm(low))).chunk(3, dim=1)
        # FP32 normalization and softmax also avoid half-precision accumulation overflow.
        with torch.autocast(device_type=x.device.type, enabled=False):
            query = F.normalize(query.float().reshape(shape), dim=-1)
            key = F.normalize(key.float().reshape(shape), dim=-1)
            attention = ((query @ key.transpose(-2, -1)) * self.attn.temperature.float()).softmax(dim=-1)
            low_update = (attention @ value.float().reshape(shape)).reshape(low.shape)
        low_delta = self.attn.project_out(low_update)
        if self.low_scale is not None:
            low_delta = self.low_scale * low_delta
        updated_low = low + low_delta
        gates = self.gates(updated_low)
        updated_high = []
        for index, (detail, gate, value_layer, output_layer) in enumerate(
                zip(high, gates, self.high_values, self.high_outputs)):
            detail_value = value_layer(detail)
            with torch.autocast(device_type=x.device.type, enabled=False):
                mixed = (attention @ detail_value.float().reshape(shape)).reshape(detail.shape)
            updated_high.append(
                detail + self.high_scale[index] * gate * output_layer(mixed))
        return haar_idwt(updated_low, tuple(updated_high), size).to(dtype=x.dtype)


def install_wavelet_blocks(model: nn.Module, variant: str) -> None:
    """Install after the entire original model is initialized. Not repeatable."""
    from model import GatedConvBlock
    from mdta_blocks import MDTAResidual
    from star_blocks import StarBlock

    if variant not in WAVELET_VARIANTS or variant == "none":
        raise ValueError(f"Unsupported enabled wavelet variant: {variant}")
    requested_variant = variant
    if variant in CANDIDATE_VARIANTS:
        variant = "all-multiscale-mdta"
    # Retain B/R1 construction draws before building new multi-scale modules.
    construction_sites = ALL_SITES if variant == "all" else ENCODER_SITES
    sites = gate_sites(variant)
    if not all(type(getattr(model, name)) is GatedConvBlock for name in construction_sites):
        raise ValueError("Expected original ordinary gated blocks; installation is not repeatable")
    if sum(type(module) is StarBlock for module in model.modules()) != 6:
        raise ValueError("Wavelet experiments require six unchanged original StarBlocks")
    if type(model.mdta) is not MDTAResidual:
        raise ValueError("Expected the original bottom MDTAResidual")
    wrappers = {name: WaveletGatedConvBlock(getattr(model, name))
                for name in construction_sites}
    if variant not in FREQUENCY_VARIANTS:
        for name in sites:
            setattr(model, name, wrappers[name])
    if variant in MDTA_VARIANTS:
        model.mdta = WaveletMDTAResidual(
            model.mdta, low_scale_init=0.1 if variant == "encoder-mdta-low-scale" else None)
    # Install context last so every parameter inherited from B keeps its seed.
    if variant == "encoder-context-mdta":
        for name in ENCODER_SITES:
            block = getattr(model, name)
            block.gates = ContextSubbandChannelGates(block.gates)
    if variant in FREQUENCY_VARIANTS:
        from multiscale_frequency_blocks import MultiscaleFrequencyFusionBlock
        # Encoder draws match the first frequency candidate. Decoder modules
        # are appended only afterwards, leaving the shared bottom MDTA intact.
        for name in sites:
            setattr(model, name, MultiscaleFrequencyFusionBlock(getattr(model, name)))

    if requested_variant in CANDIDATE_VARIANTS:
        from frequency_candidate_blocks import install_frequency_candidate
        install_frequency_candidate(model, requested_variant)
