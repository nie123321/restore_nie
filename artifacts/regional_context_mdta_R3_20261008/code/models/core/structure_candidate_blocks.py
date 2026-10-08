"""Four controlled structure candidates built on fixed D, installed last."""
from __future__ import annotations

from torch import nn

STRUCTURE_VARIANTS = (
    "all-subband-half-mdta", "all-subband-width16-mdta",
    "all-subband-interleave-mdta", "all-subband-depth7-mdta",
)
STRUCTURE_WIDTHS = dict(zip(STRUCTURE_VARIANTS, (24, 16, 24, 24)))
STRUCTURE_COUNTS = dict(zip(STRUCTURE_VARIANTS, (844999, 450215, 965047, 1093015)))
STRUCTURE_LAUNCHERS = dict(zip(STRUCTURE_VARIANTS, (
    "train_mixer_half.py", "train_width16.py", "train_mdta_interleave.py", "train_fusion_depth.py")))


def structure_recipe(variant: str) -> dict:
    if variant not in STRUCTURE_VARIANTS:
        raise ValueError(f"Unsupported structure candidate: {variant}")
    width = STRUCTURE_WIDTHS[variant]
    recipe = dict(
        version=1, variant=variant, reference="all-subband-first-mdta (fixed D)",
        model_width=width, channels=[width, 2*width, 4*width],
        parameters=STRUCTURE_COUNTS[variant], star_blocks=6, fusion_blocks=5,
        mdta_blocks=1, star_variant="all six original StarModules and DFFNs including FFT8",
        bottom_order="frequency encoder2 -> StarBlock1 -> StarBlock2 -> WaveletMDTAResidual",
        mixer="Z=concat(LL,LH,HL,HH); U=DW3(4C,groups=4C)(Z); Z_new=Z+beta*PW(K,4C)(GELU(PW(4C,K)(U)))",
        mixer_latent="K=C", train_protocol="unchanged D protocol, architecture only",
        initialization="construct complete D first, then apply this intervention; common tensors retain the same seed-100 initialization",
        quality_status="untrained; synthetic checks provide implementation evidence only",
    )
    if variant == STRUCTURE_VARIANTS[0]:
        recipe.update(
            intervention="compress only the pointwise latent width in all ten coarse/fine mixers from C to C/2",
            mixer_latent="K=C/2",
            replacement_scope="reduce/expand PW in ten mixers; DW3(4C), subband residual scales, spatial gates, output fusion, StarBlocks and MDTA unchanged",
            initialization="construct D first; fresh standard Conv2d initialization for only the twenty narrower pointwise layers; all other tensors reused exactly",
        )
    elif variant == STRUCTURE_VARIANTS[1]:
        recipe.update(
            intervention="change main width 24/48/96 to 16/32/64; keep D topology and relative mixer width K=C",
            symmetry="encoder0/decoder0:16; encoder1/decoder1:32; central encoder2:64; independent weights",
            initialization="construct complete D at width16 with seed100; no extra module draws or replacements beyond D",
        )
    elif variant == STRUCTURE_VARIANTS[2]:
        recipe.update(
            intervention="move the existing bottom subband MDTA between the two original bottom StarBlocks",
            bottom_order="frequency encoder2 -> StarBlock1 -> WaveletMDTAResidual -> StarBlock2",
            initialization="identical D parameter names and seed-100 tensors; execution order only; no added or removed parameter",
        )
    else:
        recipe.update(
            intervention="add two independent D multiscale fusion blocks after encoder1/decoder1 original StarBlocks",
            fusion_blocks=7, added_channels=[48,48], added_parameters=127968,
            encoder1_order="original frequency encoder1 -> original StarBlock -> new D fusion48 -> down2",
            decoder1_order="fuse1 -> original frequency decoder1 -> original StarBlock -> new D fusion48 -> fuse0",
            added_initialization="fresh spatial GatedConvBlock, coarse/fine SubbandFirstMixer and fusion output gate/scales; no weight sharing; output gate zero starts at one, residual scale0.1",
        )
    return recipe


def install_structure_candidate(model: nn.Module, variant: str) -> None:
    """Called after installing all ten D mixers. No checkpoint or data read."""
    from frequency_candidate_blocks import FUSION_SITES, SubbandFirstMixer
    from multiscale_frequency_blocks import MultiscaleFrequencyFusionBlock
    from star_blocks import StarBlock
    from wavelet_blocks import WaveletMDTAResidual

    if variant not in STRUCTURE_VARIANTS:
        raise ValueError(variant)
    if model.config["width"] != STRUCTURE_WIDTHS[variant]:
        raise ValueError(f"{variant} requires main width {STRUCTURE_WIDTHS[variant]}")
    if sum(type(m) is StarBlock for m in model.modules()) != 6:
        raise ValueError("Expected the six original D StarBlocks")
    if type(model.mdta) is not WaveletMDTAResidual:
        raise ValueError("Expected the original D subband MDTA")
    for site in FUSION_SITES:
        block = getattr(model, site)
        if type(block) is not MultiscaleFrequencyFusionBlock or any(
            type(mixer) is not SubbandFirstMixer for mixer in (block.coarse, block.fine)
        ):
            raise ValueError("Expected all five complete D fusion blocks")

    if variant == STRUCTURE_VARIANTS[0]:
        for site in FUSION_SITES:
            block = getattr(model, site)
            for mixer in (block.coarse, block.fine):
                channels = mixer.channels
                mixer.reduce = nn.Conv2d(4*channels, channels//2, 1, bias=True)
                mixer.expand = nn.Conv2d(channels//2, 4*channels, 1, bias=True)
    elif variant == STRUCTURE_VARIANTS[3]:
        from model import GatedConvBlock
        # Appended after all D parameters: common initial weights stay exact.
        for name in ("post_star_encoder1", "post_star_decoder1"):
            if hasattr(model, name):
                raise ValueError("Structure intervention already installed")
            block = MultiscaleFrequencyFusionBlock(GatedConvBlock(48))
            block.coarse = SubbandFirstMixer(block.coarse)
            block.fine = SubbandFirstMixer(block.fine)
            setattr(model, name, block)
    # Width16 uses D unchanged at that width. Interleave changes forward order
    # in model.py, preserving all original module names and checkpoint keys.
