"""Four independent interventions on all-multiscale-mdta, installed last.

These are research candidates, not trained models. Common baseline tensors are
reused so adding a candidate does not perturb their seeded initialization.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from multiscale_frequency_blocks import JointSubbandMixer, MultiscaleFrequencyFusionBlock
from star_blocks import DFFN, StarBlock, window_fft_filter


from structure_candidate_blocks import STRUCTURE_VARIANTS, structure_recipe, install_structure_candidate

CANDIDATE_VARIANTS = ("all-coarse-amp-mdta", "all-shared-dffn-mdta",
                      "all-subband-fusion-mdta", "all-subband-first-mdta") + STRUCTURE_VARIANTS
FUSION_SITES = ("encoder0", "encoder1", "encoder2", "decoder1", "decoder0")


def candidate_recipe(variant: str) -> dict:
    """Serialized in training config and checked on resume."""
    if variant in STRUCTURE_VARIANTS:
        result = candidate_recipe("all-subband-first-mdta")
        structure = structure_recipe(variant)
        result.update(version=8, variant=variant, reference="all-subband-first-mdta",
                      intervention=structure["intervention"], initialization=structure["initialization"],
                      structure_recipe=structure, mdta_location=structure["bottom_order"])
        if variant == STRUCTURE_VARIANTS[0]:
            result.update(joint_mixer=structure["mixer"]+"; K=C/2", replacement_scope=structure["replacement_scope"])
        if variant == STRUCTURE_VARIANTS[1]:
            result["symmetry"] = structure["symmetry"]
        if variant == STRUCTURE_VARIANTS[3]:
            result["gate_sites"] += ["post_star_encoder1", "post_star_decoder1"]
            result["replacement_scope"] = "all original ten D mixers preserved; add two independent D fusion modules"
        return result
    if variant not in CANDIDATE_VARIANTS:
        raise ValueError(f"Unsupported frequency candidate: {variant}")
    result = dict(
        version=6, variant=variant, reference="all-multiscale-mdta",
        initialization="construct the complete five-site baseline first; install only the selected intervention last",
        gate_sites=list(FUSION_SITES),
        symmetry="same module design at 24/48/96/48/24 sites; independent weights across sites",
        supervision="GT-Mean L1(sigma=0.1) + 0.1 * complex FFT L1 on raw RGB; no added RGB L1, LF8 or subband target",
        quality_status="untrained candidate; implementation checks are not quality evidence",
    )
    if variant == "all-coarse-amp-mdta":
        result.update(
            intervention="replace only the coarse mixer at each of five sites with local plus conditional Fourier-amplitude processing",
            joint_mixer="coarse: Z=concat(LL2,LH2,HL2,HH2), T=PW(4C,C)(Z), U=DW3(T)+eta*(irfft2(rfft2(T)*M)-T); update=PW(C,4C)(GELU(U)); Z'=Z+beta*update; fine mixer unchanged",
            amplitude="rfft2/irfft2, norm=ortho, FP32; a=log1p(abs(Zfft)/(mean_frequency(abs(Zfft))+1e-6)); depthwise PW -> LeakyReLU(0.1) -> depthwise PW",
            condition="concat(spatial_mean(T)/RMS(T), log1p(RMS(T))) -> dense PW(2C,C); RMS=sqrt(mean(T^2)+1e-6)",
            gain="M=1+0.5*tanh(amplitude(a)+condition); positive range (0.5,1.5), preserves phase only at this spectral multiplication",
            amplitude_initialization="last amplitude PW and condition PW weights normal(std=0.01), bias zero; eta is per-channel, initial 0.1",
            precision="all new coarse paths FP32; baseline fine mixer and output gate remain FP32",
            relation_to_uhdres="inspired by amplitude modulation; normalized spectrum, explicit mean/RMS condition and bounded positive gain are this candidate's design",
        )
    elif variant == "all-shared-dffn-mdta":
        result.update(
            intervention="replace the middle spatial processor in all six DFFNs; all five frequency blocks unchanged",
            star_blocks="six original StarModules, outer norms/residuals and DFFN projections/terminal 8x8 FFT retained; DFFN spatial processor replaced",
            dffn="PW(C,6C) -> split A,B of 3C -> shared S(A),S(B) -> GELU(S(A))*S(B) -> PW(3C,C) -> original 8x8 window FFT",
            shared_processor="S(U)=DW3x3(U)+lambda*DW7x1(DW1x7(U)); identical weights used on A and B; no convolution bias",
            dffn_initialization="shared DW3 copies the first half of the original DFFN DW3 weights; second half removed; strip convolutions freshly initialized; lambda per hidden channel, initial 0.1",
            precision="spatial convolutions follow baseline AMP; original terminal FFT remains FP32",
            relation_to_uhdres="borrows shared strip-convolution context; not a verbatim SGFN or a reproduction of UHDRes",
        )
    elif variant == "all-subband-fusion-mdta":
        result.update(
            intervention="replace the output pixel/channel gate in all five fusion blocks with four conditional subband-delta gates before IDWT",
            gate="D=concat(LL1_updated-LL1_original,H1_updated-H1_original), retaining accumulated coarse+fine LL updates; condition=concat(DWT(S),D) with 8C channels; PW(8C,C/4) -> GELU -> PW(C/4,4C) -> 2*sigmoid",
            gate_initialization="last PW weight/bias zero, all four gates start at one; old PW(2C,C) output gate removed; gamma retained at 0.1",
            reconstruction="Y=S+gamma*IDWT(g_LL*D_LL,g_LH*D_LH,g_HL*D_HL,g_HH*D_HH); gate only deltas, never the original coefficients",
            precision="conditional gate, DWT/IDWT and frequency mixers FP32; output retains spatial branch dtype",
            startup="matches baseline gate-one mapping up to Haar floating-point roundoff; first gate PW has zero task gradient until final PW leaves zero",
        )
    else:
        result.update(
            version=7,
            intervention="replace coarse and fine mixers at all five sites: independent depthwise spatial processing of each signed subband, then cross-subband pointwise mixing",
            joint_mixer="Z=concat(LL,LH,HL,HH); U=DW3(4C,groups=4C)(Z); delta=PW(C,4C)(GELU(PW(4C,C)(U))); Z_new=Z+beta*delta",
            spatial_order="subband/channel-specific DW3 kernels before the first cross-subband PW; no cross-channel or cross-subband mixing inside DW3",
            replacement_scope="ten mixers: coarse and fine at encoder0/1/2 and decoder1/0; all other fusion components, six original StarBlocks and bottom MDTA unchanged",
            spatial_initialization="new 4C-channel DW3 uses standard Conv2d initialization; previous C-channel latent DW3 removed; existing reduce/expand PW and beta copied without change",
            precision="new subband-first mixers FP32; original fusion gate and Haar FP32; other AMP behavior unchanged",
            residual="all original signed subband coefficients retained via Z+beta*delta; no additional band gates or auxiliary losses",
        )
    return result


class SubbandFirstMixer(nn.Module):
    """Learn each signed subband's spatial filter before cross-subband mixing."""

    def __init__(self, original: JointSubbandMixer):
        super().__init__()
        self.channels = original.channels
        self.reduce = original.reduce
        self.expand = original.expand
        self.scale = original.scale
        self.spatial = nn.Conv2d(4 * self.channels, 4 * self.channels, 3,
                                 padding=1, groups=4 * self.channels, bias=True)

    def forward(self, low: Tensor, high: tuple[Tensor, Tensor, Tensor]
                ) -> tuple[Tensor, tuple[Tensor, Tensor, Tensor]]:
        if len(high) != 3 or low.ndim != 4 or low.shape[1] != self.channels:
            raise ValueError("Expected LL and three C-channel NCHW subbands")
        if any(value.shape != low.shape for value in high):
            raise ValueError("Subband-first mixing requires equally shaped subbands")
        with torch.autocast(device_type=low.device.type, enabled=False):
            bands = torch.cat((low.float(), *(value.float() for value in high)), dim=1)
            # groups=4C gives independent filters to every channel in every
            # band. The following reduce PW is the first cross-band operation.
            per_band = self.spatial(bands)
            update = self.expand(F.gelu(self.reduce(per_band)))
            ll, lh, hl, hh = (bands + self.scale * update).chunk(4, dim=1)
        return ll, (lh, hl, hh)


class ConditionalAmplitudeResidual(nn.Module):
    """Real positive, image-conditioned gain on a whole coarse latent spectrum."""

    def __init__(self, channels: int):
        super().__init__()
        self.amplitude = nn.Sequential(
            nn.Conv2d(channels, channels, 1, groups=channels, bias=True),
            nn.LeakyReLU(0.1),
            nn.Conv2d(channels, channels, 1, groups=channels, bias=True),
        )
        self.condition = nn.Conv2d(2 * channels, channels, 1, bias=True)
        self.scale = nn.Parameter(torch.full((1, channels, 1, 1), 0.1))
        for head in (self.amplitude[-1], self.condition):
            nn.init.normal_(head.weight, std=0.01)
            nn.init.zeros_(head.bias)

    def modulated_spectrum(self, x: Tensor) -> tuple[Tensor, Tensor]:
        with torch.autocast(device_type=x.device.type, enabled=False):
            value = x.float()
            spectrum = torch.fft.rfft2(value, dim=(-2, -1), norm="ortho")
            magnitude = spectrum.abs()
            amplitude = torch.log1p(magnitude / (magnitude.mean((-2, -1), keepdim=True) + 1e-6))
            mean = value.mean((-2, -1), keepdim=True)
            rms = torch.sqrt(value.square().mean((-2, -1), keepdim=True) + 1e-6)
            condition = self.condition(torch.cat((mean / rms, torch.log1p(rms)), dim=1))
            gain = 1.0 + 0.5 * torch.tanh(self.amplitude(amplitude) + condition)
            # The mapping is pointwise in frequency with a real shared channel
            # condition. This preserves Hermitian symmetry on rfft edge columns.
            return spectrum * gain, gain

    def forward(self, x: Tensor) -> Tensor:
        with torch.autocast(device_type=x.device.type, enabled=False):
            spectrum, _ = self.modulated_spectrum(x)
            restored = torch.fft.irfft2(spectrum, s=x.shape[-2:], dim=(-2, -1), norm="ortho")
            return self.scale * (restored - x.float())


class CoarseAmplitudeMixer(nn.Module):
    """Reuse the complete coarse mixer and add a parallel amplitude residual."""

    def __init__(self, original: JointSubbandMixer):
        super().__init__()
        self.channels = original.channels
        self.reduce = original.reduce
        self.spatial = original.spatial
        self.expand = original.expand
        self.scale = original.scale
        self.spectral = ConditionalAmplitudeResidual(self.channels)

    def forward(self, low: Tensor, high: tuple[Tensor, Tensor, Tensor]
                ) -> tuple[Tensor, tuple[Tensor, Tensor, Tensor]]:
        if low.ndim != 4 or low.shape[1] != self.channels or len(high) != 3:
            raise ValueError("Expected LL and three C-channel NCHW subbands")
        if any(value.shape != low.shape for value in high):
            raise ValueError("Expected equally shaped subbands")
        with torch.autocast(device_type=low.device.type, enabled=False):
            bands = torch.cat((low.float(), *(value.float() for value in high)), dim=1)
            latent = self.reduce(bands)
            mixed = self.spatial(latent) + self.spectral(latent)
            ll, lh, hl, hh = (bands + self.scale * self.expand(F.gelu(mixed))).chunk(4, dim=1)
        return ll, (lh, hl, hh)


class SharedLocalStripProcessor(nn.Module):
    """One set of local/strip weights is applied to both FFN feature halves."""

    def __init__(self, original: DFFN):
        super().__init__()
        hidden = original.project_out.in_channels
        self.local = nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden, bias=False)
        self.horizontal = nn.Conv2d(hidden, hidden, (1, 7), padding=(0, 3), groups=hidden, bias=False)
        self.vertical = nn.Conv2d(hidden, hidden, (7, 1), padding=(3, 0), groups=hidden, bias=False)
        self.scale = nn.Parameter(torch.full((1, hidden, 1, 1), 0.1))
        with torch.no_grad():
            self.local.weight.copy_(original.dwconv.weight[:hidden])

    def forward(self, x: Tensor) -> Tensor:
        return self.local(x) + self.scale * self.vertical(self.horizontal(x))


class SharedSpatialDFFN(nn.Module):
    """Original pointwise/gating/window-FFT chain with a shared spatial center."""

    def __init__(self, original: DFFN):
        super().__init__()
        self.dim = original.dim
        self.patch_size = original.patch_size
        self.project_in = original.project_in
        self.project_out = original.project_out
        self.fft = original.fft
        self.processor = SharedLocalStripProcessor(original)

    def forward(self, x: Tensor) -> Tensor:
        first, second = self.project_in(x).chunk(2, dim=1)
        first = self.processor(first)
        second = self.processor(second)
        return window_fft_filter(self.project_out(F.gelu(first) * second), self.fft)


class SubbandConditionalFusion(nn.Module):
    """Gate cumulative subband deltas before reconstruction, conditioned on S."""

    def __init__(self, original: MultiscaleFrequencyFusionBlock):
        super().__init__()
        self.channels = original.channels
        if self.channels % 4:
            raise ValueError("Subband condition bottleneck requires channels divisible by four")
        self.spatial = original.spatial
        self.coarse = original.coarse
        self.fine = original.fine
        self.scale = original.scale
        self.condition = nn.Sequential(
            nn.Conv2d(8 * self.channels, self.channels // 4, 1, bias=True),
            nn.GELU(),
            nn.Conv2d(self.channels // 4, 4 * self.channels, 1, bias=True),
        )
        nn.init.zeros_(self.condition[-1].weight)
        nn.init.zeros_(self.condition[-1].bias)

    def subband_deltas(self, x: Tensor) -> tuple[Tensor, tuple[int, int]]:
        from wavelet_blocks import haar_dwt, haar_idwt

        low1, high1, size1 = haar_dwt(x)
        low2, high2, size2 = haar_dwt(low1)
        updated_low2, updated_high2 = self.coarse(low2, high2)
        coarse_low1 = haar_idwt(updated_low2, updated_high2, size2)
        updated_low1, updated_high1 = self.fine(coarse_low1, high1)
        # Compare LL to the original LL1: comparing it to coarse_low1 would
        # erase the entire coarse update when constructing the final delta.
        delta = torch.cat((updated_low1 - low1,
                           *(new - old for new, old in zip(updated_high1, high1))), dim=1)
        return delta, size1

    def forward(self, x: Tensor) -> Tensor:
        from wavelet_blocks import haar_dwt, haar_idwt

        spatial = self.spatial(x)
        with torch.autocast(device_type=x.device.type, enabled=False):
            delta, size = self.subband_deltas(x)
            spatial_low, spatial_high, _ = haar_dwt(spatial)
            condition = torch.cat((spatial_low, *spatial_high, delta), dim=1)
            gates = 2.0 * torch.sigmoid(self.condition(condition))
            ll, lh, hl, hh = (gates * delta).chunk(4, dim=1)
            frequency_delta = haar_idwt(ll, (lh, hl, hh), size)
            output = spatial.float() + self.scale * frequency_delta
        return output.to(dtype=spatial.dtype)


def install_frequency_candidate(model: nn.Module, variant: str) -> None:
    """Install only after all-multiscale-mdta is fully built; no weight loading."""
    if variant in STRUCTURE_VARIANTS:
        install_frequency_candidate(model, "all-subband-first-mdta")
        install_structure_candidate(model, variant)
        return
    if variant not in CANDIDATE_VARIANTS:
        raise ValueError(f"Unsupported candidate: {variant}")
    if not all(type(getattr(model, site)) is MultiscaleFrequencyFusionBlock for site in FUSION_SITES):
        raise ValueError("Expected the complete five-site frequency baseline")
    stars = [module for module in model.modules() if type(module) is StarBlock]
    if len(stars) != 6 or any(type(module.ffn) is not DFFN for module in stars):
        raise ValueError("Expected six original StarBlocks/DFFNs")
    if variant == "all-coarse-amp-mdta":
        for site in FUSION_SITES:
            block = getattr(model, site)
            if type(block.coarse) is not JointSubbandMixer:
                raise ValueError("Coarse candidate has already been installed")
        for site in FUSION_SITES:
            block = getattr(model, site)
            block.coarse = CoarseAmplitudeMixer(block.coarse)
    elif variant == "all-shared-dffn-mdta":
        for block in stars:
            block.ffn = SharedSpatialDFFN(block.ffn)
    elif variant == "all-subband-fusion-mdta":
        for site in FUSION_SITES:
            setattr(model, site, SubbandConditionalFusion(getattr(model, site)))
    else:
        for site in FUSION_SITES:
            block = getattr(model, site)
            block.coarse = SubbandFirstMixer(block.coarse)
            block.fine = SubbandFirstMixer(block.fine)
