"""Small research prototype: global colour, local gain, and residual restoration.

Ideas, not pretrained code/weights: NAFNet gated convolutions; Retinexformer
illumination guidance; StarIR patch-wise spectral filtering; CSEC colour correction.
The conditional spectral bank and output parameterization below are prototype
choices, not a claim of novelty or a physical illumination/reflectance decomposition.
"""

import math

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class LayerNorm2d(nn.Module):
    """Normalize channels independently at each spatial location."""

    def __init__(self, channels: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x: Tensor) -> Tensor:
        variance, mean = torch.var_mean(x, dim=1, keepdim=True, unbiased=False)
        return (x - mean) * torch.rsqrt(variance + 1e-6) * self.weight + self.bias


class GatedConvBlock(nn.Module):
    """NAFNet-inspired local processing, with two multiplicative gates."""

    def __init__(self, channels: int):
        super().__init__()
        self.norm1 = LayerNorm2d(channels)
        self.expand = nn.Conv2d(channels, 2 * channels, 1)
        self.depthwise = nn.Conv2d(2 * channels, 2 * channels, 3,
                                   padding=1, groups=2 * channels)
        self.channel_scale = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                           nn.Conv2d(channels, channels, 1))
        self.project = nn.Conv2d(channels, channels, 1)
        self.norm2 = LayerNorm2d(channels)
        self.ffn_in = nn.Conv2d(channels, 2 * channels, 1)
        self.ffn_out = nn.Conv2d(channels, channels, 1)
        self.scale1 = nn.Parameter(torch.full((1, channels, 1, 1), 0.1))
        self.scale2 = nn.Parameter(torch.full((1, channels, 1, 1), 0.1))

    def forward(self, x: Tensor) -> Tensor:
        a, b = self.depthwise(self.expand(self.norm1(x))).chunk(2, dim=1)
        feature = a * b
        x = x + self.scale1 * self.project(feature * self.channel_scale(feature))
        a, b = self.ffn_in(self.norm2(x)).chunk(2, dim=1)
        return x + self.scale2 * self.ffn_out(a * b)


class ConditionalSpectralBlock(nn.Module):
    """One spatial/frequency fusion block at the U-Net bottleneck.

    `off` retains the same spatial fusion but bypasses Fourier filtering.
    `static` uses one frequency response shared by all inputs.
    `conditional` adds a per-image mixture of learned frequency-response bases.
    Filter responses are real and positive, bounded to (0.5, 1.5).
    This is a modification inspired by StarIR, not its exact StarModule.
    `fusion_mode=additive` keeps the same layers but replaces the gated
    multiplication with a plain additive spatial convolution branch.
    """

    def __init__(self, channels: int, mode: str = "conditional", experts: int = 3,
                 patch_size: int = 8, fusion_mode: str = "gated"):
        super().__init__()
        if mode not in {"off", "static", "conditional"}:
            raise ValueError(f"Unsupported spectral mode: {mode}")
        if fusion_mode not in {"gated", "additive"}:
            raise ValueError(f"Unsupported fusion mode: {fusion_mode}")
        if experts < 2 or patch_size < 2:
            raise ValueError("experts and patch_size must be at least 2")
        self.mode, self.channels = mode, channels
        self.fusion_mode = fusion_mode
        self.experts, self.patch_size = experts, patch_size
        self.norm = LayerNorm2d(channels)
        self.project_in = nn.Conv2d(channels, 2 * channels, 1)
        self.depthwise = nn.Conv2d(2 * channels, 2 * channels, 3,
                                   padding=1, groups=2 * channels)
        self.frequency_norm = LayerNorm2d(channels)
        self.spatial_gate = nn.Conv2d(channels, channels, 3, padding=1, groups=channels)
        self.project_out = nn.Conv2d(channels, channels, 1)
        self.residual_scale = nn.Parameter(torch.full((1, channels, 1, 1), 0.1))

        if mode != "off":
            shape = (channels, patch_size, patch_size // 2 + 1)
            self.base_filter = nn.Parameter(torch.zeros(shape))
        if mode == "conditional":
            # Different, trainable starting bases avoid an initially identical
            # expert bank. Their mean is zero, so uniform routing starts neutral.
            fy = torch.fft.fftfreq(patch_size).abs()[:, None]
            fx = torch.fft.rfftfreq(patch_size)[None, :]
            radius = torch.sqrt(fy.square() + fx.square()) / math.sqrt(0.5)
            centers = torch.linspace(0, 1, experts)
            basis = torch.stack([torch.exp(-8 * (radius - center).square())
                                 for center in centers])
            basis = 0.1 * (basis - basis.mean(dim=0, keepdim=True))
            self.filter_basis = nn.Parameter(basis[:, None].repeat(1, channels, 1, 1))
            self.router = nn.Linear(channels + 2, experts)
            nn.init.zeros_(self.router.weight)
            nn.init.zeros_(self.router.bias)

    def frequency_response(self, condition: Tensor):
        """Expose responses for diagnostics; never uses a reference image."""
        if self.mode == "off":
            return None, None
        response = self.base_filter.unsqueeze(0).expand(condition.shape[0], -1, -1, -1)
        weights = None
        if self.mode == "conditional":
            weights = self.router(condition).softmax(dim=-1)
            response = response + torch.einsum("bk,kcpq->bcpq", weights.float(),
                                               self.filter_basis.float())
        return 1.0 + 0.5 * torch.tanh(response.float()), weights

    def _filter(self, x: Tensor, response: Tensor) -> Tensor:
        batch, channels, height, width = x.shape
        size = self.patch_size
        # FFT is deliberately FP32 even when the rest of the network uses AMP.
        with torch.autocast(device_type=x.device.type, enabled=False):
            value = F.pad(x.float(), (0, (-width) % size, 0, (-height) % size),
                          mode="replicate")
            hp, wp = value.shape[-2:]
            tiles = value.reshape(batch, channels, hp // size, size, wp // size, size)
            tiles = tiles.permute(0, 1, 2, 4, 3, 5)
            spectrum = torch.fft.rfft2(tiles, norm="ortho")
            spectrum = spectrum * response[:, :, None, None, :, :]
            tiles = torch.fft.irfft2(spectrum, s=(size, size), norm="ortho")
            value = tiles.permute(0, 1, 2, 4, 3, 5).reshape(batch, channels, hp, wp)
            return value[:, :, :height, :width]

    def forward(self, x: Tensor, condition: Tensor):
        q, v = self.depthwise(self.project_in(self.norm(x))).chunk(2, dim=1)
        response, weights = self.frequency_response(condition)
        if response is not None:
            q = self._filter(q, response)
        if self.fusion_mode == "gated":
            fused = self.frequency_norm(q) * (v * torch.sigmoid(self.spatial_gate(v)))
        else:
            fused = self.frequency_norm(q) + self.spatial_gate(v)
        return x + self.residual_scale * self.project_out(fused), weights


class StarFusion(nn.Module):
    """One full StarIR block replacing only the standalone spatial fusion.

    The original encoder2 and decoder blocks remain. The block contains its
    own residual connections, so the adapter adds no second residual path.
    """

    def __init__(self, channels: int, blocks: int = 1):
        super().__init__()
        from star_blocks import StarBlock
        if blocks < 1:
            raise ValueError("Star blocks must be positive")
        self.block = (StarBlock(channels, ffn_expansion_factor=3.0) if blocks == 1 else
                      nn.Sequential(*[StarBlock(channels, ffn_expansion_factor=3.0)
                                      for _ in range(blocks)]))

    def forward(self, x: Tensor, condition: Tensor):
        return self.block(x), None


class RGBInteractionHead(nn.Module):
    """Per-colour residual on shared features plus the input image.

    Each branch is Conv3x3, GELU, Conv3x3, GELU with fixed width 8.
    A 1x1 mix exchanges the concatenated branches and a sigmoid 1x1 gate
    scales that mix per destination colour. Only the final residual
    convolutions are zero-initialized. This is a proposed adaptation,
    not a LYT or CSEC reproduction.
    """

    def __init__(self, width: int):
        super().__init__()
        branch = 8
        self.branch_width = branch
        branches = []
        for _ in range(3):
            branches.append(nn.Sequential(
                nn.Conv2d(width + 1, branch, 3, padding=1),
                nn.GELU(),
                nn.Conv2d(branch, branch, 3, padding=1),
                nn.GELU(),
            ))
        self.branches = nn.ModuleList(branches)
        self.mix = nn.Conv2d(3 * branch, 3 * branch, 1)
        self.gate = nn.Conv2d(3 * branch, 3, 1)
        self.residual = nn.ModuleList(
            nn.Conv2d(branch, 1, 3, padding=1) for _ in range(3)
        )
        for layer in self.residual:
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, feature: Tensor, image: Tensor) -> Tensor:
        if (feature.ndim != 4 or image.ndim != 4 or image.shape[1] != 3
                or image.shape[0] != feature.shape[0]
                or image.shape[-2:] != feature.shape[-2:]):
            raise ValueError("Expected matching NCHW feature and RGB image")
        parts = []
        for index, branch in enumerate(self.branches):
            channel = image[:, index:index + 1].to(dtype=feature.dtype)
            parts.append(branch(torch.cat((feature, channel), dim=1)))
        fused = torch.cat(parts, dim=1)
        mixed = self.mix(fused)
        gates = torch.sigmoid(self.gate(fused)).to(dtype=fused.dtype)
        channels = self.branch_width
        deltas = []
        for index, layer in enumerate(self.residual):
            start = index * channels
            own = fused[:, start:start + channels]
            cross = mixed[:, start:start + channels]
            updated = own + gates[:, index:index + 1] * cross
            deltas.append(layer(updated))
        return torch.cat(deltas, dim=1)


class EnhancementDemo(nn.Module):
    """RGB [0,1] in, unclamped RGB out; no GT or region masks at inference.

    Structured head: output = input * global_colour_gain * spatial_gain
                            + additive_luminance + zero_luminance_chroma.
    The formula operates on encoded RGB, not radiometrically linear sensor data.
    These terms are computational controls, not identifiable physical factors.
    Direct mode adds one 3x3 residual. rgb_interaction uses three colour
    branches, a gated mix and per-colour residuals. Gains stay neutral in
    both, and neither output is clamped.
    """

    def __init__(self, width: int = 24, spectral_mode: str = "conditional",
                 output_mode: str = "structured", fusion_mode: str = "gated",
                 bottleneck_attention: str = "none", half_decoder: str = "gated",
                 half_encoder_frequency: str = "none", bottleneck_prior: str = "none",
                 illumination_guidance: str = "none", star_refinement: str = "none",
                 lf_guidance: str = "none", star_variant: str = "original",
                 wavelet_variant: str = "none"):
        super().__init__()
        if width < 4:
            raise ValueError("width must be >= 4")
        from wavelet_blocks import WAVELET_VARIANTS
        if wavelet_variant == "all-subband-fusion-mdta" and width % 4:
            raise ValueError("Subband fusion requires width divisible by four")
        if wavelet_variant not in WAVELET_VARIANTS:
            raise ValueError(f"Unsupported wavelet variant: {wavelet_variant}")
        if wavelet_variant != "none" and (
                star_variant != "original" or star_refinement != "multiscale"
                or spectral_mode != "off" or output_mode != "direct" or fusion_mode != "gated"
                or bottleneck_attention != "mdta" or half_decoder != "gated"
                or half_encoder_frequency != "none" or bottleneck_prior != "none"
                or illumination_guidance != "none" or lf_guidance != "none"):
            raise ValueError("Wavelet variants require the original six-Star direct A+MDTA baseline")
        if star_variant not in {"original", "multiscale-amp"}:
            raise ValueError(f"Unsupported Star variant: {star_variant}")
        if star_variant == "multiscale-amp" and (star_refinement != "multiscale"
                                               or lf_guidance != "none" or width % 2):
            raise ValueError("Multiscale amplitude blocks require six Star sites, no LF guidance, and even width")
        if output_mode not in {"structured", "direct", "rgb_interaction"}:
            raise ValueError(f"Unsupported output mode: {output_mode}")
        if fusion_mode not in {"gated", "additive", "none", "star"}:
            raise ValueError(f"Unsupported fusion mode: {fusion_mode}")
        if bottleneck_attention not in {"none", "mdta"}:
            raise ValueError(f"Unsupported bottleneck attention: {bottleneck_attention}")
        if half_decoder not in {"gated", "mdta-gated", "restormer"}:
            raise ValueError(f"Unsupported half decoder: {half_decoder}")
        if fusion_mode == "star" and spectral_mode != "off":
            raise ValueError("Full StarBlock owns its FFT; use spectral_mode=off with fusion_mode=star")
        if half_encoder_frequency not in {"none", "fremlp"}:
            raise ValueError(f"Unsupported half encoder frequency: {half_encoder_frequency}")
        if bottleneck_prior not in {"none", "lfpv"}:
            raise ValueError(f"Unsupported bottleneck prior: {bottleneck_prior}")
        if illumination_guidance not in {"none", "three-stage", "bottom-v"}:
            raise ValueError(f"Unsupported illumination guidance: {illumination_guidance}")
        if illumination_guidance in {"three-stage", "bottom-v"} and (bottleneck_attention != "mdta"
                or half_decoder != "gated" or half_encoder_frequency != "none"
                or bottleneck_prior != "none"):
            raise ValueError("Brightness guidance requires bottom MDTA, original half blocks and no LFPV")
        if star_refinement not in {"none", "bottleneck-two", "multiscale"}:
            raise ValueError(f"Unsupported Star refinement: {star_refinement}")
        if star_refinement != "none" and (spectral_mode != "off" or output_mode != "direct"
                or fusion_mode != "gated" or bottleneck_attention != "mdta"
                or half_decoder != "gated" or half_encoder_frequency != "none"
                or bottleneck_prior != "none" or illumination_guidance != "none"):
            raise ValueError("Star comparisons require the unchanged A+MDTA direct baseline")
        if lf_guidance not in {"none", "decoder1", "decoder1-decoder0"}:
            raise ValueError(f"Unsupported low-frequency guidance: {lf_guidance}")
        if lf_guidance != "none" and star_refinement != "multiscale":
            raise ValueError("Low-frequency guidance requires the Star-Multiscale-A baseline")
        self.config = dict(width=width, spectral_mode=spectral_mode, output_mode=output_mode)
        if wavelet_variant != "none":
            self.config["wavelet_variant"] = wavelet_variant
        if star_variant != "original":
            self.config["star_variant"] = star_variant
        if star_refinement != "none":
            self.config["star_refinement"] = star_refinement
        if lf_guidance != "none":
            self.config["lf_guidance"] = lf_guidance
        if fusion_mode != "gated":
            self.config["fusion_mode"] = fusion_mode
        if bottleneck_attention != "none":
            self.config["bottleneck_attention"] = bottleneck_attention
        if half_decoder != "gated":
            self.config["half_decoder"] = half_decoder
        if half_encoder_frequency != "none":
            self.config["half_encoder_frequency"] = half_encoder_frequency
        if bottleneck_prior != "none":
            self.config["bottleneck_prior"] = bottleneck_prior
        if illumination_guidance != "none":
            self.config["illumination_guidance"] = illumination_guidance
        self.output_mode = output_mode
        self.max_log_gain = math.log(32.0)
        self.max_log_color = math.log(2.0)
        self.stem = nn.Conv2d(3, width, 3, padding=1)
        self.encoder0 = GatedConvBlock(width)
        self.down1 = nn.Conv2d(width, 2 * width, 3, stride=2, padding=1)
        self.encoder1 = GatedConvBlock(2 * width)
        self.down2 = nn.Conv2d(2 * width, 4 * width, 3, stride=2, padding=1)
        self.encoder2 = GatedConvBlock(4 * width)
        self.spectral = ConditionalSpectralBlock(
            4 * width, mode=spectral_mode,
            fusion_mode="gated" if fusion_mode in {"none", "star"} else fusion_mode,
        )
        if fusion_mode == "none":
            # Consume the same initialization draws as A for the later layers,
            # then remove the entire bottleneck block and its parameters.
            self.spectral = None
        self.fuse1 = nn.Conv2d(6 * width, 2 * width, 1)
        self.decoder1 = GatedConvBlock(2 * width)
        self.fuse0 = nn.Conv2d(3 * width, width, 1)
        self.decoder0 = GatedConvBlock(width)

        luminance = torch.tensor([0.2126, 0.7152, 0.0722])
        # Orthonormal basis spanning the plane perpendicular to Rec.709 weights.
        axis1 = torch.tensor([luminance[1], -luminance[0], 0.0])
        axis1 = F.normalize(axis1, dim=0)
        axis2 = F.normalize(torch.linalg.cross(luminance, axis1), dim=0)
        self.register_buffer("luminance_weights", luminance.view(1, 3, 1, 1))
        self.register_buffer("chroma_basis", torch.stack((axis1, axis2), dim=1))

        if output_mode == "structured":
            self.global_color_head = nn.Linear(4 * width, 3)
            self.gain_head = nn.Conv2d(4 * width, 1, 3, padding=1)
            self.gain_condition1 = nn.Conv2d(1, 2 * width, 1)
            self.gain_condition0 = nn.Conv2d(1, width, 1)
            self.luminance_head = nn.Conv2d(width, 1, 3, padding=1)
            self.chroma_head = nn.Conv2d(width, 2, 3, padding=1)
            heads = (self.global_color_head, self.gain_head,
                     self.luminance_head, self.chroma_head)
        elif output_mode == "direct":
            self.direct_head = nn.Conv2d(width, 3, 3, padding=1)
            heads = (self.direct_head,)
        else:
            self.interaction_head = RGBInteractionHead(width)
            heads = ()
        for head in heads:
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)
        if fusion_mode == "star":
            # Initialize shared A layers exactly as the baseline before replacement.
            self.spectral = StarFusion(4 * width)
        # Construct last to preserve shared A initialization.
        self.mdta = None
        if bottleneck_attention == "mdta":
            from mdta_blocks import MDTAResidual
            self.mdta = MDTAResidual(4 * width, LayerNorm2d, heads=4)
        # Initialize new half-decoder components after shared A and bottom MDTA.
        self.half_mdta = None
        if half_decoder == "mdta-gated":
            from mdta_blocks import MDTAResidual
            self.half_mdta = MDTAResidual(2 * width, LayerNorm2d, heads=2)
        elif half_decoder == "restormer":
            from restormer_blocks import RestormerBlock
            self.decoder1 = RestormerBlock(2 * width, heads=2, expansion=2.66)
        # Construct last so common A+MDTA weights retain their seeded initialization.
        self.half_encoder_fremlp = None
        if half_encoder_frequency == "fremlp":
            from fremlp_blocks import FreMLPResidual
            self.half_encoder_fremlp = FreMLPResidual(2 * width, LayerNorm2d)

        # Construct last to preserve all shared A+MDTA initialization draws.
        self.lfpv = None
        if bottleneck_prior == "lfpv":
            from lfpv_blocks import BottomLFPV
            self.lfpv = BottomLFPV(4 * width)

        # Independent brightness branch is initialized after all shared weights.
        self.luma_guidance = None
        if illumination_guidance == "three-stage":
            from illumination_blocks import ThreeStageIllumination
            self.luma_guidance = ThreeStageIllumination(width, LayerNorm2d)
        elif illumination_guidance == "bottom-v":
            from illumination_blocks import BottomVIllumination
            self.luma_guidance = BottomVIllumination(width)

        # Add new modules after every common layer, preserving seeded A+MDTA weights.
        self.star_encoder0 = self.star_encoder1 = None
        self.star_decoder1 = self.star_decoder0 = None
        if star_refinement != "none":
            self.spectral = StarFusion(4 * width, blocks=2)
        if star_refinement == "multiscale":
            from star_blocks import StarBlock
            self.star_encoder0 = StarBlock(width, ffn_expansion_factor=3.0)
            self.star_encoder1 = StarBlock(2 * width, ffn_expansion_factor=3.0)
            self.star_decoder1 = StarBlock(2 * width, ffn_expansion_factor=3.0)
            self.star_decoder0 = StarBlock(width, ffn_expansion_factor=3.0)

        # Append after all six StarBlocks so every shared seeded weight stays equal.
        self.lf_branch = self.lf_film1 = self.lf_film0 = None
        if lf_guidance != "none":
            from lf_guidance import LowFrequencyBranch, GainFiLM
            self.lf_branch = LowFrequencyBranch(GatedConvBlock)
            self.lf_film1 = GainFiLM(2 * width)
            if lf_guidance == "decoder1-decoder0":
                self.lf_film0 = GainFiLM(width)

        # Replace last: common backbone, local FFT, and projection weights retain
        # their original seeded initialization. Removed DFFN FFTs are not registered.
        if star_variant == "multiscale-amp":
            from multiscale_amp_blocks import install_multiscale_amp_blocks
            install_multiscale_amp_blocks(self)
        if wavelet_variant != "none":
            from wavelet_blocks import install_wavelet_blocks
            install_wavelet_blocks(self, wavelet_variant)

    @staticmethod
    def _resize(x: Tensor, shape) -> Tensor:
        return F.interpolate(x, size=shape, mode="bilinear", align_corners=False)

    def forward(self, x: Tensor, return_aux: bool = False, lfpv_training: bool = False,
                lf_bypass: bool = False):
        if x.ndim != 4 or x.shape[1] != 3 or min(x.shape[-2:]) < 1:
            raise ValueError("Expected nonempty NCHW RGB tensor")
        luma_gates = self.luma_guidance.make_gates(x) if self.luma_guidance is not None else None
        lf_gain = self.lf_branch(x) if self.lf_branch is not None else None
        skip0 = self.encoder0(self.stem(x))
        if self.star_encoder0 is not None:
            skip0 = self.star_encoder0(skip0)
        skip1 = self.encoder1(self.down1(skip0))
        if self.star_encoder1 is not None:
            skip1 = self.star_encoder1(skip1)
        if getattr(self, "post_star_encoder1", None) is not None:
            skip1 = self.post_star_encoder1(skip1)
        if self.half_encoder_fremlp is not None:
            skip1 = self.half_encoder_fremlp(skip1)
        if luma_gates is not None and "encoder" in luma_gates:
            skip1 = self.luma_guidance.encoder_attention(skip1, luma_gates["encoder"])
        bottom = self.encoder2(self.down2(skip1))
        pooled = bottom.mean(dim=(-2, -1))
        if self.output_mode == "structured":
            log_color = self.max_log_color * torch.tanh(self.global_color_head(pooled).float())
            log_color = log_color - log_color.mean(dim=1, keepdim=True)
            color_gain = log_color.exp()[:, :, None, None]
            log_gain_small = self.max_log_gain * torch.tanh(self.gain_head(bottom).float())
        else:
            color_gain = x.new_ones(x.shape[0], 3, 1, 1)
            log_gain_small = x.new_zeros(x.shape[0], 1, *bottom.shape[-2:])
        # RMS rather than std keeps the condition differentiable at initialization.
        gain_mean = log_gain_small.mean(dim=(1, 2, 3))
        gain_rms = torch.sqrt(log_gain_small.square().mean(dim=(1, 2, 3)) + 1e-6)
        condition = torch.cat((pooled, gain_mean[:, None], gain_rms[:, None]), dim=1)
        experimental_order = getattr(self, "experimental_bottom_order", None)
        if experimental_order == "bottom-stacked":
            # F1 was encoder2: F1 -> F2 -> F3 -> M1 -> M2.
            bottom, spectral_weights = self.spectral(bottom, condition)
            bottom = self.mdta(bottom)
            bottom = self.mdta_second(bottom)
        elif experimental_order == "bottom-interleaved":
            # F1 was encoder2: F1 -> M1 -> F2 -> M2 -> F3.
            bottom = self.mdta(bottom)
            bottom = self.spectral.block[0](bottom)
            bottom = self.mdta_second(bottom)
            bottom = self.spectral.block[1](bottom)
            spectral_weights = None
        elif self.config.get("wavelet_variant") == "all-subband-interleave-mdta":
            # Same D modules and state keys; change execution order only.
            bottom = self.spectral.block[0](bottom)
            bottom = self.mdta(bottom)
            bottom = self.spectral.block[1](bottom)
            spectral_weights = None
        else:
            if self.spectral is None:
                spectral_weights = None
            else:
                bottom, spectral_weights = self.spectral(bottom, condition)
            if self.mdta is not None:
                bottom = self.mdta(bottom, illumination=luma_gates["bottom"]) if luma_gates is not None else self.mdta(bottom)
        if lfpv_training:
            if self.lfpv is None or not self.training or return_aux:
                raise ValueError("Dual-path LFPV forward requires its training mode")
            base_output = self._decode_features(x, skip0, skip1, bottom,
                                                color_gain, log_gain_small, spectral_weights)
            refined_bottom, update = self.lfpv.update_and_query(bottom)
            output = self._decode_features(x, skip0, skip1, refined_bottom,
                                           color_gain, log_gain_small, spectral_weights)
            return {"output": output, "base_output": base_output, "lfpv_update": update}
        if self.lfpv is not None:
            bottom = self.lfpv(bottom)
        return self._decode_features(x, skip0, skip1, bottom, color_gain,
                                     log_gain_small, spectral_weights, return_aux, luma_gates=luma_gates,
                                     lf_gain=lf_gain, lf_bypass=lf_bypass)

    def _decode_features(self, x, skip0, skip1, bottom, color_gain,
                         log_gain_small, spectral_weights, return_aux=False, luma_gates=None,
                         lf_gain=None, lf_bypass=False):
        lf_stats = {}
        feature = self.fuse1(torch.cat((self._resize(bottom, skip1.shape[-2:]), skip1), dim=1))
        if lf_gain is not None:
            feature, stats = self.lf_film1(feature, lf_gain, diagnostics=return_aux, bypass=lf_bypass)
            lf_stats.update({"lf_decoder1_" + key: value for key, value in stats.items()})
        if self.output_mode == "structured":
            light = self._resize(log_gain_small / self.max_log_gain, feature.shape[-2:])
            feature = feature * (1 + 0.1 * torch.tanh(self.gain_condition1(light)))
        if self.half_mdta is not None:
            feature = self.half_mdta(feature)
        feature = self.decoder1(feature)
        if self.star_decoder1 is not None:
            feature = self.star_decoder1(feature)
        if getattr(self, "post_star_decoder1", None) is not None:
            feature = self.post_star_decoder1(feature)
        if luma_gates is not None and "decoder" in luma_gates:
            feature = self.luma_guidance.decoder_attention(feature, luma_gates["decoder"])
        feature = self.fuse0(torch.cat((self._resize(feature, skip0.shape[-2:]), skip0), dim=1))
        if lf_gain is not None and self.lf_film0 is not None:
            feature, stats = self.lf_film0(feature, lf_gain, diagnostics=return_aux, bypass=lf_bypass)
            lf_stats.update({"lf_decoder0_" + key: value for key, value in stats.items()})
        if self.output_mode == "structured":
            light = self._resize(log_gain_small / self.max_log_gain, feature.shape[-2:])
            feature = feature * (1 + 0.1 * torch.tanh(self.gain_condition0(light)))
        feature = self.decoder0(feature)
        if self.star_decoder0 is not None:
            feature = self.star_decoder0(feature)

        gain = self._resize(log_gain_small, x.shape[-2:]).exp()
        coarse = x * gain * color_gain
        if self.output_mode == "structured":
            luminance_residual = self.luminance_head(feature).float()
            chroma_coordinates = self.chroma_head(feature).float()
            with torch.autocast(device_type=x.device.type, enabled=False):
                chroma_residual = torch.einsum("rc,bchw->brhw", self.chroma_basis.float(),
                                               chroma_coordinates)
            output = coarse + luminance_residual + chroma_residual
        else:
            if self.output_mode == "rgb_interaction":
                residual = self.interaction_head(feature, x).float()
            else:
                residual = self.direct_head(feature).float()
            luminance_residual = (residual * self.luminance_weights).sum(dim=1, keepdim=True)
            chroma_residual = residual - luminance_residual
            output = x + residual
        if not return_aux:
            return output
        result = dict(output=output, coarse=coarse, gain=gain,
                      global_color_gain=color_gain,
                      luminance_residual=luminance_residual,
                      chroma_residual=chroma_residual,
                      spectral_weights=spectral_weights)
        if lf_gain is not None:
            result.update(lf_gain=lf_gain, lf_diagnostics=lf_stats)
        return result
