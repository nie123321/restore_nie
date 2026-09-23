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
    """

    def __init__(self, channels: int, mode: str = "conditional", experts: int = 3,
                 patch_size: int = 8):
        super().__init__()
        if mode not in {"off", "static", "conditional"}:
            raise ValueError(f"Unsupported spectral mode: {mode}")
        if experts < 2 or patch_size < 2:
            raise ValueError("experts and patch_size must be at least 2")
        self.mode, self.channels = mode, channels
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
        fused = self.frequency_norm(q) * (v * torch.sigmoid(self.spatial_gate(v)))
        return x + self.residual_scale * self.project_out(fused), weights


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
                 output_mode: str = "structured"):
        super().__init__()
        if width < 4:
            raise ValueError("width must be >= 4")
        if output_mode not in {"structured", "direct", "rgb_interaction"}:
            raise ValueError(f"Unsupported output mode: {output_mode}")
        self.config = dict(width=width, spectral_mode=spectral_mode, output_mode=output_mode)
        self.output_mode = output_mode
        self.max_log_gain = math.log(32.0)
        self.max_log_color = math.log(2.0)
        self.stem = nn.Conv2d(3, width, 3, padding=1)
        self.encoder0 = GatedConvBlock(width)
        self.down1 = nn.Conv2d(width, 2 * width, 3, stride=2, padding=1)
        self.encoder1 = GatedConvBlock(2 * width)
        self.down2 = nn.Conv2d(2 * width, 4 * width, 3, stride=2, padding=1)
        self.encoder2 = GatedConvBlock(4 * width)
        self.spectral = ConditionalSpectralBlock(4 * width, mode=spectral_mode)
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

    @staticmethod
    def _resize(x: Tensor, shape) -> Tensor:
        return F.interpolate(x, size=shape, mode="bilinear", align_corners=False)

    def forward(self, x: Tensor, return_aux: bool = False):
        if x.ndim != 4 or x.shape[1] != 3 or min(x.shape[-2:]) < 1:
            raise ValueError("Expected nonempty NCHW RGB tensor")
        skip0 = self.encoder0(self.stem(x))
        skip1 = self.encoder1(self.down1(skip0))
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
        bottom, spectral_weights = self.spectral(bottom, condition)
        feature = self.fuse1(torch.cat((self._resize(bottom, skip1.shape[-2:]), skip1), dim=1))
        if self.output_mode == "structured":
            light = self._resize(log_gain_small / self.max_log_gain, feature.shape[-2:])
            feature = feature * (1 + 0.1 * torch.tanh(self.gain_condition1(light)))
        feature = self.decoder1(feature)
        feature = self.fuse0(torch.cat((self._resize(feature, skip0.shape[-2:]), skip0), dim=1))
        if self.output_mode == "structured":
            light = self._resize(log_gain_small / self.max_log_gain, feature.shape[-2:])
            feature = feature * (1 + 0.1 * torch.tanh(self.gain_condition0(light)))
        feature = self.decoder0(feature)

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
        return dict(output=output, coarse=coarse, gain=gain,
                    global_color_gain=color_gain,
                    luminance_residual=luminance_residual,
                    chroma_residual=chroma_residual,
                    spectral_weights=spectral_weights)
