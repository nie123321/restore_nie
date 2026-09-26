"""Three-resolution U-Net with explicit brightness and directional guidance.

No FFT, global spatial attention, segmentation, or StarIR reproduction.
"""
from __future__ import annotations

import math
import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from .priors import FixedPriors

ARCH_VERSION = "endo_prior_a_v2_1"


def default_config():
    return dict(arch_version=ARCH_VERSION, widths=[32, 64, 128],
                prior_widths=[16, 32, 64], prior_blocks=[1, 2, 2],
                groups_half=4, groups_quarter=8, max_log_gain=math.log(8),
                bias_bound=.1, activation_checkpointing=True)


class LayerNorm2d(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x):
        # FP32 variance prevents FP16 square overflow in multiplicative blocks.
        with torch.amp.autocast(x.device.type, enabled=False):
            variance, mean = torch.var_mean(x.float(), dim=1, keepdim=True, unbiased=False)
            result = (x.float() - mean) * torch.rsqrt(variance + 1e-6)
            result = result * self.weight + self.bias
        return result.to(x.dtype)


class GatedConvBlock(nn.Module):
    """Copied local NAFNet-inspired processing pattern from the A baseline."""
    def __init__(self, channels):
        super().__init__()
        self.norm1, self.norm2 = LayerNorm2d(channels), LayerNorm2d(channels)
        self.expand = nn.Conv2d(channels, 2 * channels, 1)
        self.depthwise = nn.Conv2d(2 * channels, 2 * channels, 3, padding=1, groups=2 * channels)
        self.channel_scale = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(channels, channels, 1))
        self.project = nn.Conv2d(channels, channels, 1)
        self.ffn_in, self.ffn_out = nn.Conv2d(channels, 2 * channels, 1), nn.Conv2d(channels, channels, 1)
        self.scale1 = nn.Parameter(torch.full((1, channels, 1, 1), .1))
        self.scale2 = nn.Parameter(torch.full((1, channels, 1, 1), .1))

    def forward(self, x):
        a, b = self.depthwise(self.expand(self.norm1(x))).chunk(2, 1)
        feature = a * b
        x = x + self.scale1 * self.project(feature * self.channel_scale(feature))
        a, b = self.ffn_in(self.norm2(x)).chunk(2, 1)
        return x + self.scale2 * self.ffn_out(a * b)


class PriorResidual(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.body = nn.Sequential(nn.Conv2d(channels, channels, 3, padding=1), nn.GELU(),
                                  nn.Conv2d(channels, channels, 3, padding=1))

    def forward(self, x):
        return x + .1 * self.body(x)


class PriorEncoder(nn.Module):
    def __init__(self, input_channels, widths, blocks):
        super().__init__()
        self.stem = nn.Conv2d(input_channels, widths[0], 3, padding=1)
        self.levels = nn.ModuleList(nn.Sequential(*(PriorResidual(c) for _ in range(n)))
                                    for c, n in zip(widths, blocks))
        self.down = nn.ModuleList(nn.Conv2d(a, b, 3, stride=2, padding=1)
                                  for a, b in zip(widths[:-1], widths[1:]))

    def forward(self, x):
        result = [self.levels[0](self.stem(x))]
        for down, level in zip(self.down, self.levels[1:]):
            result.append(level(down(result[-1])))
        return result


class BoundedAffine(nn.Module):
    def __init__(self, prior_channels, channels):
        super().__init__()
        self.project = nn.Conv2d(prior_channels, 2 * channels, 1)

    def forward(self, feature, prior):
        gamma, residual = self.project(prior).chunk(2, 1)
        return feature * (1 + .2 * torch.tanh(gamma)) + .1 * torch.tanh(residual)


class GuidedRestoration(nn.Module):
    """Local conv, group-shared dynamic 3x3 aggregation, mixing and FFN.

The nine weights softmax over neighbors (dim=2), independently at each group
and pixel. Contiguous channel groups share one kernel. Nine shifted views
avoid per-channel unfold; Wo=0 starts this aggregation residual at identity.
Logits retain their ordinary initialization. Uniform weights would average
neighbors, not implement identity; the identity comes from the residual.
"""
    def __init__(self, channels, structure_channels, luma_channels, groups):
        super().__init__()
        if channels % groups:
            raise ValueError("channels must be divisible by aggregation groups")
        self.channels, self.groups = channels, groups
        self.local = GatedConvBlock(channels)
        self.norm = LayerNorm2d(channels)
        hidden = channels // 2
        self.weight_features = nn.Sequential(nn.Conv2d(channels + structure_channels, hidden, 1),
                                             nn.GELU(), nn.Conv2d(hidden, hidden, 3, padding=1,
                                                                 groups=hidden), nn.GELU())
        self.logits = nn.Conv2d(hidden, 9 * groups, 1)
        self.value = nn.Conv2d(channels, channels, 1)
        self.luma_value = BoundedAffine(luma_channels, channels)
        self.output = nn.Conv2d(channels, channels, 1)
        self.ffn_norm = LayerNorm2d(channels)
        self.ffn = nn.Sequential(nn.Conv2d(channels, 2 * channels, 1), nn.GELU(),
                                 nn.Conv2d(2 * channels, channels, 1))
        self.ffn_scale = nn.Parameter(torch.full((1, channels, 1, 1), .1))
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def weights(self, feature, structure):
        logits = self.logits(self.weight_features(torch.cat((self.norm(feature), structure), 1)))
        b, _, h, w = logits.shape
        return logits.reshape(b, self.groups, 9, h, w).softmax(dim=2)

    def aggregate(self, value, weights):
        b, c, h, w = value.shape
        padded = F.pad(value, (1, 1, 1, 1), mode="replicate")
        padded = padded.reshape(b, self.groups, c // self.groups, h + 2, w + 2)
        result = None
        for index in range(9):
            yy, xx = divmod(index, 3)
            shifted = padded[..., yy:yy+h, xx:xx+w]
            term = shifted * weights[:, :, index:index+1]
            result = term if result is None else result + term
        return result.reshape(b, c, h, w)

    def guided_residual(self, feature, structure, luma):
        weights = self.weights(feature, structure)
        value = self.luma_value(self.value(self.norm(feature)), luma)
        return feature + self.output(self.aggregate(value, weights))

    def forward(self, feature, structure, luma):
        feature = self.local(feature)
        feature = self.guided_residual(feature, structure, luma)
        return feature + self.ffn_scale * self.ffn(self.ffn_norm(feature))


class PriorAV2(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self._config = default_config()
        if config is not None:
            if set(config) != set(self._config):
                raise ValueError("Model config keys must exactly match this architecture")
            self._config.update(config)
        cfg = self._config
        if cfg["arch_version"] != ARCH_VERSION or cfg["widths"] != [32, 64, 128] or cfg["prior_widths"] != [16, 32, 64] or cfg["prior_blocks"] != [1, 2, 2]:
            raise ValueError("This version fixes the agreed widths and prior depth")
        if not math.isfinite(cfg["max_log_gain"]) or not 0 < cfg["max_log_gain"] <= math.log(8):
            raise ValueError("max_log_gain must be positive and at most log(8)")
        if not math.isfinite(cfg["bias_bound"]) or not 0 <= cfg["bias_bound"] <= .1:
            raise ValueError("bias_bound must be within [0, 0.1]")
        self.priors = FixedPriors()
        self.luma_encoder = PriorEncoder(6, cfg["prior_widths"], cfg["prior_blocks"])
        self.structure_encoder = PriorEncoder(5, cfg["prior_widths"], cfg["prior_blocks"])
        self.brightness_context = nn.Sequential(nn.Conv2d(96, 32, 3, padding=1), nn.GELU())
        self.brightness_head = nn.Conv2d(32, 2, 3, padding=1)
        self.stem = nn.Conv2d(6, 32, 3, padding=1)
        self.encoder0 = GatedConvBlock(32)
        self.down1 = nn.Conv2d(32, 64, 3, stride=2, padding=1)
        self.encoder1 = nn.Sequential(GatedConvBlock(64), GatedConvBlock(64))
        self.luma_half, self.luma_quarter = BoundedAffine(32, 64), BoundedAffine(64, 128)
        self.down2 = nn.Conv2d(64, 128, 3, stride=2, padding=1)
        self.bottleneck = nn.ModuleList(GuidedRestoration(128, 64, 64, cfg["groups_quarter"]) for _ in range(2))
        self.up1, self.merge1 = nn.Conv2d(128, 64, 1), nn.Conv2d(128, 64, 1)
        self.decoder1 = GatedConvBlock(64)
        self.guide_half = GuidedRestoration(64, 32, 32, cfg["groups_half"])
        self.up0, self.merge0 = nn.Conv2d(64, 32, 1), nn.Conv2d(64, 32, 1)
        self.decoder0 = GatedConvBlock(32)
        self.structure_full = BoundedAffine(16, 32)
        self.residual_head = nn.Conv2d(32, 3, 3, padding=1)
        for layer in (self.brightness_head, self.residual_head):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    def config(self):
        import copy
        return copy.deepcopy(self._config)

    def _execute(self, block, *args):
        if self._config["activation_checkpointing"] and self.training and torch.is_grad_enabled():
            return checkpoint(block, *args, use_reentrant=False)
        return block(*args)

    def brightness(self, low, luma):
        context = torch.cat((luma[1], F.interpolate(luma[2], size=luma[1].shape[-2:], mode="bilinear", align_corners=False)), 1)
        gb = self.brightness_head(self.brightness_context(context))
        gb = F.interpolate(gb.float(), size=low.shape[-2:], mode="bilinear", align_corners=False)
        gain = torch.exp(self._config["max_log_gain"] * torch.tanh(gb[:, :1]))
        bias = self._config["bias_bound"] * torch.tanh(gb[:, 1:])
        return low.float() * gain + bias, gain, bias

    def forward(self, low: Tensor, return_aux=False):
        if low.ndim != 4 or low.shape[1] != 3 or min(low.shape[-2:]) < 1:
            raise ValueError("Expected nonempty NCHW RGB input")
        h, w = low.shape[-2:]
        padded = F.pad(low, (0, (-w) % 4, 0, (-h) % 4), mode="replicate")
        lp, sp = self.priors(padded)
        luma = self._execute(self.luma_encoder, lp)
        structure = self._execute(self.structure_encoder, sp)
        corrected, gain, bias = self.brightness(padded, luma)
        full = self._execute(self.encoder0, self.stem(torch.cat((padded, corrected), 1)))
        half = self._execute(self.encoder1, self.luma_half(self.down1(full), luma[1]))
        feature = self.luma_quarter(self.down2(half), luma[2])
        for block in self.bottleneck:
            feature = self._execute(block, feature, structure[2], luma[2])
        feature = self.merge1(torch.cat((self.up1(F.interpolate(feature, size=half.shape[-2:], mode="bilinear", align_corners=False)), half), 1))
        feature = self._execute(self.decoder1, feature)
        feature = self._execute(self.guide_half, feature, structure[1], luma[1])
        feature = self.merge0(torch.cat((self.up0(F.interpolate(feature, size=full.shape[-2:], mode="bilinear", align_corners=False)), full), 1))
        feature = self._execute(self.decoder0, feature)
        residual = self.residual_head(self.structure_full(feature, structure[0])).float()
        output = (corrected + residual)[..., :h, :w]
        if return_aux:
            return dict(output=output, corrected=corrected[..., :h, :w], gain=gain[..., :h, :w],
                        bias=bias[..., :h, :w], residual=residual[..., :h, :w])
        return output
