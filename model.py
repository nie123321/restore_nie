"""A-family U-Net with U3 feature-level luminance and structure guidance.

Main widths 32/64/128; independent prior widths 16/32/64. U3 local
fusion and prior skip fusion replace v2 dynamic aggregation. Output is I+R.
"""
from __future__ import annotations
import copy
import math
import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from .priors import FixedPriors

ARCH_VERSION = "prior_a_u3_v1"


def default_config():
    return dict(arch_version=ARCH_VERSION, widths=[32,64,128], prior_widths=[16,32,64],
                prior_blocks=[1,2,2], bottleneck_blocks=2, half_decoder_prior_blocks=1,
                luma_channels=6, structure_channels=1, structure_scale=.9,
                gamma_limit=.5, output_mode="input_plus_residual", activation_checkpointing=True,
                luma_precision="fp32_encoder_and_gamma")


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


class U3LayerNorm2d(nn.Module):
    def __init__(self, channels: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x: Tensor) -> Tensor:
        device_type = x.device.type if x.device.type in {"cpu", "cuda"} else "cpu"
        with torch.amp.autocast(device_type=device_type, enabled=False):
            # Native LayerNorm accumulates low-precision statistics in FP32;
            # avoid an eager graph retaining several full-sized FP32 buffers.
            value = x.permute(0, 2, 3, 1).contiguous()
            normalized = F.layer_norm(value, (x.shape[1],),
                                      self.weight.reshape(-1).to(value.dtype),
                                      self.bias.reshape(-1).to(value.dtype), self.eps)
            return normalized.permute(0, 3, 1, 2).contiguous()


def _residual_add(feature: Tensor, residual: Tensor, scale: Tensor) -> Tensor:
    # Keep FP32 master parameters without promoting every AMP feature to FP32.
    return feature + scale.to(dtype=residual.dtype) * residual


def _channel_attention(query: Tensor, key: Tensor, value: Tensor, heads: int, raw_temperature: Tensor) -> Tensor:
    batch, channels, height, width = query.shape
    depth = channels // heads
    dtype = query.dtype
    q = F.normalize(query.reshape(batch, heads, depth, height * width).float(), dim=-1, eps=1e-6)
    k = F.normalize(key.reshape(batch, heads, depth, height * width).float(), dim=-1, eps=1e-6)
    v = value.reshape(batch, heads, depth, height * width)
    temperature = F.softplus(raw_temperature.float()).view(1, heads, 1, 1)
    scores = temperature * torch.matmul(q.to(dtype), k.to(dtype).transpose(-2, -1)).float()
    weights = torch.softmax(scores, dim=-1)
    mixed = torch.matmul(weights.to(dtype), v)
    return mixed.reshape(batch, channels, height, width).to(dtype)


class LocalPriorBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        prior = channels // 2
        heads = channels // 32
        self.channels = channels
        self.heads = heads
        self.norm_sf = U3LayerNorm2d(channels)
        self.gamma_in = nn.Conv2d(prior, channels, 1)
        self.gamma_out = nn.Conv2d(channels, channels, 1)
        self.in_sf = nn.Conv2d(channels, 2 * channels, 1)
        self.dw3_sf = nn.Conv2d(2 * channels, 2 * channels, 3, padding=1, groups=2 * channels)
        self.norm_q = U3LayerNorm2d(channels)
        self.dw3_gate = nn.Conv2d(channels, channels, 3, padding=1, groups=channels)
        self.out_sf = nn.Conv2d(channels, channels, 1)
        self.scale_sf = nn.Parameter(torch.full((1, channels, 1, 1), 0.1))
        self.norm_image = U3LayerNorm2d(channels)
        self.q_proj = nn.Conv2d(channels, channels, 1)
        self.q_dw = nn.Conv2d(channels, channels, 3, padding=1, groups=channels)
        self.norm_structure = U3LayerNorm2d(prior)
        self.kv_proj = nn.Conv2d(prior, 2 * channels, 1)
        self.kv_dw = nn.Conv2d(2 * channels, 2 * channels, 3, padding=1, groups=2 * channels)
        self.raw_temperature = nn.Parameter(torch.full((heads,), math.log(math.exp(1.0) - 1.0)))
        self.structure_out = nn.Conv2d(channels, channels, 1)
        self.scale_structure = nn.Parameter(torch.full((1, channels, 1, 1), 0.1))
        self.norm_ffn = U3LayerNorm2d(channels)
        self.ffn_in = nn.Conv2d(channels, 4 * channels, 1)
        self.ffn_dw = nn.Conv2d(4 * channels, 4 * channels, 3, padding=1, groups=4 * channels)
        self.ffn_out = nn.Conv2d(2 * channels, channels, 1)
        self.scale_ffn = nn.Parameter(torch.full((1, channels, 1, 1), 0.1))
        nn.init.normal_(self.gamma_out.weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.gamma_out.bias)

    def forward(self, feature: Tensor, luma: Tensor, structure: Tensor, return_gamma: bool = False):
        # The small U3 gamma-head initialization can underflow the deep luma
        # encoder gradient in FP16. Preserve this conditioning path in FP32.
        with torch.amp.autocast(feature.device.type, enabled=False):
            gamma = 0.5 * torch.tanh(self.gamma_out(F.gelu(self.gamma_in(luma.float()))))
        mixed = self.norm_sf(feature) * (1.0 + gamma)
        query, value = self.dw3_sf(self.in_sf(mixed)).chunk(2, dim=1)
        spatial = self.norm_q(query) * (value * torch.sigmoid(self.dw3_gate(value)))
        updated = _residual_add(feature, self.out_sf(spatial), self.scale_sf)
        image_q = self.q_dw(self.q_proj(self.norm_image(updated)))
        key, val = self.kv_dw(self.kv_proj(self.norm_structure(structure))).chunk(2, dim=1)
        attended = _channel_attention(image_q, key, val, self.heads, self.raw_temperature)
        updated = _residual_add(updated, self.structure_out(attended), self.scale_structure)
        hidden = self.ffn_in(self.norm_ffn(updated))
        first, second = self.ffn_dw(hidden).chunk(2, dim=1)
        updated = _residual_add(updated, self.ffn_out(F.gelu(first) * second), self.scale_ffn)
        if return_gamma:
            return updated, gamma
        return updated


class PriorSkipFusion(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        prior = channels // 2
        self.norm_u = U3LayerNorm2d(channels)
        self.norm_e = U3LayerNorm2d(channels)
        self.norm_l = U3LayerNorm2d(prior)
        self.norm_s = U3LayerNorm2d(prior)
        self.gate_in = nn.Conv2d(3 * channels, channels, 1)
        self.gate_dw = nn.Conv2d(channels, channels, 3, padding=1, groups=channels)
        self.gate_out = nn.Conv2d(channels, channels, 1)
        self.mix = nn.Conv2d(2 * channels, channels, 1)
        nn.init.normal_(self.gate_out.weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.gate_out.bias)

    def forward(self, deep: Tensor, skip: Tensor, luma: Tensor, structure: Tensor, return_gate: bool = False):
        tokens = torch.cat((
            self.norm_u(deep), self.norm_e(skip), self.norm_l(luma), self.norm_s(structure),
        ), dim=1)
        logits = self.gate_out(F.gelu(self.gate_dw(F.gelu(self.gate_in(tokens)))))
        gate = 2.0 * torch.sigmoid(logits)
        fused = self.mix(torch.cat((deep, skip * gate), dim=1))
        if return_gate:
            return fused, gate
        return fused


class PriorAU3(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        expected=default_config()
        self._config=copy.deepcopy(expected if config is None else config)
        if set(self._config)!=set(expected):
            raise ValueError("Architecture configuration keys differ")
        for key,value in expected.items():
            if key != "activation_checkpointing" and self._config[key] != value:
                raise ValueError(f"Architecture setting differs: {key}")
        if not isinstance(self._config["activation_checkpointing"],bool):
            raise ValueError("activation_checkpointing must be boolean")
        self.priors=FixedPriors()
        self.luma_encoder=PriorEncoder(6,[16,32,64],[1,2,2])
        self.structure_encoder=PriorEncoder(1,[16,32,64],[1,2,2])
        self.stem=nn.Conv2d(3,32,3,padding=1)
        self.encoder0=GatedConvBlock(32)
        self.down1=nn.Conv2d(32,64,3,stride=2,padding=1)
        self.encoder1=nn.Sequential(GatedConvBlock(64),GatedConvBlock(64))
        self.down2=nn.Conv2d(64,128,3,stride=2,padding=1)
        self.bottleneck=nn.ModuleList(LocalPriorBlock(128) for _ in range(2))
        self.up1=nn.Conv2d(128,64,1)
        self.skip1=PriorSkipFusion(64)
        self.decoder1=GatedConvBlock(64)
        self.guide_half=LocalPriorBlock(64)
        self.up0=nn.Conv2d(64,32,1)
        self.skip0=PriorSkipFusion(32)
        self.decoder0=GatedConvBlock(32)
        self.residual_head=nn.Conv2d(32,3,3,padding=1)
        # Preserve U3's zero convolution biases within its transplanted blocks.
        for block in self.modules():
            if isinstance(block,(LocalPriorBlock,PriorSkipFusion)):
                for layer in block.modules():
                    if isinstance(layer,nn.Conv2d) and layer.bias is not None:
                        nn.init.zeros_(layer.bias)
        nn.init.zeros_(self.residual_head.weight)
        nn.init.zeros_(self.residual_head.bias)

    def config(self):
        return copy.deepcopy(self._config)

    def _execute(self,block,*args):
        if self._config["activation_checkpointing"] and self.training and torch.is_grad_enabled():
            return checkpoint(block,*args,use_reentrant=False)
        return block(*args)

    def forward(self,low: Tensor):
        if low.ndim!=4 or low.shape[1]!=3 or min(low.shape[-2:])<1:
            raise ValueError("Expected nonempty NCHW RGB input")
        h,w=low.shape[-2:]
        # Compute original U3 priors on actual pixels, then align all branches.
        lp,sp=self.priors(low)
        padding=(0,(-w)%4,0,(-h)%4)
        image=F.pad(low,padding,mode="replicate")
        lp=F.pad(lp,padding,mode="replicate")
        sp=F.pad(sp,padding,mode="replicate")
        with torch.amp.autocast(low.device.type,enabled=False):
            luma=self._execute(self.luma_encoder,lp.float())
        structure=self._execute(self.structure_encoder,sp)
        full=self._execute(self.encoder0,self.stem(image))
        half=self._execute(self.encoder1,self.down1(full))
        feature=self.down2(half)
        for block in self.bottleneck:
            feature=self._execute(block,feature,luma[2],structure[2])
        feature=self.up1(F.interpolate(feature,size=half.shape[-2:],mode="bilinear",align_corners=False))
        feature=self._execute(self.skip1,feature,half,luma[1],structure[1])
        feature=self._execute(self.decoder1,feature)
        feature=self._execute(self.guide_half,feature,luma[1],structure[1])
        feature=self.up0(F.interpolate(feature,size=full.shape[-2:],mode="bilinear",align_corners=False))
        feature=self._execute(self.skip0,feature,full,luma[0],structure[0])
        feature=self._execute(self.decoder0,feature)
        residual=self.residual_head(feature).float()
        return (image.float()+residual)[...,:h,:w]
