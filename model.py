"""Prior-Fusion-U3: three-scale U-Net with two prior encoders.

Spatial mixing follows A's gated Spatial Fusion core, without Fourier
parameters. This is not a StarIR reproduction.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

from priors import CIConvW, six_luminance


def _zero_conv_bias(module: nn.Module) -> None:
    for child in module.modules():
        if isinstance(child, nn.Conv2d) and child.bias is not None:
            nn.init.zeros_(child.bias)


def _residual_add(feature: Tensor, residual: Tensor, scale: Tensor) -> Tensor:
    # Keep FP32 master parameters without promoting every AMP feature to FP32.
    return feature + scale.to(dtype=residual.dtype) * residual


def _run_block(module: nn.Module, *inputs: Tensor, enabled: bool = True):
    if enabled and module.training and torch.is_grad_enabled():
        return checkpoint(module, *inputs, use_reentrant=False)
    return module(*inputs)


class LayerNorm2d(nn.Module):
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


class PriorResBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.norm = LayerNorm2d(channels)
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.scale = nn.Parameter(torch.full((1, channels, 1, 1), 0.1))

    def forward(self, x: Tensor) -> Tensor:
        residual = self.conv2(F.gelu(self.conv1(self.norm(x))))
        return _residual_add(x, residual, self.scale)


class PriorEncoder(nn.Module):
    def __init__(self, in_channels: int, activation_checkpointing: bool = True):
        super().__init__()
        self.activation_checkpointing = activation_checkpointing
        self.stem = nn.Conv2d(in_channels, 32, 3, padding=1)
        self.block0 = nn.Sequential(PriorResBlock(32), PriorResBlock(32))
        self.down1 = nn.Conv2d(32, 64, 3, stride=2, padding=1)
        self.block1 = nn.Sequential(PriorResBlock(64), PriorResBlock(64))
        self.down2 = nn.Conv2d(64, 128, 3, stride=2, padding=1)
        self.block2 = nn.Sequential(PriorResBlock(128), PriorResBlock(128))

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        p0 = _run_block(self.block0, self.stem(x), enabled=self.activation_checkpointing)
        p1 = _run_block(self.block1, self.down1(p0), enabled=self.activation_checkpointing)
        p2 = _run_block(self.block2, self.down2(p1), enabled=self.activation_checkpointing)
        return p0, p1, p2


class LocalPriorBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        prior = channels // 2
        heads = channels // 32
        self.channels = channels
        self.heads = heads
        self.norm_sf = LayerNorm2d(channels)
        self.gamma_in = nn.Conv2d(prior, channels, 1)
        self.gamma_out = nn.Conv2d(channels, channels, 1)
        self.in_sf = nn.Conv2d(channels, 2 * channels, 1)
        self.dw3_sf = nn.Conv2d(2 * channels, 2 * channels, 3, padding=1, groups=2 * channels)
        self.norm_q = LayerNorm2d(channels)
        self.dw3_gate = nn.Conv2d(channels, channels, 3, padding=1, groups=channels)
        self.out_sf = nn.Conv2d(channels, channels, 1)
        self.scale_sf = nn.Parameter(torch.full((1, channels, 1, 1), 0.1))
        self.norm_image = LayerNorm2d(channels)
        self.q_proj = nn.Conv2d(channels, channels, 1)
        self.q_dw = nn.Conv2d(channels, channels, 3, padding=1, groups=channels)
        self.norm_structure = LayerNorm2d(prior)
        self.kv_proj = nn.Conv2d(prior, 2 * channels, 1)
        self.kv_dw = nn.Conv2d(2 * channels, 2 * channels, 3, padding=1, groups=2 * channels)
        self.raw_temperature = nn.Parameter(torch.full((heads,), math.log(math.exp(1.0) - 1.0)))
        self.structure_out = nn.Conv2d(channels, channels, 1)
        self.scale_structure = nn.Parameter(torch.full((1, channels, 1, 1), 0.1))
        self.norm_ffn = LayerNorm2d(channels)
        self.ffn_in = nn.Conv2d(channels, 4 * channels, 1)
        self.ffn_dw = nn.Conv2d(4 * channels, 4 * channels, 3, padding=1, groups=4 * channels)
        self.ffn_out = nn.Conv2d(2 * channels, channels, 1)
        self.scale_ffn = nn.Parameter(torch.full((1, channels, 1, 1), 0.1))
        nn.init.normal_(self.gamma_out.weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.gamma_out.bias)

    def forward(self, feature: Tensor, luma: Tensor, structure: Tensor, return_gamma: bool = False):
        gamma = 0.5 * torch.tanh(self.gamma_out(F.gelu(self.gamma_in(luma))))
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


class GlobalSpatialBlock(nn.Module):
    def __init__(self, channels: int, heads: int = 8, reduction: int = 4):
        super().__init__()
        self.heads = heads
        self.reduction = reduction
        self.head_dim = channels // heads
        self.norm_attn = LayerNorm2d(channels)
        self.q_proj = nn.Conv2d(channels, channels, 1)
        self.q_dw = nn.Conv2d(channels, channels, 3, padding=1, groups=channels)
        self.kv_proj = nn.Conv2d(channels, 2 * channels, 1)
        self.kv_dw = nn.Conv2d(2 * channels, 2 * channels, 3, padding=1, groups=2 * channels)
        self.out_proj = nn.Conv2d(channels, channels, 1)
        self.scale_attn = nn.Parameter(torch.full((1, channels, 1, 1), 0.1))
        self.norm_ffn = LayerNorm2d(channels)
        self.ffn_in = nn.Conv2d(channels, 4 * channels, 1)
        self.ffn_dw = nn.Conv2d(4 * channels, 4 * channels, 3, padding=1, groups=4 * channels)
        self.ffn_out = nn.Conv2d(2 * channels, channels, 1)
        self.scale_ffn = nn.Parameter(torch.full((1, channels, 1, 1), 0.1))

    def forward(self, feature: Tensor) -> Tensor:
        normalized = self.norm_attn(feature)
        query = self.q_dw(self.q_proj(normalized))
        height, width = feature.shape[-2:]
        pooled = F.adaptive_avg_pool2d(
            normalized, (max(1, math.ceil(height / self.reduction)), max(1, math.ceil(width / self.reduction))),
        )
        key, value = self.kv_dw(self.kv_proj(pooled)).chunk(2, dim=1)
        attended = _spatial_attention(query, key, value, self.heads, self.head_dim)
        updated = _residual_add(feature, self.out_proj(attended), self.scale_attn)
        hidden = self.ffn_in(self.norm_ffn(updated))
        first, second = self.ffn_dw(hidden).chunk(2, dim=1)
        return _residual_add(updated, self.ffn_out(F.gelu(first) * second), self.scale_ffn)


class PriorSkipFusion(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        prior = channels // 2
        self.norm_u = LayerNorm2d(channels)
        self.norm_e = LayerNorm2d(channels)
        self.norm_l = LayerNorm2d(prior)
        self.norm_s = LayerNorm2d(prior)
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


def _spatial_attention(query: Tensor, key: Tensor, value: Tensor, heads: int, head_dim: int) -> Tensor:
    batch, _, height, width = query.shape
    positions = height * width
    query = query.reshape(batch, heads, head_dim, positions).permute(0, 1, 3, 2).contiguous()
    key = key.reshape(batch, heads, head_dim, -1).permute(0, 1, 3, 2).contiguous()
    value = value.reshape(batch, heads, head_dim, -1).permute(0, 1, 3, 2).contiguous()
    mixed = F.scaled_dot_product_attention(query, key, value, dropout_p=0.0, is_causal=False)
    return mixed.permute(0, 1, 3, 2).contiguous().view(batch, heads * head_dim, height, width)


class PriorFusionU3(nn.Module):
    def __init__(self, config: dict | None = None):
        super().__init__()
        settings = default_config()
        if config:
            settings.update(config)
        self._validate(settings)
        self.settings = settings
        self.luma_input_mode = settings["luma_input_mode"]
        self.structure_input_mode = settings["structure_input_mode"]
        self.aux_outputs = settings["aux_outputs"]
        self.activation_checkpointing = settings["activation_checkpointing"]
        self.stem = nn.Conv2d(3, 64, 3, padding=1)
        self.e0 = nn.ModuleList([LocalPriorBlock(64), LocalPriorBlock(64)])
        self.down1 = nn.Conv2d(64, 128, 3, stride=2, padding=1)
        self.e1 = nn.ModuleList([LocalPriorBlock(128) for _ in range(3)])
        self.down2 = nn.Conv2d(128, 256, 3, stride=2, padding=1)
        self.b0 = LocalPriorBlock(256)
        self.b1 = GlobalSpatialBlock(256)
        self.b2 = LocalPriorBlock(256)
        self.b3 = GlobalSpatialBlock(256)
        self.up1 = nn.Conv2d(256, 128, 1)
        self.skip1 = PriorSkipFusion(128)
        self.d1 = nn.ModuleList([LocalPriorBlock(128) for _ in range(3)])
        self.up0 = nn.Conv2d(128, 64, 1)
        self.skip0 = PriorSkipFusion(64)
        self.d0 = nn.ModuleList([LocalPriorBlock(64), LocalPriorBlock(64)])
        self.head = nn.Conv2d(64, 3, 3, padding=1)
        self.luma_encoder = PriorEncoder(6, self.activation_checkpointing)
        self.structure_encoder = PriorEncoder(1, self.activation_checkpointing)
        self.luma_prior = six_luminance
        self.structure_prior = CIConvW()
        if self.aux_outputs:
            self.head_half = nn.Conv2d(128, 3, 3, padding=1)
            self.head_quarter = nn.Conv2d(256, 3, 3, padding=1)
            nn.init.normal_(self.head_half.weight, mean=0.0, std=1e-3)
            nn.init.normal_(self.head_quarter.weight, mean=0.0, std=1e-3)
        _zero_conv_bias(self)
        nn.init.normal_(self.head.weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.head.bias)
        for block in self.modules():
            if isinstance(block, LocalPriorBlock):
                nn.init.normal_(block.gamma_out.weight, mean=0.0, std=1e-3)
                nn.init.zeros_(block.gamma_out.bias)
            if isinstance(block, PriorSkipFusion):
                nn.init.normal_(block.gate_out.weight, mean=0.0, std=1e-3)
                nn.init.zeros_(block.gate_out.bias)

    def config(self) -> dict:
        return dict(self.settings)

    def forward(self, x: Tensor, return_debug: bool = False, return_aux: bool = False):
        if x.ndim != 4 or x.shape[1] != 3 or min(x.shape[-2:]) < 1:
            raise ValueError("Expected nonempty NCHW RGB tensor")
        device_type = x.device.type if x.device.type in {"cpu", "cuda"} else "cpu"
        with torch.amp.autocast(device_type=device_type, enabled=False):
            luma = six_luminance(x.float())
            structure = self.structure_prior(x.float())
        if self.luma_input_mode == "zero":
            luma = torch.zeros_like(luma)
        if self.structure_input_mode == "zero":
            structure = torch.zeros_like(structure)
        luma_scales = self.luma_encoder(luma)
        structure_scales = self.structure_encoder(structure)
        feature = self.stem(x)
        recompute = self.activation_checkpointing and not return_debug
        gamma = None
        for index, block in enumerate(self.e0):
            if return_debug and index == 0:
                feature, gamma = block(feature, luma_scales[0], structure_scales[0], return_gamma=True)
            else:
                feature = _run_block(block, feature, luma_scales[0], structure_scales[0], enabled=recompute)
        skip0 = feature
        feature = self.down1(feature)
        for block in self.e1:
            feature = _run_block(block, feature, luma_scales[1], structure_scales[1], enabled=recompute)
        skip1 = feature
        feature = self.down2(feature)
        feature = _run_block(self.b0, feature, luma_scales[2], structure_scales[2], enabled=recompute)
        feature = _run_block(self.b1, feature, enabled=recompute)
        feature = _run_block(self.b2, feature, luma_scales[2], structure_scales[2], enabled=recompute)
        bottleneck = _run_block(self.b3, feature, enabled=recompute)
        half = self.up1(F.interpolate(bottleneck, size=skip1.shape[-2:], mode="bilinear", align_corners=False))
        if return_debug:
            half, gate1 = self.skip1(half, skip1, luma_scales[1], structure_scales[1], return_gate=True)
        else:
            half = _run_block(self.skip1, half, skip1, luma_scales[1], structure_scales[1], enabled=recompute)
            gate1 = None
        for block in self.d1:
            half = _run_block(block, half, luma_scales[1], structure_scales[1], enabled=recompute)
        full = self.up0(F.interpolate(half, size=skip0.shape[-2:], mode="bilinear", align_corners=False))
        if return_debug:
            full, gate0 = self.skip0(full, skip0, luma_scales[0], structure_scales[0], return_gate=True)
        else:
            full = _run_block(self.skip0, full, skip0, luma_scales[0], structure_scales[0], enabled=recompute)
            gate0 = None
        for block in self.d0:
            full = _run_block(block, full, luma_scales[0], structure_scales[0], enabled=recompute)
        output = x.float() + self.head(full).float()
        if return_aux and self.aux_outputs:
            pred_half = F.interpolate(x.float(), size=half.shape[-2:], mode="area") + self.head_half(half).float()
            pred_quarter = F.interpolate(x.float(), size=bottleneck.shape[-2:], mode="area") + self.head_quarter(bottleneck).float()
            return {"output": output, "pred_half": pred_half, "pred_quarter": pred_quarter}
        if not return_debug:
            return output
        return {
            "output": output,
            "luma": luma_scales,
            "structure": structure_scales,
            "gamma": gamma,
            "skip_gate0": gate0,
            "skip_gate1": gate1,
        }


def default_config() -> dict:
    return {
        "arch_version": "prior_fusion_u3_v1",
        "input_color": "RGB",
        "input_range": [0, 1],
        "main_channels": [64, 128, 256],
        "prior_channels": [32, 64, 128],
        "encoder_blocks": [2, 3],
        "bottleneck_blocks": ["local_prior", "global_spatial", "local_prior", "global_spatial"],
        "decoder_blocks": [3, 2],
        "prior_blocks_per_scale": [2, 2, 2],
        "structure_heads": [2, 4, 8],
        "global_heads": 8,
        "global_kv_reduction": 4,
        "ffn_expansion": 2,
        "gamma_limit": 0.5,
        "residual_scale_init": 0.1,
        "norm_eps": 1e-6,
        "conv_bias": True,
        "dropout": 0,
        "drop_path": 0,
        "fft_enabled": False,
        "aux_outputs": False,
        "activation_checkpointing": True,
        "luma_input_mode": "real",
        "structure_input_mode": "real",
    }


def _validate(settings: dict) -> None:
    if settings.get("arch_version") != "prior_fusion_u3_v1":
        raise ValueError("Unsupported architecture")
    if settings.get("fft_enabled"):
        raise ValueError("This version does not enable FFT")
    if settings.get("luma_input_mode") not in {"real", "zero"}:
        raise ValueError("luma_input_mode must be real or zero")
    if settings.get("structure_input_mode") not in {"real", "zero"}:
        raise ValueError("structure_input_mode must be real or zero")


PriorFusionU3._validate = staticmethod(_validate)
