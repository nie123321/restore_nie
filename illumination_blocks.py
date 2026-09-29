"""Explicit low-image brightness features and Retinexformer-style IG-MSA.

IG-MSA follows caiyuanhao1998/Retinexformer, including its KQ attention
ordering and the ungated-V positional branch. The brightness extractor,
bounded neutral gates, residual wrappers and insertion sites are A adaptations.
"""
import torch
from torch import nn
import torch.nn.functional as F


def brightness_maps(low):
    """Multinex's six encoded-RGB brightness descriptors; no GT input."""
    rgb = low.float()
    r, g, b = rgb.split(1, dim=1)
    maximum = rgb.amax(dim=1, keepdim=True)
    minimum = rgb.amin(dim=1, keepdim=True)
    return torch.cat((rgb.mean(dim=1, keepdim=True),
                      0.2126 * r + 0.7152 * g + 0.0722 * b,
                      maximum, 0.5 * (maximum + minimum),
                      0.25 * r + 0.5 * g + 0.25 * b,
                      torch.sqrt(rgb.square().sum(dim=1, keepdim=True) + 1e-8)), dim=1)


def guided_value(value, gate):
    if gate.shape != value.shape:
        raise ValueError(f"Brightness gate {tuple(gate.shape)} differs from V {tuple(value.shape)}")
    # Keep the learned prior and multiplication in FP32, then return to AMP
    # attention dtype. This avoids FP16 prior convolutions and parameters.
    with torch.autocast(device_type=value.device.type, enabled=False):
        return (value.float() * gate.float()).to(value.dtype)


class IGMSAAttention(nn.Module):
    """Complete IG-MSA attention core, with NCHW inputs; no IGAB FFN."""
    def __init__(self, channels, heads=2):
        super().__init__()
        if channels % heads:
            raise ValueError("Channels must be divisible by heads")
        self.num_heads = heads
        self.dim_head = channels // heads
        self.to_q = nn.Linear(channels, channels, bias=False)
        self.to_k = nn.Linear(channels, channels, bias=False)
        self.to_v = nn.Linear(channels, channels, bias=False)
        self.rescale = nn.Parameter(torch.ones(heads, 1, 1))
        self.proj = nn.Linear(channels, channels, bias=True)
        self.pos_emb = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False),
        )

    def forward(self, feature, gate):
        b, c, h, w = feature.shape
        tokens = feature.permute(0, 2, 3, 1).reshape(b, h * w, c)
        q, k, original_v = self.to_q(tokens), self.to_k(tokens), self.to_v(tokens)
        v_image = original_v.reshape(b, h, w, c).permute(0, 3, 1, 2)
        v = guided_value(v_image, gate).permute(0, 2, 3, 1).reshape(b, h * w, c)
        q, k, v = (item.reshape(b, h * w, self.num_heads, self.dim_head)
                   .permute(0, 2, 3, 1) for item in (q, k, v))
        q, k = F.normalize(q, dim=-1), F.normalize(k, dim=-1)
        attention = ((k @ q.transpose(-2, -1)) * self.rescale).softmax(dim=-1)
        attended = (attention @ v).permute(0, 3, 1, 2).reshape(b, h * w, c)
        channel_out = self.proj(attended).reshape(b, h, w, c).permute(0, 3, 1, 2)
        return channel_out + self.pos_emb(v_image)


class IGMSAResidual(nn.Module):
    def __init__(self, channels, norm_layer, heads=2):
        super().__init__()
        self.norm = norm_layer(channels)
        self.attn = IGMSAAttention(channels, heads)
        # A-specific neutral insertion. The original GatedConvBlock stays.
        nn.init.zeros_(self.attn.proj.weight)
        nn.init.zeros_(self.attn.proj.bias)
        nn.init.zeros_(self.attn.pos_emb[-1].weight)

    def forward(self, feature, gate):
        return feature + self.attn(self.norm(feature), gate)


class ThreeStageIllumination(nn.Module):
    def __init__(self, width, norm_layer):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(6, 16, 3, padding=1), nn.GELU(),
            nn.Conv2d(16, 16, 3, padding=1, groups=16), nn.GELU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.GELU(),
            nn.Conv2d(32, 32, 3, padding=1, groups=32), nn.GELU(),
        )
        self.down = nn.Sequential(
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.GELU(),
            nn.Conv2d(64, 64, 3, padding=1, groups=64), nn.GELU(),
        )
        self.gate_encoder = nn.Conv2d(32, 2 * width, 1)
        self.gate_bottom = nn.Conv2d(64, 4 * width, 1)
        self.gate_decoder = nn.Conv2d(32, 2 * width, 1)
        for head in (self.gate_encoder, self.gate_bottom, self.gate_decoder):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)
        self.encoder_attention = IGMSAResidual(2 * width, norm_layer, heads=2)
        self.decoder_attention = IGMSAResidual(2 * width, norm_layer, heads=2)

    def make_gates(self, low):
        with torch.autocast(device_type=low.device.type, enabled=False):
            half = self.encoder(brightness_maps(low))
            bottom = self.down(half)
            return {"encoder": 1 + torch.tanh(self.gate_encoder(half)),
                    "bottom": 1 + torch.tanh(self.gate_bottom(bottom)),
                    "decoder": 1 + torch.tanh(self.gate_decoder(half))}



class BottomVIllumination(nn.Module):
    """Low RGB brightness branch used only by existing bottom MDTA's V."""
    def __init__(self, width):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(6, 16, 3, padding=1), nn.GELU(),
            nn.Conv2d(16, 16, 3, padding=1, groups=16), nn.GELU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.GELU(),
            nn.Conv2d(32, 32, 3, padding=1, groups=32), nn.GELU(),
        )
        self.down = nn.Sequential(
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.GELU(),
            nn.Conv2d(64, 64, 3, padding=1, groups=64), nn.GELU(),
        )
        self.gate_bottom = nn.Conv2d(64, 4 * width, 1)
        nn.init.zeros_(self.gate_bottom.weight)
        nn.init.zeros_(self.gate_bottom.bias)

    def make_gates(self, low):
        with torch.autocast(device_type=low.device.type, enabled=False):
            bottom = self.down(self.encoder(brightness_maps(low)))
            return {"bottom": 1 + torch.tanh(self.gate_bottom(bottom))}
