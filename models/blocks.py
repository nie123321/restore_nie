import torch
import torch.nn.functional as F
from torch import Tensor, nn


class LayerNorm2d(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x: Tensor) -> Tensor:
        (variance, mean) = torch.var_mean(x, dim=1, keepdim=True, unbiased=False)
        return (x - mean) * torch.rsqrt(variance + 1e-06) * self.weight + self.bias


class SpatialGatedPath(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.spatial_norm = LayerNorm2d(channels)
        self.spatial_in = nn.Conv2d(channels, 2 * channels, 1)
        self.spatial_dw = nn.Conv2d(2 * channels, 2 * channels, 3, padding=1, groups=2 * channels)
        self.channel_gate = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(channels, channels, 1))
        self.spatial_out = nn.Conv2d(channels, channels, 1)
        self.ffn_norm = LayerNorm2d(channels)
        self.ffn_in = nn.Conv2d(channels, 2 * channels, 1)
        self.ffn_out = nn.Conv2d(channels, channels, 1)
        self.fusion_scale = nn.Parameter(torch.full((1, channels, 1, 1), 0.1))
        self.ffn_scale = nn.Parameter(torch.full((1, channels, 1, 1), 0.1))

    def spatial_update(self, x: Tensor) -> Tensor:
        (first, second) = self.spatial_dw(self.spatial_in(self.spatial_norm(x))).chunk(2, dim=1)
        mixed = first * second
        return self.spatial_out(mixed * self.channel_gate(mixed))

    def ffn_update(self, x: Tensor) -> Tensor:
        (first, second) = self.ffn_in(self.ffn_norm(x)).chunk(2, dim=1)
        return self.ffn_out(first * second)


def haar_dwt(x: Tensor) -> tuple[Tensor, tuple[Tensor, Tensor, Tensor], tuple[int, int]]:
    if x.ndim != 4 or min(x.shape[-2:]) < 1 or (not x.is_floating_point()):
        raise ValueError('Haar DWT expects a nonempty floating NCHW tensor')
    (height, width) = x.shape[-2:]
    with torch.autocast(device_type=x.device.type, enabled=False):
        value = x.float()
        if height % 2 or width % 2:
            value = F.pad(value, (0, width % 2, 0, height % 2), mode='replicate')
        (a, b) = (value[..., 0::2, 0::2], value[..., 0::2, 1::2])
        (c, d) = (value[..., 1::2, 0::2], value[..., 1::2, 1::2])
        ll = (a + b + c + d) * 0.5
        lh = (-a - b + c + d) * 0.5
        hl = (-a + b - c + d) * 0.5
        hh = (a - b - c + d) * 0.5
    return (ll, (lh, hl, hh), (height, width))


def haar_idwt(ll: Tensor, high: tuple[Tensor, Tensor, Tensor], size: tuple[int, int]) -> Tensor:
    if len(high) != 3 or ll.ndim != 4 or any((h.shape != ll.shape for h in high)):
        raise ValueError('Haar IDWT expects LL and three equally shaped NCHW subbands')
    (height, width) = size
    if height < 1 or width < 1 or ((height + 1) // 2, (width + 1) // 2) != ll.shape[-2:]:
        raise ValueError('Original size does not match the Haar subband dimensions')
    with torch.autocast(device_type=ll.device.type, enabled=False):
        ll = ll.float()
        (lh, hl, hh) = (h.float() for h in high)
        a = (ll - lh - hl + hh) * 0.5
        b = (ll - lh + hl - hh) * 0.5
        c = (ll + lh - hl - hh) * 0.5
        d = (ll + lh + hl + hh) * 0.5
        (batch, channels, half_h, half_w) = ll.shape
        cells = torch.stack((a, b, c, d), dim=-1).reshape(batch, channels, half_h, half_w, 2, 2)
        value = cells.permute(0, 1, 2, 4, 3, 5).reshape(batch, channels, half_h * 2, half_w * 2)
        return value[..., :height, :width]


def check_bands(low: Tensor, high: tuple[Tensor, Tensor, Tensor], channels: int) -> None:
    if low.ndim != 4 or low.shape[1] != channels or len(high) != 3:
        raise ValueError('Expected one C-channel LL and three signed high subbands')
    if any((band.shape != low.shape for band in high)):
        raise ValueError('All four subbands must have the same NCHW shape')


class SignedDetailUpdates(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.filters = nn.ModuleList([nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False) for _ in range(3)])
        for layer in self.filters:
            nn.init.zeros_(layer.weight)

    def forward(self, high: tuple[Tensor, Tensor, Tensor]) -> tuple[Tensor, Tensor, Tensor]:
        return tuple((band.float() + layer(band.float()) for (band, layer) in zip(high, self.filters)))


class CoarseLowFrequencyAffineMixer(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.channels = channels
        self.norm = LayerNorm2d(channels)
        hidden = 2 * channels
        self.context_pw = nn.Conv2d(3 * channels, hidden, 1, bias=True)
        self.context_dw = nn.Conv2d(hidden, hidden, 5, padding=2, groups=hidden, bias=True)
        self.affine_head = nn.Conv2d(hidden, 2 * channels, 1, bias=True)
        nn.init.zeros_(self.affine_head.weight)
        nn.init.zeros_(self.affine_head.bias)
        self.high = SignedDetailUpdates(channels)

    def affine_parameters(self, low: Tensor) -> tuple[Tensor, Tensor]:
        value = low.float()
        (variance, mean) = torch.var_mean(value, dim=(-2, -1), keepdim=True, unbiased=False)
        deviation = torch.sqrt(variance + 1e-06)
        context = torch.cat((self.norm(value), mean.expand_as(value), deviation.expand_as(value)), dim=1)
        hidden = F.gelu(self.context_dw(self.context_pw(context)))
        (log_gain, bias) = self.affine_head(hidden).chunk(2, dim=1)
        gain = torch.exp(torch.tanh(log_gain))
        return (gain, bias)

    def forward(self, low: Tensor, high: tuple[Tensor, Tensor, Tensor]) -> tuple[Tensor, tuple[Tensor, Tensor, Tensor]]:
        check_bands(low, high, self.channels)
        with torch.autocast(device_type=low.device.type, enabled=False):
            (gain, bias) = self.affine_parameters(low)
            updated_low = gain * low.float() + bias
            updated_high = self.high(high)
        return (updated_low, updated_high)


class FineDetailMixer(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.channels = channels
        self.high = SignedDetailUpdates(channels)

    def forward(self, low: Tensor, high: tuple[Tensor, Tensor, Tensor]) -> tuple[Tensor, tuple[Tensor, Tensor, Tensor]]:
        check_bands(low, high, self.channels)
        with torch.autocast(device_type=low.device.type, enabled=False):
            updated_high = self.high(high)
        return (low, updated_high)


class MSWFBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.channels = channels
        self.spatial_path = SpatialGatedPath(channels)
        self.coarse_processor = CoarseLowFrequencyAffineMixer(channels)
        self.fine_processor = FineDetailMixer(channels)

    def wavelet_transform(self, x: Tensor) -> Tensor:
        (low1, high1, size1) = haar_dwt(x)
        (low2, high2, size2) = haar_dwt(low1)
        (low2, high2) = self.coarse_processor(low2, high2)
        low1 = haar_idwt(low2, high2, size2)
        (low1, high1) = self.fine_processor(low1, high1)
        return haar_idwt(low1, high1, size1)

    def forward(self, x: Tensor) -> Tensor:
        with torch.autocast(device_type=x.device.type, enabled=False):
            source = x.float()
            path = self.spatial_path
            spatial_proposal = source + path.spatial_update(source)
            joint_update = self.wavelet_transform(spatial_proposal) - source
            updated = source + path.fusion_scale * joint_update
            output = updated + path.ffn_scale * path.ffn_update(updated)
        return output.to(dtype=x.dtype)


class BottleneckFusionStages(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.blocks = nn.Sequential(MSWFBlock(channels), MSWFBlock(channels))


class LowSubbandChannelAttention(nn.Module):
    def __init__(self, channels: int, heads: int=4):
        super().__init__()
        if channels % heads:
            raise ValueError('Channels must be divisible by attention heads')
        self.heads = heads
        self.temperature = nn.Parameter(torch.ones(heads, 1, 1))
        self.qkv = nn.Conv2d(channels, 3 * channels, 1, bias=False)
        self.qkv_depthwise = nn.Conv2d(3 * channels, 3 * channels, 3, padding=1, groups=3 * channels, bias=False)
        self.output_project = nn.Conv2d(channels, channels, 1, bias=False)


class SubbandChannelGates(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.project = nn.Conv2d(channels, 3 * channels, 1, bias=True)
        nn.init.zeros_(self.project.weight)
        nn.init.zeros_(self.project.bias)

    def forward(self, low: Tensor):
        return (2.0 * torch.sigmoid(self.project(low))).chunk(3, dim=1)


class LFGSIBlock(nn.Module):
    def __init__(self, channels: int, heads: int=4):
        super().__init__()
        self.low_norm = LayerNorm2d(channels)
        self.low_attention = LowSubbandChannelAttention(channels, heads)
        (self.channels, self.heads) = (channels, heads)
        self.high_values = nn.ModuleList([nn.Sequential(nn.Conv2d(channels, channels, 1, bias=False), nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False)) for _ in range(3)])
        self.high_outputs = nn.ModuleList([nn.Conv2d(channels, channels, 1, bias=False) for _ in range(3)])
        self.high_gates = SubbandChannelGates(channels)
        self.high_scale = nn.Parameter(torch.full((3, 1, channels, 1, 1), 0.1))

    def forward(self, x: Tensor) -> Tensor:
        (low, high, size) = haar_dwt(x)
        (batch, channels, height, width) = low.shape
        shape = (batch, self.heads, channels // self.heads, height * width)
        operator = self.low_attention
        (query, key, value) = operator.qkv_depthwise(operator.qkv(self.low_norm(low))).chunk(3, dim=1)
        with torch.autocast(device_type=x.device.type, enabled=False):
            query = F.normalize(query.float().reshape(shape), dim=-1)
            key = F.normalize(key.float().reshape(shape), dim=-1)
            attention = (query @ key.transpose(-2, -1) * operator.temperature.float()).softmax(dim=-1)
            low_update = (attention @ value.float().reshape(shape)).reshape(low.shape)
        updated_low = low + operator.output_project(low_update)
        gates = self.high_gates(updated_low)
        updated_high = []
        for (index, (detail, gate, value_layer, output_layer)) in enumerate(zip(high, gates, self.high_values, self.high_outputs)):
            detail_value = value_layer(detail)
            with torch.autocast(device_type=x.device.type, enabled=False):
                mixed = (attention @ detail_value.float().reshape(shape)).reshape(detail.shape)
            updated_high.append(detail + self.high_scale[index] * gate * output_layer(mixed))
        return haar_idwt(updated_low, tuple(updated_high), size).to(dtype=x.dtype)
