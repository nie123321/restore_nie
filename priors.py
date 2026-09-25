"""Fixed illumination and CIConv-W structure priors for Prior-A.

Luminance channels follow Multinex IlluminationExtractor, except the L2
channel is divided by sqrt(3). CIConv-W follows the discrete Gaussian and
derivative normalization used by SG-LLIE / CIConv (k=3, scale=0.9 as an
exponent, sigma=2**0.9). This is an RGB float adaptation: it does not copy
the official BGR cv2.imread path or the truncated PNG export.
Source: https://github.com/minyan8/imagine/blob/0b96263167b7511bede010537b64f8b44ba240a4/Enhancement/test/ciconv2d0.py
The upstream CIConv2d header credits Attila Lengyel; see the upstream file
for its original authorship and repository licensing information.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn


LUMA_NAMES = ("mean", "rec709", "vmax", "lightness", "ycgco", "l2norm_scaled")
EPS_L = 1e-6
EPS_W = 1e-5
GCM = (
    (0.06, 0.63, 0.27),
    (0.30, 0.04, -0.35),
    (0.34, -0.60, 0.17),
)


def six_luminance(rgb: Tensor) -> Tensor:
    """rgb [N,3,H,W] in encoded [0,1] -> [N,6,H,W] FP32."""
    red, green, blue = rgb[:, 0:1], rgb[:, 1:2], rgb[:, 2:3]
    mean = (red + green + blue) / 3.0
    rec709 = 0.2126 * red + 0.7152 * green + 0.0722 * blue
    vmax = torch.maximum(torch.maximum(red, green), blue)
    vmin = torch.minimum(torch.minimum(red, green), blue)
    lightness = (vmax + vmin) / 2.0
    ycgco = 0.25 * red + 0.5 * green + 0.25 * blue
    l2norm = torch.sqrt(red.square() + green.square() + blue.square() + EPS_L) / math.sqrt(3.0)
    return torch.cat((mean, rec709, vmax, lightness, ycgco, l2norm), dim=1)


def _ciconv_kernels(scale: float = 0.9, k: int = 3) -> tuple[Tensor, Tensor, Tensor]:
    sigma = 2.0 ** scale
    radius = math.ceil(k * sigma + 0.5)
    coords = torch.arange(-radius, radius + 1, dtype=torch.float32)
    yy, xx = torch.meshgrid(coords, coords, indexing="ij")
    gaussian = torch.exp(-(xx.square() + yy.square()) / (2.0 * sigma * sigma))
    gaussian = gaussian / gaussian.sum()
    dx = (-xx / (sigma * sigma)) * gaussian
    dy = (-yy / (sigma * sigma)) * gaussian
    dx = dx / dx.abs().sum()
    dy = dy / dy.abs().sum()
    return gaussian, dx, dy


class CIConvW(nn.Module):
    """Fixed RGB CIConv-W. No trainable parameters."""

    def __init__(self, scale: float = 0.9, k: int = 3):
        super().__init__()
        gaussian, dx, dy = _ciconv_kernels(scale, k)
        self.scale = scale
        self.k = k
        self.register_buffer("gaussian", gaussian[None, None])
        self.register_buffer("dx", dx[None, None])
        self.register_buffer("dy", dy[None, None])
        self.register_buffer("gcm", torch.tensor(GCM, dtype=torch.float32))

    def _conv(self, image: Tensor, kernel: Tensor) -> Tensor:
        radius = kernel.shape[-1] // 2
        return F.conv2d(image, kernel, padding=radius)

    def forward(self, rgb: Tensor) -> Tensor:
        device_type = rgb.device.type if rgb.device.type in {"cpu", "cuda"} else "cpu"
        with torch.amp.autocast(device_type=device_type, enabled=False):
            value = rgb.float()
            gcm = torch.einsum("ij,bjhw->bihw", self.gcm, value)
            energy = self._conv(gcm[:, 0:1], self.gaussian)
            terms = []
            for channel in range(3):
                plane = gcm[:, channel:channel + 1]
                terms.append(self._conv(plane, self.dx) / (energy + EPS_W))
                terms.append(self._conv(plane, self.dy) / (energy + EPS_W))
            response = torch.stack(terms, dim=0).square().sum(dim=0)
            log_response = torch.log(response + EPS_W)
            mean = log_response.mean(dim=(-2, -1), keepdim=True)
            var = log_response.var(dim=(-2, -1), unbiased=False, keepdim=True)
            return (log_response - mean) / torch.sqrt(var + EPS_W)


class PriorAdapter(nn.Module):
    def __init__(self, in_channels: int, mid: int = 8, out_channels: int = 96):
        super().__init__()
        self.project_in = nn.Conv2d(in_channels, mid, 1)
        self.depthwise = nn.Conv2d(mid, mid, 3, padding=1, groups=mid)
        self.project_out = nn.Conv2d(mid, out_channels, 1)
        nn.init.zeros_(self.project_out.weight)
        nn.init.zeros_(self.project_out.bias)

    def forward(self, prior: Tensor) -> Tensor:
        hidden = F.gelu(self.project_in(prior))
        hidden = F.gelu(self.depthwise(hidden))
        return self.project_out(hidden)
