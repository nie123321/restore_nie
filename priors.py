"""Fixed priors computed only from the augmented low-light RGB input.

CIConv-W is an RGB-float adaptation of the Gaussian/GCM calculation in
endo_prior_fusion_u3/priors.py, based on the CIConv implementation credited
to Attila Lengyel in SG-LLIE:
https://github.com/minyan8/imagine/blob/0b96263167b7511bede010537b64f8b44ba240a4/Enhancement/test/ciconv2d0.py
Scale in that interface means log2(sigma), not sigma. Replicate padding here
avoids manufacturing image-border edges; fixed scales are 1, 2**0.9, 3 px.
"""
from __future__ import annotations

import math
import torch
from torch import Tensor, nn
import torch.nn.functional as F

LUMA_NAMES = ("mean", "rec709", "vmax", "lightness", "ycgco", "l2norm_scaled")
GCM = ((0.06, 0.63, 0.27), (0.30, 0.04, -0.35), (0.34, -0.60, 0.17))


def rec709(rgb: Tensor) -> Tensor:
    return 0.2126 * rgb[:, 0:1] + 0.7152 * rgb[:, 1:2] + 0.0722 * rgb[:, 2:3]


def six_luminance(rgb: Tensor) -> Tensor:
    r, g, b = rgb[:, 0:1], rgb[:, 1:2], rgb[:, 2:3]
    maximum = torch.maximum(torch.maximum(r, g), b)
    minimum = torch.minimum(torch.minimum(r, g), b)
    return torch.cat(((r + g + b) / 3, rec709(rgb), maximum,
                      (maximum + minimum) / 2, .25 * r + .5 * g + .25 * b,
                      torch.sqrt(r.square() + g.square() + b.square() + 1e-6)
                      / math.sqrt(3)), dim=1)


def gaussian_kernels(sigma: float, k: int = 3):
    radius = math.ceil(k * sigma + .5)
    coords = torch.arange(-radius, radius + 1, dtype=torch.float32)
    yy, xx = torch.meshgrid(coords, coords, indexing="ij")
    gaussian = torch.exp(-(xx.square() + yy.square()) / (2 * sigma * sigma))
    gaussian /= gaussian.sum()
    # Positive response for a rising left-to-right / top-to-bottom image,
    # because torch conv2d uses correlation rather than kernel reversal.
    dx, dy = xx * gaussian / sigma**2, yy * gaussian / sigma**2
    dx /= dx.abs().sum()
    dy /= dy.abs().sum()
    return tuple(t[None, None] for t in (gaussian, dx, dy))


def fixed_conv(image: Tensor, kernel: Tensor) -> Tensor:
    radius = kernel.shape[-1] // 2
    channels = image.shape[1]
    return F.conv2d(F.pad(image, (radius,) * 4, mode="replicate"),
                    kernel.expand(channels, 1, -1, -1), groups=channels)


class CIConvW(nn.Module):
    def __init__(self, scale: float):
        super().__init__()
        self.scale, self.sigma = scale, 2**scale
        for name, kernel in zip(("gaussian", "dx", "dy"), gaussian_kernels(self.sigma)):
            self.register_buffer(name, kernel)
        self.register_buffer("gcm", torch.tensor(GCM, dtype=torch.float32))

    def forward(self, rgb: Tensor) -> Tensor:
        value = torch.einsum("ij,bjhw->bihw", self.gcm, rgb)
        energy = fixed_conv(value[:, :1], self.gaussian)
        gx = fixed_conv(value, self.dx) / (energy + 1e-5)
        gy = fixed_conv(value, self.dy) / (energy + 1e-5)
        response = torch.log((gx.square() + gy.square()).sum(1, keepdim=True) + 1e-5)
        mean = response.mean((-2, -1), keepdim=True)
        variance = response.var((-2, -1), unbiased=False, keepdim=True)
        return (response - mean) * torch.rsqrt(variance + 1e-5)


class FixedPriors(nn.Module):
    def __init__(self):
        super().__init__()
        self.ciconv = nn.ModuleList(CIConvW(math.log2(s)) for s in (1., 2**.9, 3.))
        _, dx, dy = gaussian_kernels(.9)
        self.register_buffer("dx", dx)
        self.register_buffer("dy", dy)

    def signed_gradients(self, rgb: Tensor) -> Tensor:
        luminance = rec709(rgb)
        gradient = torch.cat((fixed_conv(luminance, self.dx),
                              fixed_conv(luminance, self.dy)), dim=1)
        # One shared per-image 95th-percentile absolute scale for both axes.
        # 0.03 in encoded [0,1] units prevents dark noise (~1e-3) becoming a
        # unit-strength structure. tanh bounds outliers without losing sign.
        scale = torch.quantile(gradient.detach().abs().flatten(1), .95, dim=1)
        scale = scale.clamp_min(.03).view(-1, 1, 1, 1)
        return torch.tanh(gradient / scale)

    @torch.no_grad()
    def forward(self, low: Tensor) -> tuple[Tensor, Tensor]:
        with torch.amp.autocast(low.device.type, enabled=False):
            low = low.float()
            luminance = six_luminance(low)
            structure = torch.cat(tuple(op(low) for op in self.ciconv)
                                  + (self.signed_gradients(low),), dim=1)
        return luminance, structure
