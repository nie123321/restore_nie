"""Fixed Haar DWT/IWT in pure PyTorch. No wavelet library.

Subband order after DWT is [LH, HL, HH] along channels. IWT expects that order.
The transform is applied to learned features, not as a claim of lossless
signal/noise separation.
"""
from __future__ import annotations

import torch
from torch import Tensor


def haar_dwt2d(x: Tensor) -> tuple[Tensor, Tensor]:
    if x.shape[-1] % 2 or x.shape[-2] % 2:
        raise ValueError("Haar DWT requires even height and width.")
    a = x[..., 0::2, 0::2]
    b = x[..., 0::2, 1::2]
    c = x[..., 1::2, 0::2]
    d = x[..., 1::2, 1::2]
    low = (a + b + c + d) * 0.5
    lh = (-a - b + c + d) * 0.5
    hl = (-a + b - c + d) * 0.5
    hh = (a - b - c + d) * 0.5
    return low, torch.cat((lh, hl, hh), dim=1)


def haar_iwt2d(low: Tensor, high: Tensor) -> Tensor:
    channels = low.shape[1]
    if high.shape[1] != 3 * channels:
        raise ValueError("High-frequency tensor must have 3c channels [LH, HL, HH].")
    lh, hl, hh = high.split(channels, dim=1)
    top_left = (low - lh - hl + hh) * 0.5
    top_right = (low - lh + hl - hh) * 0.5
    bottom_left = (low + lh - hl - hh) * 0.5
    bottom_right = (low + lh + hl + hh) * 0.5
    height, width = low.shape[-2:]
    out = low.new_empty(low.shape[0], channels, height * 2, width * 2)
    out[..., 0::2, 0::2] = top_left
    out[..., 0::2, 1::2] = top_right
    out[..., 1::2, 0::2] = bottom_left
    out[..., 1::2, 1::2] = bottom_right
    return out
