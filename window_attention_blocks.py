"""Attention-only W-MSA/SW-MSA pair for A's bottleneck; no extra FFN.

Uses SwinIR's window/shift attention design with relative position bias:
https://github.com/JingyunLiang/SwinIR/blob/main/models/network_swinir.py
Includes a pre-norm and a small learned channel residual scale for each layer.
"""
from __future__ import annotations

import torch
from torch import nn


def partition_windows(value, window_size):
    """BHWC -> (B * windows, window_size**2, C), without changing pixel order."""
    batch, height, width, channels = value.shape
    if height % window_size or width % window_size:
        raise ValueError("Window attention requires feature dimensions divisible by window size")
    return (value.reshape(batch, height // window_size, window_size,
                          width // window_size, window_size, channels)
            .permute(0, 1, 3, 2, 4, 5).reshape(-1, window_size**2, channels))


def merge_windows(windows, window_size, batch, height, width):
    channels = windows.shape[-1]
    return (windows.reshape(batch, height // window_size, width // window_size,
                            window_size, window_size, channels)
            .permute(0, 1, 3, 2, 4, 5).reshape(batch, height, width, channels))


class WindowAttentionResidual(nn.Module):
    def __init__(self, channels, heads=4, window_size=8, shift=0, initial_scale=0.1):
        super().__init__()
        if channels % heads or not 0 <= shift < window_size:
            raise ValueError("Channels must be divisible by heads and shift must be inside window")
        self.channels, self.heads = channels, heads
        self.window_size, self.shift = window_size, shift
        self.qk_scale = (channels // heads) ** -0.5
        self.norm = nn.LayerNorm(channels)
        self.qkv = nn.Linear(channels, 3 * channels)
        self.projection = nn.Linear(channels, channels)
        self.relative_bias = nn.Parameter(torch.empty((2 * window_size - 1)**2, heads))
        nn.init.trunc_normal_(self.relative_bias, std=0.02)
        yy, xx = torch.meshgrid(torch.arange(window_size), torch.arange(window_size), indexing="ij")
        positions = torch.stack((yy.flatten(), xx.flatten()))
        offsets = positions[:, :, None] - positions[:, None, :]
        index = (offsets[0] + window_size - 1) * (2 * window_size - 1)
        index = index + offsets[1] + window_size - 1
        self.register_buffer("relative_index", index.long())
        self.residual_scale = nn.Parameter(torch.full((1, 1, 1, channels), initial_scale))
        self._mask_cache = {}

    def shift_mask(self, height, width, device):
        """Block cyclic wraparound; neighboring interior windows may communicate."""
        if self.shift == 0:
            return None
        key = (height, width, device)
        if key not in self._mask_cache:
            rows = torch.arange(height, device=device)
            columns = torch.arange(width, device=device)
            row_region = ((rows >= height - self.window_size).long()
                          + (rows >= height - self.shift).long())
            column_region = ((columns >= width - self.window_size).long()
                             + (columns >= width - self.shift).long())
            regions = (3 * row_region[:, None] + column_region[None, :])[None, ..., None]
            labels = partition_windows(regions, self.window_size).squeeze(-1)
            blocked = labels[:, :, None] != labels[:, None, :]
            self._mask_cache[key] = torch.zeros_like(blocked, dtype=torch.float32).masked_fill(blocked, -100.0)
        return self._mask_cache[key]

    def attention(self, windows, mask=None):
        count, tokens, channels = windows.shape
        qkv = (self.qkv(windows).reshape(count, tokens, 3, self.heads, channels // self.heads)
               .permute(2, 0, 3, 1, 4))
        query, key, value = qkv.unbind(0)
        scores = (query * self.qk_scale) @ key.transpose(-2, -1)
        bias = self.relative_bias[self.relative_index.flatten()]
        bias = bias.reshape(tokens, tokens, self.heads).permute(2, 0, 1)
        scores = scores.float() + bias.float()[None]
        if mask is not None:
            per_image = mask.shape[0]
            scores = (scores.reshape(-1, per_image, self.heads, tokens, tokens)
                      + mask[None, :, None]).reshape(count, self.heads, tokens, tokens)
        weights = scores.softmax(dim=-1).to(value.dtype)
        result = (weights @ value).transpose(1, 2).reshape(count, tokens, channels)
        return self.projection(result)

    def forward(self, feature):
        batch, channels, height, width = feature.shape
        if channels != self.channels:
            raise ValueError("Feature channels differ from window attention configuration")
        shortcut = feature.permute(0, 2, 3, 1)
        normalized = self.norm(shortcut)
        if self.shift:
            normalized = torch.roll(normalized, (-self.shift, -self.shift), (1, 2))
        windows = partition_windows(normalized, self.window_size)
        windows = self.attention(windows, self.shift_mask(height, width, feature.device))
        result = merge_windows(windows, self.window_size, batch, height, width)
        if self.shift:
            result = torch.roll(result, (self.shift, self.shift), (1, 2))
        return (shortcut + self.residual_scale * result).permute(0, 3, 1, 2).contiguous()


class ShiftedWindowPair(nn.Module):
    def __init__(self, channels, heads=4, window_size=8):
        super().__init__()
        self.regular = WindowAttentionResidual(channels, heads, window_size, shift=0)
        self.shifted = WindowAttentionResidual(channels, heads, window_size, shift=window_size // 2)

    def forward(self, feature):
        return self.shifted(self.regular(feature))
