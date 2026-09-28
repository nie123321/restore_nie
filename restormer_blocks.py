"""Restormer TransformerBlock adapted to A's NCHW feature tensors.

MDTA + GDFN, expansion 2.66, bias-free convolutions and WithBias LayerNorm
(eps=1e-5), following swz30/Restormer/basicsr/models/archs/restormer_arch.py.
The attention core is shared with the existing bottom-MDTA experiment.

MIT License
Copyright (c) 2022 Syed Waqas Zamir and contributors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""
import torch
from torch import nn
import torch.nn.functional as F

from mdta_blocks import MDTAAttention


class RestormerLayerNorm2d(nn.Module):
    """Official WithBias LayerNorm across channels, expressed directly in NCHW."""
    def __init__(self, channels):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x):
        mean = x.mean(dim=1, keepdim=True)
        variance = x.var(dim=1, keepdim=True, unbiased=False)
        return (x - mean) / torch.sqrt(variance + 1e-5) * self.weight + self.bias


class GDFN(nn.Module):
    def __init__(self, channels, expansion=2.66):
        super().__init__()
        hidden = int(channels * expansion)
        self.project_in = nn.Conv2d(channels, 2 * hidden, 1, bias=False)
        self.dwconv = nn.Conv2d(2 * hidden, 2 * hidden, 3, padding=1,
                                groups=2 * hidden, bias=False)
        self.project_out = nn.Conv2d(hidden, channels, 1, bias=False)

    def forward(self, x):
        left, right = self.dwconv(self.project_in(x)).chunk(2, dim=1)
        return self.project_out(F.gelu(left) * right)


class RestormerBlock(nn.Module):
    def __init__(self, channels, heads=2, expansion=2.66):
        super().__init__()
        self.norm1 = RestormerLayerNorm2d(channels)
        self.attn = MDTAAttention(channels, heads)
        self.norm2 = RestormerLayerNorm2d(channels)
        self.ffn = GDFN(channels, expansion)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        return x + self.ffn(self.norm2(x))
