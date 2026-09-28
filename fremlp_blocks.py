"""DarkIR amplitude MLP and its multiplicative residual modulation.

Adapted from https://github.com/cidautai/DarkIR/blob/main/archs/arch_model.py.
FFT, amplitude MLP, normalization and modulation run in FP32 under AMP.

MIT License
Copyright (c) 2025 cidautai

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
import torch
from torch import nn


class FreMLP(nn.Module):
    def __init__(self, channels: int, expand: int = 2):
        super().__init__()
        self.process1 = nn.Sequential(
            nn.Conv2d(channels, expand * channels, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(expand * channels, channels, 1),
        )

    def forward(self, x):
        with torch.autocast(device_type=x.device.type, enabled=False):
            spectrum = torch.fft.rfft2(x.float(), norm='backward')
            magnitude = self.process1(spectrum.abs())
            phase = torch.angle(spectrum)
            spectrum = torch.complex(magnitude * phase.cos(), magnitude * phase.sin())
            return torch.fft.irfft2(spectrum, s=x.shape[-2:], norm='backward')


class FreMLPResidual(nn.Module):
    def __init__(self, channels: int, norm_layer):
        super().__init__()
        self.norm = norm_layer(channels)
        self.freq = FreMLP(channels, expand=2)
        self.gamma = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x):
        with torch.autocast(device_type=x.device.type, enabled=False):
            feature = x.float()
            modulation = self.freq(self.norm(feature))
            output = feature + self.gamma * (feature * modulation)
        return output.to(dtype=x.dtype)
