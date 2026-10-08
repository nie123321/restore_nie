"""Normalized Restormer MDTA residual, adapted for A; no additional GDFN.
Reference: Zamir et al., CVPR 2022; swz30/Restormer restormer_arch.py.
"""
import torch
from torch import nn
import torch.nn.functional as F

class MDTAAttention(nn.Module):
    def __init__(self, channels, heads=4):
        super().__init__()
        if channels % heads:
            raise ValueError('Channels must be divisible by heads')
        self.heads = heads
        self.temperature = nn.Parameter(torch.ones(heads, 1, 1))
        self.qkv = nn.Conv2d(channels, 3*channels, 1, bias=False)
        self.qkv_dwconv = nn.Conv2d(3*channels, 3*channels, 3, padding=1, groups=3*channels, bias=False)
        self.project_out = nn.Conv2d(channels, channels, 1, bias=False)
    def forward(self, x, illumination=None):
        b,c,h,w = x.shape
        q,k,v = self.qkv_dwconv(self.qkv(x)).chunk(3, dim=1)
        if illumination is not None:
            from illumination_blocks import guided_value
            v = guided_value(v, illumination)
        shape = (b,self.heads,c//self.heads,h*w)
        q,k,v = (t.reshape(shape) for t in (q,k,v))
        q,k = F.normalize(q,dim=-1), F.normalize(k,dim=-1)
        attn = ((q @ k.transpose(-2,-1))*self.temperature).softmax(dim=-1)
        return self.project_out((attn @ v).reshape(b,c,h,w))

class MDTAResidual(nn.Module):
    def __init__(self, channels, norm_layer, heads=4):
        super().__init__()
        self.norm = norm_layer(channels)
        self.attn = MDTAAttention(channels,heads)
    def forward(self,x,illumination=None):
        return x + self.attn(self.norm(x), illumination=illumination)
