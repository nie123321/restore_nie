import torch
from torch import nn


class JointSubbandResidual(nn.Module):
    """Per-band normalization, then local cross-band/channel residual mixing."""
    def __init__(self, channels, norm_class):
        super().__init__()
        self.norms = nn.ModuleList([norm_class(channels) for _ in range(4)])
        hidden = channels // 2
        self.input_project = nn.Conv2d(4 * channels, hidden, 1, bias=True)
        self.local_filter = nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden, bias=True)
        self.activation = nn.GELU()
        self.output_project = nn.Conv2d(hidden, 4 * channels, 1, bias=True)
        nn.init.zeros_(self.output_project.weight)
        nn.init.zeros_(self.output_project.bias)

    def forward(self, low, high):
        bands = (low, *high)
        normalized = torch.cat([norm(band) for norm, band in zip(self.norms, bands)], dim=1)
        hidden = self.activation(self.local_filter(self.input_project(normalized)))
        deltas = self.output_project(hidden).chunk(4, dim=1)
        updated = tuple(band + delta for band, delta in zip(bands, deltas))
        return updated[0], updated[1:]


def install_mswf_variant(model):
    # Construct original V3 first so all unchanged tensors retain its seed stream.
    sites = [block for block in model.modules() if type(block).__name__ == 'MSWFBlock']
    if len(sites) != 11:
        raise ValueError('Expected eleven MSWF instances')
    for block in sites:
        channels = block.channels
        norm_class = type(block.spatial_path.spatial_norm)
        block.coarse_processor = JointSubbandResidual(channels, norm_class)
        block.fine_processor = JointSubbandResidual(channels, norm_class)
