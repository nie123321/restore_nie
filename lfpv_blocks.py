"""Full LFPV core adapted from Xiaogang Xu et al., ICCV 2025.

Reference: https://github.com/xiaogang00/LFPVS_ICCV
The public implementation is models/archs/low_light_transformer.py.
Retains its 16 vector/patch references, 4x4 patches, identity embedding,
seven-convolution SU/MU networks, normalization and vector/patch queries.
Defaults retain the published 64-channel, 4x-hidden updater layout.
The optional 32-channel, 2x-hidden variant changes widths only.
The A adaptation adds bottleneck<->prior projections and explicitly commits reference
state only after a successful optimizer step. Reference state is persistent
buffers, not ordinary optimizer parameters (the upstream require_grad typo
does not freeze its Parameters). Evaluation never updates references.
"""
from __future__ import annotations

import random

import torch
from torch import Tensor, nn
import torch.nn.functional as F


def _updater(channels: int, inputs: int, kernel: int, expansion: int = 4) -> nn.Sequential:
    # Preserve seven convolutions and BN/ReLU placement; optionally narrow hidden channels.
    widths = [expansion * channels] * 5 + [channels, channels]
    layers = []
    previous = inputs * channels
    for index, width in enumerate(widths):
        layers.append(nn.Conv2d(previous, width, kernel, padding=kernel // 2))
        if index < len(widths) - 1:
            layers.extend([nn.BatchNorm2d(width), nn.ReLU()])
        previous = width
    return nn.Sequential(*layers)


class LFPVCore(nn.Module):
    def __init__(self, channels: int = 64, references: int = 16, patch_size: int = 4,
                 updater_expansion: int = 4):
        super().__init__()
        if min(channels, references, patch_size, updater_expansion) < 1:
            raise ValueError("LFPV widths, references and patch size must be positive")
        self.channel = channels
        self.num_feature = references
        self.patch_size = patch_size
        self.register_buffer("common_feature", torch.zeros(references, channels))
        self.register_buffer("common_feature_patch", torch.zeros(references, channels, patch_size, patch_size))
        self.embedding = nn.Embedding(references, channels)
        self.update_feature = _updater(channels, 3, 1, updater_expansion)
        self.update_feature_patch = _updater(channels, 3, 3, updater_expansion)
        self.propagate = _updater(channels, 4, 1, updater_expansion)
        self.propagate_patch = _updater(channels, 4, 3, updater_expansion)
        self.feature_unshuffle = nn.PixelUnshuffle(patch_size)
        self.feature_shuffle = nn.PixelShuffle(patch_size)
        self.norm_query1 = nn.LayerNorm(channels)
        self.norm_query2 = nn.LayerNorm(channels)
        self.norm_query3 = nn.LayerNorm(channels * patch_size * patch_size)
        self.norm_query4 = nn.LayerNorm(channels * patch_size * patch_size)
        self.norm_feature = nn.LayerNorm(channels)
        self.norm_patch = nn.LayerNorm([channels, patch_size, patch_size])
        self.norm_feature2 = nn.LayerNorm(channels)
        self.norm_patch2 = nn.LayerNorm([channels, patch_size, patch_size])

    def query_and_fuse(self, feature: Tensor, vectors: Tensor, patches: Tensor) -> Tensor:
        b, c, h, w = feature.shape
        k, count = self.patch_size, self.num_feature
        # Whole 224x448 -> 56x112 is already divisible by four. Padding makes
        # inference on other image dimensions possible without resizing inputs.
        ph, pw = (-h) % k, (-w) % k
        value = F.pad(feature, (0, pw, 0, ph), mode="replicate") if ph or pw else feature
        hp, wp = value.shape[-2:]
        qv = self.norm_query1(value.flatten(2).transpose(1, 2))
        kv = self.norm_query2(vectors).transpose(0, 1)
        av = (qv @ kv).softmax(dim=-1)
        delta_v = (av @ vectors).transpose(1, 2).reshape(b, c, hp, wp)
        qp = self.feature_unshuffle(value).flatten(2).transpose(1, 2)
        qp = self.norm_query3(qp)
        flat_patches = patches.reshape(count, c * k * k)
        kp = self.norm_query4(flat_patches).transpose(0, 1)
        ap = (qp @ kp).softmax(dim=-1)
        delta_p = (ap @ flat_patches).transpose(1, 2).reshape(b, c * k * k, hp // k, wp // k)
        delta_p = self.feature_shuffle(delta_p)
        return (value + delta_v + delta_p)[..., :h, :w]

    def sample_update(self, feature: Tensor) -> tuple[Tensor, Tensor]:
        b, c, h, w = feature.shape
        k, count = self.patch_size, self.num_feature
        if min(h, w) < k:
            feature = F.pad(feature, (0, max(0, k - w), 0, max(0, k - h)), mode="replicate")
            h, w = feature.shape[-2:]
        # Python RNG follows the upstream sampling and is saved by our runner.
        y, x = random.randint(0, h - 1), random.randint(0, w - 1)
        py, px = random.randint(0, h - k), random.randint(0, w - k)
        fv = feature[:, :, y, x]
        fp = feature[:, :, py:py + k, px:px + k]
        ids = torch.arange(count, device=feature.device)
        embedded = self.embedding(ids)[:, None].expand(count, b, c)
        old_v = self.common_feature.detach().clone()[:, None].expand(count, b, c)
        selected_v = fv[None].expand(count, b, c)
        inputs_v = torch.cat((old_v, selected_v, embedded), dim=2).reshape(count * b, 3 * c, 1, 1)
        v = self.update_feature(inputs_v).reshape(count, b, c).mean(dim=1)
        v = self.norm_feature(v)
        old_p = self.common_feature_patch.detach().clone()[:, None].expand(count, b, c, k, k)
        selected_p = fp[None].expand(count, b, c, k, k)
        ep = embedded[..., None, None].expand(count, b, c, k, k)
        inputs_p = torch.cat((old_p, selected_p, ep), dim=2).reshape(count * b, 3 * c, k, k)
        p = self.update_feature_patch(inputs_p).reshape(count, b, c, k, k).mean(dim=1)
        return v, self.norm_patch(p)

    def mutual_update(self, vectors: Tensor, patches: Tensor) -> tuple[Tensor, Tensor]:
        count, c, k = self.num_feature, self.channel, self.patch_size
        peers_v = [random.randint(0, count - 1) for _ in range(count)]
        peers_p = [random.randint(0, count - 1) for _ in range(count)]
        ids = torch.arange(count, device=vectors.device)
        target_e = self.embedding(ids)
        peer_e = self.embedding(torch.tensor(peers_v, device=vectors.device))
        iv = torch.cat((vectors, vectors[peers_v], target_e, peer_e), dim=1)
        v = self.propagate(iv.reshape(count, 4 * c, 1, 1)).reshape(count, c)
        v = self.norm_feature2(v)
        # Preserve the published code's embedding pairing for the patch updater.
        ep = target_e[..., None, None].expand(count, c, k, k)
        eh = peer_e[..., None, None].expand(count, c, k, k)
        ip = torch.cat((patches, patches[peers_p], ep, eh), dim=1)
        p = self.norm_patch2(self.propagate_patch(ip))
        return v, p

    def query(self, feature: Tensor) -> Tensor:
        with torch.autocast(device_type=feature.device.type, enabled=False):
            return self.query_and_fuse(feature.float(), self.common_feature, self.common_feature_patch)

    def update_and_query(self, feature: Tensor) -> tuple[Tensor, dict[str, Tensor]]:
        if not self.training:
            raise RuntimeError("LFPV reference updates are training-only")
        with torch.autocast(device_type=feature.device.type, enabled=False):
            value = feature.float()
            vectors, patches = self.sample_update(value)
            vectors, patches = self.mutual_update(vectors, patches)
            refined = self.query_and_fuse(value, vectors, patches)
        return refined, {"vectors": vectors.detach(), "patches": patches.detach()}

    @torch.no_grad()
    def commit(self, update: dict[str, Tensor]) -> None:
        vectors, patches = update["vectors"], update["patches"]
        if not torch.isfinite(vectors).all() or not torch.isfinite(patches).all():
            raise RuntimeError("Non-finite LFPV reference update")
        self.common_feature.copy_(vectors)
        self.common_feature_patch.copy_(patches)


class BottomLFPV(nn.Module):
    """A feature F + project_out(LFPV(project_in(F)) - project_in(F))."""
    def __init__(self, channels: int, prior_channels: int = 64, updater_expansion: int = 4):
        super().__init__()
        self.input_projection = nn.Conv2d(channels, prior_channels, 1)
        self.core = LFPVCore(channels=prior_channels, references=16, patch_size=4,
                             updater_expansion=updater_expansion)
        self.output_projection = nn.Conv2d(prior_channels, channels, 1)
        nn.init.zeros_(self.output_projection.bias)

    def forward(self, feature: Tensor) -> Tensor:
        mapped = self.input_projection(feature)
        refined = self.core.query(mapped)
        correction = self.output_projection((refined - mapped.float()).to(mapped.dtype))
        return feature + correction

    def update_and_query(self, feature: Tensor) -> tuple[Tensor, dict[str, Tensor]]:
        mapped = self.input_projection(feature)
        refined, update = self.core.update_and_query(mapped)
        correction = self.output_projection((refined - mapped.float()).to(mapped.dtype))
        return feature + correction, update

    def commit(self, update: dict[str, Tensor]) -> None:
        self.core.commit(update)
