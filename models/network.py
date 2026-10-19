"""Standalone V3; native PyTorch state loading without legacy name conversion."""
import torch
import torch.nn.functional as F
from torch import nn

from .blocks import MSWFBlock, LFGSIBlock, BottleneckFusionStages
from .checkpoint import MODEL_VARIANT, validate_model_config
from .mswf_variant import install_mswf_variant


class EnhancementNet(nn.Module):
    def __init__(self, width=24, experiment_variant=MODEL_VARIANT):
        super().__init__()
        validate_model_config({"width": width, "experiment_variant": experiment_variant})
        self.config = {"width": width, "experiment_variant": MODEL_VARIANT}
        self.stem = nn.Conv2d(3, width, 3, padding=1)
        self.encoder_full_1 = MSWFBlock(width)
        self.downsample_half = nn.Conv2d(width, 2 * width, 3, stride=2, padding=1)
        self.encoder_half_1 = MSWFBlock(2 * width)
        self.downsample_quarter = nn.Conv2d(2 * width, 4 * width, 3, stride=2, padding=1)
        self.bottleneck_fusion_in = MSWFBlock(4 * width)
        self.bottleneck_fusion_stages = BottleneckFusionStages(4 * width)
        self.decoder_half_merge = nn.Conv2d(6 * width, 2 * width, 1)
        self.decoder_half_1 = MSWFBlock(2 * width)
        self.decoder_full_merge = nn.Conv2d(3 * width, width, 1)
        self.decoder_full_1 = MSWFBlock(width)
        self.output_head = nn.Conv2d(width, 3, 3, padding=1)
        nn.init.zeros_(self.output_head.weight)
        nn.init.zeros_(self.output_head.bias)
        self.lfgsi1 = LFGSIBlock(4 * width)
        self.encoder_full_2 = MSWFBlock(width)
        self.encoder_half_2 = MSWFBlock(2 * width)
        self.decoder_half_2 = MSWFBlock(2 * width)
        self.decoder_full_2 = MSWFBlock(width)
        self.lfgsi2 = LFGSIBlock(4 * width)
        luminance = torch.tensor([0.2126, 0.7152, 0.0722])
        axis1 = F.normalize(torch.tensor([luminance[1], -luminance[0], 0.0]), dim=0)
        axis2 = F.normalize(torch.linalg.cross(luminance, axis1), dim=0)
        self.register_buffer("luminance_weights", luminance.view(1, 3, 1, 1))
        self.register_buffer("chroma_basis", torch.stack((axis1, axis2), dim=1))
        self.theta1 = nn.Parameter(torch.zeros(1, 4 * width, 1, 1))
        self.theta2 = nn.Parameter(torch.zeros(1, 4 * width, 1, 1))
        install_mswf_variant(self)

    @staticmethod
    def _scaled_interaction(x, lfgsi, theta):
        interacted = lfgsi(x)
        gamma = 2.0 * torch.sigmoid(theta)
        # Equivalent to x + gamma * (G(x) - x), with exact R3 output at gamma=1.
        return interacted + (gamma - 1.0) * (interacted - x)

    def forward(self, x):
        full = self.encoder_full_2(self.encoder_full_1(self.stem(x)))
        half = self.encoder_half_2(self.encoder_half_1(self.downsample_half(full)))
        bottom = self.downsample_quarter(half)
        bottom = self.bottleneck_fusion_in(bottom)
        bottom = self._scaled_interaction(bottom, self.lfgsi1, self.theta1)
        bottom = self.bottleneck_fusion_stages.blocks[0](bottom)
        bottom = self._scaled_interaction(bottom, self.lfgsi2, self.theta2)
        bottom = self.bottleneck_fusion_stages.blocks[1](bottom)
        up = F.interpolate(bottom, size=half.shape[-2:], mode="bilinear", align_corners=False)
        feature = self.decoder_half_merge(torch.cat((up, half), dim=1))
        feature = self.decoder_half_2(self.decoder_half_1(feature))
        up = F.interpolate(feature, size=full.shape[-2:], mode="bilinear", align_corners=False)
        feature = self.decoder_full_merge(torch.cat((up, full), dim=1))
        feature = self.decoder_full_2(self.decoder_full_1(feature))
        return x + self.output_head(feature).float()


def load_checkpoint(path, device="cpu"):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model_config = checkpoint["config"]["model"]
    validate_model_config(model_config)
    model = EnhancementNet(**model_config)
    model.load_state_dict(checkpoint["model"], strict=True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    return model.to(device=device, dtype=torch.float32).eval(), checkpoint
