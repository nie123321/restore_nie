"""Strict model/checkpoint API for the experiment identified by model.json."""
from __future__ import annotations

import inspect
import json
from pathlib import Path
import sys
from typing import BinaryIO

import torch

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
CORE_ROOT = Path(__file__).resolve().parent / "core"
if str(CORE_ROOT) not in sys.path:
    sys.path.insert(0, str(CORE_ROOT))
from model import EnhancementDemo as ReferenceModel
from models.baseline_a_blocks import install_eleven_own_blocks
from models.experiment_blocks import install_joint_fusion
from models.candidate_blocks import install_candidate

MODEL_SPEC = json.loads((PACKAGE_ROOT/"configs"/"model.json").read_text(encoding="utf-8"))
BASE_MODEL_CONFIG = {
    "width": 24, "spectral_mode": "off", "output_mode": "direct",
    "wavelet_variant": "all-subband-first-mdta", "star_refinement": "multiscale",
    "bottleneck_attention": "mdta",
}
FINAL_MODEL_CONFIG = {**BASE_MODEL_CONFIG, "experiment_variant": MODEL_SPEC["variant"]}


def validate_model_config(config: dict) -> None:
    defaults = {name: value.default for name, value in
                inspect.signature(ReferenceModel.__init__).parameters.items() if name != "self"}
    defaults["experiment_variant"] = MODEL_SPEC["variant"]
    unexpected = set(config) - set(defaults)
    if unexpected:
        raise ValueError(f"Unknown model configuration keys: {sorted(unexpected)}")
    # Existing D checkpoints do not identify these experiments and cannot be resumed.
    if config.get("experiment_variant") != MODEL_SPEC["variant"]:
        raise ValueError("Checkpoint/config does not belong to this experiment folder")
    expected = {**defaults, **FINAL_MODEL_CONFIG}
    actual = {**defaults, **config}
    differences = {key: (actual[key], value) for key, value in expected.items() if actual[key] != value}
    if differences:
        raise ValueError(f"Experiment architecture mismatch: {differences}")


class EnhancementDemo(ReferenceModel):
    def __init__(self, **config):
        config = {**FINAL_MODEL_CONFIG, **config}
        validate_model_config(config)
        variant = config.pop("experiment_variant")
        # Construct the entire frozen reference first, then replace only intended modules.
        super().__init__(**config)
        install_eleven_own_blocks(self)
        install_joint_fusion(self, MODEL_SPEC["recipe"]["fusion_mode"])
        install_candidate(self, MODEL_SPEC["recipe"])
        self.config["experiment_variant"] = variant


def build_model() -> EnhancementDemo:
    """Leave seed control to the caller; no parameter accounting."""
    return EnhancementDemo(**FINAL_MODEL_CONFIG)


def load_checkpoint(path: str | Path | BinaryIO, device="cpu"):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    validate_model_config(checkpoint["config"]["model"])
    model = EnhancementDemo(**checkpoint["config"]["model"])
    model.load_state_dict(checkpoint["model"], strict=True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    return model.to(device=device, dtype=torch.float32).eval(), checkpoint
