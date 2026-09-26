"""Small shared training primitives, strict state checks and evaluation."""
from __future__ import annotations
from datetime import datetime
import json
import math
import os
from pathlib import Path
import random
import tempfile
import numpy as np
import torch
import torch.nn.functional as F
from .model import ARCH_VERSION, PriorAU3

HERE = Path(__file__).resolve().parent


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def rng_state():
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"] is not None:
        if not torch.cuda.is_available():
            raise ValueError("CUDA checkpoint RNG cannot be restored without CUDA")
        torch.cuda.set_rng_state_all(state["cuda"])


def compute_loss(output, target):
    """Final-output raw RGB L1; no auxiliary image correction loss."""
    return F.l1_loss(output.float(), target.float())


def learning_rate(config, step):
    progress = (step - 1) / max(1, config["steps"] - 1)
    return config["min_lr"] + .5 * (config["lr"] - config["min_lr"]) * (1 + math.cos(math.pi * progress))


def ensure_run_location(path):
    path = Path(path).resolve()
    runs = (HERE / "runs").resolve()
    if runs not in path.parents:
        raise ValueError(f"Run must be a child of the independent version's runs directory: {runs}")
    return path


def atomic_save(state, path):
    path = Path(path)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    os.close(fd)
    try:
        torch.save(state, temporary)
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def write_json(path, data):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def write_status(run, state, step, target, **details):
    write_json(Path(run) / "status.json", dict(state=state, step=step, target=target,
               pid=os.getpid(), updated_at=datetime.now().isoformat(), **details))


def load_checkpoint(path):
    state = torch.load(path, map_location="cpu", weights_only=False)
    required = {"model", "optimizer", "scaler", "config", "step", "rng", "sampler", "filenames"}
    if required - state.keys() or state["config"].get("arch_version") != ARCH_VERSION:
        raise ValueError("Not a complete Prior-A U3 checkpoint")
    if not isinstance(state["step"], int) or not 0 <= state["step"] <= state["config"]["steps"]:
        raise ValueError("Invalid checkpoint global step")
    return state


def check_resume(state, config, sampler, filenames):
    # Runtime logging / worker count are not training semantics; every saved
    # architecture, augmentation, optimizer, loss and LR-horizon field is.
    for key in config:
        if key not in {"workers", "log_every", "device"} and state["config"].get(key) != config[key]:
            raise ValueError(f"Resume configuration differs for {key}")
    if set(state["config"]) != set(config):
        raise ValueError("Checkpoint configuration schema differs")
    if state["sampler"] != sampler.state(state["step"]):
        raise ValueError("Checkpoint sampler cursor/configuration is inconsistent")
    if state["filenames"] != filenames:
        raise ValueError("Manifest filename order differs from checkpoint")


def build_model(config):
    model = PriorAU3(config)
    if model.config() != config:
        raise ValueError("Model configuration does not rebuild exactly")
    return model


@torch.no_grad()
def validate(model, loader, device, limit=0):
    was_training = model.training
    model.eval()
    l1_total = psnr_total = 0.
    count = 0
    try:
        for low, target in loader:
            low, target = low.to(device), target.to(device)
            # Deliberately full-resolution FP32 (no autocast).
            with torch.amp.autocast(device.type, enabled=False):
                output = model(low).float()
            if not torch.isfinite(output).all():
                raise RuntimeError("Non-finite validation output")
            l1 = (output - target).abs().flatten(1).mean(1)
            mse = (output.clamp(0, 1) - target).square().flatten(1).mean(1)
            psnr = -10 * torch.log10(mse.clamp_min(1e-12))
            l1_total += l1.sum().item()
            psnr_total += psnr.sum().item()
            count += low.shape[0]
            if limit and count >= limit:
                break
    finally:
        model.train(was_training)
    if count == 0:
        raise ValueError("Empty validation loader")
    return dict(val_count=count, val_l1=l1_total / count, val_psnr_rgb_float=psnr_total / count)
