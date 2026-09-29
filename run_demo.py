"""Original A with optional paired crop, flips or D4, and epoch budgets.

Validation and inference use full images. RGB values remain encoded [0, 1].
"""
from __future__ import annotations

import argparse
import hashlib
from datetime import datetime
import json
import math
import os
from pathlib import Path
import random
import shutil
import tempfile
import time

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Sampler

from model import EnhancementDemo


def build_model(config):
    if config.get("model_kind") == "color-prompt-v1":
        from color_prompt import ColorPromptEnhancement
        return ColorPromptEnhancement(config["model"], config["color_prompt"])
    return EnhancementDemo(**config["model"])


DEFAULT_DATA = Path(r"M:\picture data\cholec80_t\train_test")
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def read_rgb(path: Path) -> torch.Tensor:
    with Image.open(path) as image:
        array = np.asarray(image.convert("RGB"), dtype=np.float32).copy() / 255.0
    return torch.from_numpy(array).permute(2, 0, 1)


def image_files(directory: Path) -> list[Path]:
    return sorted(p for p in directory.iterdir()
                  if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)


def augment_pair(low, target, crop, seed, epoch, sample_id, quarter_turn=False):
    if low.shape != target.shape:
        raise ValueError("Paired dimensions differ")
    digest = hashlib.sha256(f"prior-a-v2:{seed}:{epoch}:{sample_id}".encode()).digest()
    local_seed = int.from_bytes(digest[:8], "little") % (2**63 - 1)
    generator = torch.Generator().manual_seed(local_seed)
    if crop is not None:
        ch, cw = crop
        h, w = low.shape[-2:]
        if h < ch or w < cw:
            raise ValueError(f"Training image {h}x{w} smaller than crop {ch}x{cw}")
        y = int(torch.randint(h - ch + 1, (), generator=generator))
        x = int(torch.randint(w - cw + 1, (), generator=generator))
        low, target = low[..., y:y+ch, x:x+cw], target[..., y:y+ch, x:x+cw]
    horizontal = bool(torch.rand((), generator=generator) < .5)
    vertical = bool(torch.rand((), generator=generator) < .5)
    axes = ([-1] if horizontal else []) + ([-2] if vertical else [])
    if axes:
        low, target = low.flip(axes), target.flip(axes)
    if quarter_turn:
        low = torch.rot90(low, 1, dims=(-2, -1))
        target = torch.rot90(target, 1, dims=(-2, -1))
    return low.contiguous(), target.contiguous()


class PairedImages(Dataset):
    def __init__(self, root: Path, split: str, crop_size=None, seed=100, flip_hv=False,
                 geometric_augs=False):
        self.crop_size = tuple(crop_size) if crop_size is not None and split == "train" else None
        self.flip_hv = bool(flip_hv) and split == "train"
        self.geometric_augs = bool(geometric_augs) and split == "train"
        self.seed = seed
        if split not in {"train", "val"}:
            raise ValueError("Only train and val splits are supported.")
        low, gt = root / split / "lowlight", root / split / "gt"
        self.low = image_files(low)
        targets = {p.name: p for p in image_files(gt)}
        if not self.low or {p.name for p in self.low} != set(targets):
            raise ValueError(f"Empty or unmatched lowlight/gt filenames: {split}")
        self.gt = [targets[p.name] for p in self.low]

    def __len__(self):
        return len(self.low)

    def __getitem__(self, key):
        quarter_turn = False
        if isinstance(key, tuple):
            index, epoch, *orientation = key
            if orientation:
                quarter_turn = bool(orientation[0])
        else:
            index, epoch = key, 0
        if self.geometric_augs and (not isinstance(key, tuple) or len(key) != 3):
            raise ValueError("D4 requires StepBatches with geometric_augs=True for shared batch orientation")
        low, gt = read_rgb(self.low[index]), read_rgb(self.gt[index])
        if low.shape != gt.shape:
            raise ValueError(f"Mismatched paired image shapes: {self.low[index]}")
        if self.crop_size is not None or self.flip_hv or self.geometric_augs:
            low, gt = augment_pair(low, gt, self.crop_size, self.seed, epoch, self.low[index].stem,
                                   quarter_turn=quarter_turn and self.geometric_augs)
        return low, gt


class StepBatches(Sampler):
    """Recreate each epoch's shuffle from seed; resumes at an exact batch.

    Keys carry the epoch for deterministic paired crops and flips. D4 keys also
    carry one shared 90-degree rotation bit per batch, keeping rectangles stackable.
    A final short batch is retained each epoch.
    """
    def __init__(self, count: int, batch: int, seed: int, start: int, stop: int,
                 geometric_augs=False):
        self.count, self.batch, self.seed = count, batch, seed
        self.start, self.stop = start, stop
        self.geometric_augs = bool(geometric_augs)

    def __len__(self):
        return max(0, self.stop - self.start)

    def __iter__(self):
        per_epoch = math.ceil(self.count / self.batch)
        previous_epoch, order = -1, []
        for step in range(self.start, self.stop):
            epoch, batch_index = divmod(step, per_epoch)
            if epoch != previous_epoch:
                generator = torch.Generator().manual_seed(self.seed + epoch)
                order = torch.randperm(self.count, generator=generator).tolist()
                previous_epoch = epoch
            offset = batch_index * self.batch
            indices = order[offset:offset + self.batch]
            if self.geometric_augs:
                digest = hashlib.sha256(f"prior-a-d4:{self.seed}:{step}".encode()).digest()
                quarter_turn = bool(digest[0] & 1)
                yield [(index, epoch, quarter_turn) for index in indices]
            else:
                yield [(index, epoch) for index in indices]


def check_output_location(output: Path, protected: list[Path]):
    for source in protected:
        source = source.resolve()
        if output == source or source in output.parents or output in source.parents:
            raise ValueError(f"Output must be separate from input/data: {output} vs {source}")


def choose_device(name: str) -> torch.device:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if name == "auto" else torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable.")
    return device


def atomic_save(value, path: Path):
    handle, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    os.close(handle)
    try:
        torch.save(value, temporary)
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def write_status(run: Path, state: str, step: int, target: int, **details):
    record = dict(state=state, step=step, target=target, pid=os.getpid(),
                  updated_at=datetime.now().isoformat(), **details)
    temporary = run / "status.json.tmp"
    temporary.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, run / "status.json")


def learning_rate(config: dict, step: int) -> float:
    """LR used by this 1-based optimizer step; resume needs no scheduler state."""
    if config["lr_schedule"] == "constant":
        return config["lr"]
    progress = (step - 1) / max(1, config["steps"] - 1)
    return config["min_lr"] + 0.5 * (config["lr"] - config["min_lr"]) * (1 + math.cos(math.pi * progress))


def rng_state():
    return {
        "python": random.getstate(), "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    l1_sum, psnr_sum, count = 0.0, 0.0, 0
    for low, target in loader:
        low, target = low.to(device), target.to(device)
        output = model(low).float()  # Validation intentionally uses FP32.
        if not torch.isfinite(output).all():
            raise RuntimeError("Non-finite validation output.")
        l1 = (output - target).abs().flatten(1).mean(1)
        mse = (output.clamp(0, 1) - target).square().flatten(1).mean(1)
        psnr = -10.0 * torch.log10(mse.clamp_min(1e-12))
        l1_sum += l1.sum().item()
        psnr_sum += psnr.sum().item()
        count += low.shape[0]
    model.train()
    return {"val_count": count, "val_l1": l1_sum / count,
            "val_psnr_rgb_float": psnr_sum / count}


def multiscale_l1(output, target):
    """Mean RGB L1 at full, half and quarter sizes, with normalized weights."""
    with torch.autocast(device_type=output.device.type, enabled=False):
        output, target = output.float(), target.float()
        full = F.l1_loss(output, target)
        half = F.l1_loss(F.avg_pool2d(output, 2), F.avg_pool2d(target, 2))
        quarter = F.l1_loss(F.avg_pool2d(output, 4), F.avg_pool2d(target, 4))
        total = (full + 0.5 * half + 0.25 * quarter) / 1.75
    return total, (full, half, quarter)


def train(args):
    data, run = args.data_root.resolve(), args.run_dir.resolve()
    check_output_location(run, [data])
    existing = list(run.glob("*.pt")) if run.exists() else []
    if existing and not args.resume:
        raise ValueError("Run directory contains checkpoints; use --resume or a new --run-dir.")
    if run.exists() and any(run.iterdir()) and not args.resume:
        raise ValueError("Use an empty run directory, or resume an existing checkpoint.")
    device = choose_device(args.device)
    amp = device.type == "cuda" and args.amp
    model_config = dict(width=args.width, spectral_mode=args.spectral_mode, output_mode=args.output_mode)
    if args.fusion_mode != "gated":
        model_config["fusion_mode"] = args.fusion_mode
    if args.bottleneck_attention != "none":
        model_config["bottleneck_attention"] = args.bottleneck_attention
    if args.bottleneck_spatial_attention != "none":
        model_config["bottleneck_spatial_attention"] = args.bottleneck_spatial_attention
    if args.half_decoder != "gated":
        model_config["half_decoder"] = args.half_decoder
    if args.half_encoder_frequency != "none":
        model_config["half_encoder_frequency"] = args.half_encoder_frequency
    if args.bottleneck_prior != "none":
        model_config["bottleneck_prior"] = args.bottleneck_prior
        if args.lfpv_channels != 64:
            model_config["lfpv_channels"] = args.lfpv_channels
        if args.lfpv_updater_expansion != 4:
            model_config["lfpv_updater_expansion"] = args.lfpv_updater_expansion
    if args.illumination_guidance != "none":
        model_config["illumination_guidance"] = args.illumination_guidance
    if args.brightness_prior_attention != "none":
        model_config["brightness_prior_attention"] = args.brightness_prior_attention
    dataset = PairedImages(data, "train", crop_size=args.crop_size, seed=args.seed,
                           flip_hv=args.flip_hv, geometric_augs=args.geometric_augs)
    color_config = None
    if args.color_prompt != "none":
        from color_prompt import ColorPromptPairs, prompt_config
        bundle = json.loads(args.color_prompt_bundle.read_text(encoding="utf-8"))
        color_config = prompt_config(bundle, args.color_prompt)
        dataset = ColorPromptPairs(dataset, bundle, args.color_prompt)
    steps_per_epoch = math.ceil(len(dataset) / args.batch_size)
    if args.epochs:
        args.steps = args.epochs * steps_per_epoch
    config = {
        "model": model_config, "data_root": str(data), "batch_size": args.batch_size,
        "seed": args.seed, "lr": args.lr, "weight_decay": args.weight_decay,
        "lr_schedule": args.lr_schedule, "min_lr": args.min_lr,
        "amp": amp, "grad_clip": args.grad_clip, "steps": args.steps,
        "workers": args.workers, "val_every": args.val_every, "save_every": args.save_every,
        "device": str(device), "loss": "L1 on unclamped encoded RGB",
        "loss_mode": args.loss_mode,
        "epochs": args.epochs or None, "steps_per_epoch": steps_per_epoch,
        "crop_size": list(args.crop_size) if args.crop_size else None,
        "horizontal_flip_probability": 0.5 if (args.crop_size or args.flip_hv or args.geometric_augs) else 0.0,
        "vertical_flip_probability": 0.5 if (args.crop_size or args.flip_hv or args.geometric_augs) else 0.0,
        "flip_hv": bool(args.flip_hv),
        "geometric_augs": bool(args.geometric_augs),
        "image_policy": (f"train: paired random crop {args.crop_size[0]}x{args.crop_size[1]}, "
                         "horizontal/vertical flip each p=0.5; val/infer: whole image"
                         if args.crop_size else
                         "train: whole image, horizontal/vertical flip each p=0.5; val/infer: whole image"
                         if args.flip_hv else "whole image, no resize/crop/augmentation"),
        "best_metric": args.best_metric,
        "val_policy": "image-equal raw L1 and clamped RGB float PSNR (MSE floor 1e-12); no GT mean; "
                      f"best_val.pt selected by {'minimum raw L1' if args.best_metric == 'l1' else 'maximum clamped RGB float PSNR'}",
        "resume_policy": "exact epoch/batch shuffle and saved RNG; backend bitwise determinism not guaranteed",
    }
    if color_config is not None:
        config.update(model_kind="color-prompt-v1", color_prompt=color_config,
                      color_prompt_bundle_sha256=hashlib.sha256(args.color_prompt_bundle.read_bytes()).hexdigest(),
                      color_prompt_parameters_added=432, initialization="from scratch; seed100; shared A+MDTA weights retained")
    if args.geometric_augs:
        image_policy = (f"paired random crop {args.crop_size[0]}x{args.crop_size[1]}"
                        if args.crop_size else "whole image")
        config.update(
            geometric_recipe="d4-batch-quarterturn-image-hv-v1",
            d4_transform_probability=0.125, d4_quarter_turn_probability=0.5,
            d4_batch_policy="shared 90-degree rotation bit; independent image horizontal/vertical flips",
            image_policy=f"train: {image_policy}, paired uniform D4; val/infer: whole image",
        )
    if args.loss_mode == "ms-l1":
        config.update(
            loss="(L1_full + 0.5 * L1_half + 0.25 * L1_quarter) / 1.75 on unclamped encoded RGB",
            loss_scales=[1, 2, 4], loss_weights=[1.0, 0.5, 0.25],
            loss_normalizer=1.75, loss_downsample="avg_pool2d; kernel=stride=scale; FP32",
        )
    if args.loss_mode == "gt-mean-l1":
        from gt_mean_loss import gt_mean_l1
        config.update(
            loss="W * raw RGB L1 + (1-W) * brightness-aligned clamped RGB L1",
            gt_mean_sigma=args.gt_mean_sigma, gt_mean_eps=1e-6,
            gt_mean_gray_weights=[0.2989, 0.5870, 0.1140],
            gt_mean_weight="clipped Bhattacharyya distance; detached",
            gt_mean_gain="GT gray mean / prediction gray mean; differentiable",
            gt_mean_guard="raw L1 fallback for either gray mean <= 1e-6",
            gt_mean_precision="FP32; GT mean used only in training loss",
        )
    if args.loss_mode == "gt-mean-blur-l1":
        from blur_match_loss import blur_match_l1
        from gt_mean_loss import gt_mean_l1
        config.update(
            loss="GT-Mean L1 + sum_s w_s * MAE(G_s(output), G_s(GT)), normalized Gaussian, FP32",
            gt_mean_sigma=args.gt_mean_sigma, gt_mean_eps=1e-6,
            gt_mean_gray_weights=[0.2989, 0.5870, 0.1140],
            gt_mean_weight="clipped Bhattacharyya distance; detached",
            gt_mean_gain="GT gray mean / prediction gray mean; differentiable",
            gt_mean_guard="raw L1 fallback for either gray mean <= 1e-6",
            gt_mean_precision="FP32; GT mean used only in training loss",
            blur_loss_sigmas=list(args.blur_loss_sigmas),
            blur_loss_weights=list(args.blur_loss_weights),
            blur_loss_kernel="normalized separable Gaussian; radius ceil(3*sigma); reflect padding",
            blur_loss_precision="FP32; pixel mean absolute difference of blurred tensors",
        )
    if args.illumination_guidance == "three-stage":
        config.update(
            illumination_recipe="explicit-six-three-stage-v1",
            illumination_inputs=["mean", "rec709", "max", "lightness", "ycgco_y", "rgb_l2"],
            illumination_widths=[16, 32, 64],
            illumination_sites=["after half encoder GCB: IG-MSA", "bottom MDTA V", "after half decoder GCB: IG-MSA"],
            illumination_gate="1+tanh(stage projection); initial one; V only",
            illumination_attention="complete IG-MSA attention, positional branch, no extra IGAB FFN",
            illumination_initialization="shared A+MDTA seed preserved; zero gate heads and half-attention output projections",
            illumination_precision="FP32 descriptors/CNN/gates/V multiplication; AMP backbone and attention",
            illumination_supervision="final GT-Mean L1 only; prior input low RGB only",
            illumination_source="https://github.com/caiyuanhao1998/Retinexformer",
        )
    if args.illumination_guidance == "bottom-v":
        config.update(
            illumination_recipe="explicit-six-bottom-v-v1",
            illumination_inputs=["mean", "rec709", "max", "lightness", "ycgco_y", "rgb_l2"],
            illumination_widths=[16, 32, 64],
            illumination_sites=["bottom MDTA V"],
            illumination_gate="1+tanh(bottom projection); initial one; V only",
            illumination_attention="existing bottom MDTA; unchanged Q/K calculation; no additional attention blocks",
            illumination_initialization="shared A+MDTA seed preserved; zero bottom gate head",
            illumination_precision="FP32 descriptors/CNN/gate/V multiplication; AMP backbone and attention",
            illumination_supervision="final GT-Mean L1 only; prior input low RGB only",
            illumination_source="https://github.com/caiyuanhao1998/Retinexformer",
        )
    if args.bottleneck_prior == "lfpv":
        config.update(
            lfpv_site="after bottom Spatial Fusion and MDTA, before decoder",
            lfpv_channels=args.lfpv_channels, lfpv_vectors=16, lfpv_patches=16, lfpv_patch_size=4,
            lfpv_updater_layers=7, lfpv_updater_expansion=args.lfpv_updater_expansion,
            lfpv_updater_hidden=f"{args.lfpv_updater_expansion}c x5, c x1; final c",
            lfpv_updater_activation="BatchNorm + ReLU; published placement, configurable widths",
            lfpv_training="shared encoder/decoder; bypass and SU+MU+query outputs",
            lfpv_loss_weights=[0.5, 0.5],
            lfpv_precision="FP32 SU/MU/query/reference buffers; AMP backbone/projections",
            lfpv_reference_update="detached updater output committed after successful optimizer step",
            lfpv_eval="query saved references only; no SU/MU or reference updates",
            lfpv_source="https://github.com/xiaogang00/LFPVS_ICCV",
        )
        config["loss"] = "0.5 * GT-Mean L1(bypass) + 0.5 * GT-Mean L1(LFPV-guided)"
    if args.bottleneck_spatial_attention == "shifted-window":
        config.update(
            window_attention_recipe="attention-only-wmsa-swmsa-v1",
            window_attention_site="after original Spatial Fusion, before original bottom MDTA",
            window_attention_channels=4 * args.width, window_attention_size=8,
            window_attention_heads=4, window_attention_shifts=[0, 4],
            window_attention_ffn=False, window_attention_relative_position_bias=True,
            window_attention_mask="block cyclic boundary wraparound; recompute per feature size/device",
            window_attention_residual_scale="learned per channel, initialized 0.1 per layer",
            window_attention_initialization="construct last; shared A+MDTA seeded weights unchanged",
            window_attention_precision="AMP projections/matmul; FP32 scores, bias, mask and softmax",
            window_attention_source="https://github.com/JingyunLiang/SwinIR/blob/main/models/network_swinir.py",
        )
    if args.brightness_prior_attention in {"bottom-cross4", "multiscale-cross4"}:
        config.update(
            brightness_prior_recipe="four-fixed-bottom-cross-attention-v1",
            brightness_prior_inputs=["shifted-power-gamma0.4-eps0.02", "gaussian5-log20", "sine-power0.2", "log20"],
            brightness_prior_gray_weights=[0.299, 0.587, 0.114],
            brightness_prior_normalization="fixed [0,1]; no per-image min/max",
            brightness_prior_encoder_widths=[4, 16, 32, 4 * args.width],
            brightness_prior_site="after original Spatial Fusion, before unchanged bottom MDTA",
            brightness_prior_attention="main Q; prior K,V; channel cross-attention; heads4",
            brightness_prior_ffn_expansion=2, brightness_prior_residual_scale_init=0.1,
            brightness_prior_initialization="construct last; shared A+MDTA seeded weights retained",
            brightness_prior_precision="FP32 fixed maps, Q/K normalization and scores; AMP encoder/projections/FFN",
            brightness_prior_supervision="low RGB after augmentation only; GT only in unchanged final GT-Mean loss",
            brightness_prior_source="https://github.com/minyan8/imagine/blob/main/basicsr/models/archs/RetinexFormer_arch.py",
        )
    if args.brightness_prior_attention == "multiscale-cross4":
        config.update(
            brightness_prior_recipe="four-fixed-five-site-cross-attention-v1",
            brightness_prior_sites=["after encoder0", "after encoder1", "after Spatial Fusion before MDTA", "after decoder1", "after decoder0"],
            brightness_prior_site="two encoder sites + unchanged bottom site + two decoder sites",
            brightness_prior_channels=[args.width, 2 * args.width, 4 * args.width, 2 * args.width, args.width],
            brightness_prior_heads=[1, 2, 4, 2, 1],
            brightness_prior_attention="main Q; shared-pyramid prior K,V; independent channel cross-attention+2xFFN at five sites",
            brightness_prior_pyramid="four maps once; existing half16 and bottom96; new full4/16/24 and half16/48 projections",
            brightness_prior_initialization="common A+MDTA and bottom-guidance seed100 weights retained; four extra sites initialized last",
        )
    checkpoint = None
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        previous = dict(checkpoint["config"])
        previous.setdefault("lr_schedule", "constant")
        previous.setdefault("min_lr", 2e-6)
        previous.setdefault("best_metric", "l1")
        previous.setdefault("loss_mode", "l1")
        previous.setdefault("crop_size", None)
        previous.setdefault("flip_hv", previous.get("crop_size") is not None)
        previous.setdefault("geometric_augs", False)
        keys = ["model", "data_root", "batch_size", "seed", "lr", "weight_decay", "amp", "grad_clip", "lr_schedule", "best_metric", "crop_size", "flip_hv", "geometric_augs", "loss_mode"]
        if color_config is not None or previous.get("model_kind") == "color-prompt-v1":
            keys.extend(["model_kind", "color_prompt", "color_prompt_bundle_sha256"])
        if args.geometric_augs:
            keys.extend(["geometric_recipe", "d4_transform_probability", "d4_batch_policy"])
        if args.loss_mode in {"gt-mean-l1", "gt-mean-blur-l1"}:
            keys.extend(["gt_mean_sigma", "gt_mean_eps"])
        if args.loss_mode == "gt-mean-blur-l1":
            keys.extend(["blur_loss_sigmas", "blur_loss_weights"])
        if args.illumination_guidance in {"three-stage", "bottom-v"}:
            keys.extend(["illumination_recipe", "illumination_initialization", "illumination_precision"])
        if args.bottleneck_prior == "lfpv":
            keys.extend(["lfpv_loss_weights", "lfpv_precision", "lfpv_reference_update"])
        if args.bottleneck_spatial_attention == "shifted-window":
            keys.extend(["window_attention_recipe", "window_attention_initialization", "window_attention_precision"])
        if args.lr_schedule == "cosine":
            keys.extend(["min_lr", "steps"])
        for key in keys:
            if config[key] != previous[key]:
                raise ValueError(f"Resume configuration differs for {key}; repeat the original training flags.")
        if checkpoint["step"] >= args.steps:
            raise ValueError("--steps must exceed the checkpoint's completed step.")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    model = build_model(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=amp, init_scale=1024.0)
    start_step, best_val = 0, None
    if checkpoint is not None:
        model.load_state_dict(checkpoint["model"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        scaler.load_state_dict(checkpoint["scaler"])
        start_step, best_val = checkpoint["step"], checkpoint.get("best_val")
        restore_rng(checkpoint["rng"])
    names = [p.name for p in dataset.low]
    if checkpoint is not None and checkpoint["train_filenames"] != names:
        raise ValueError("Training filename list changed since checkpoint.")
    stop_step = min(args.steps, start_step + args.stop_after) if args.stop_after else args.steps
    sampler = StepBatches(len(dataset), args.batch_size, args.seed, start_step, stop_step,
                          geometric_augs=args.geometric_augs)
    # A dedicated loader generator avoids consuming the model's RNG state.
    loader = DataLoader(dataset, batch_sampler=sampler, num_workers=args.workers,
                        generator=torch.Generator().manual_seed(args.seed + 100000),
                        pin_memory=device.type == "cuda")
    val_loader = None
    if args.val_every:
        val_loader = DataLoader(PairedImages(data, "val"), batch_size=1, shuffle=False,
                                num_workers=args.workers,
                                generator=torch.Generator().manual_seed(args.seed + 200000))
    run.mkdir(parents=True, exist_ok=True)
    if (args.bottleneck_attention == "mdta" or args.loss_mode in {"ms-l1", "gt-mean-l1", "gt-mean-blur-l1"}
            or args.half_decoder != "gated" or args.half_encoder_frequency != "none") and not args.resume:
        snapshot = run / "code"
        snapshot.mkdir()
        files = ["model.py", "run_demo.py", "test_best.py"]
        if args.bottleneck_attention == "mdta":
            files.extend(["mdta_blocks.py", "train_mdta.py"])
        if args.bottleneck_spatial_attention == "shifted-window":
            files.extend(["window_attention_blocks.py", "train_shifted_window.py"])
        if args.brightness_prior_attention in {"bottom-cross4", "multiscale-cross4"}:
            files.extend(["brightness_cross_blocks.py", "train_brightness_cross.py", "BRIGHTNESS_CROSS_SOURCE.md"])
            if args.brightness_prior_attention == "multiscale-cross4":
                files.extend(["brightness_multiscale_blocks.py", "train_brightness_multiscale.py", "BRIGHTNESS_MULTISCALE_SOURCE.md"])
        if args.loss_mode == "ms-l1":
            files.append("train_ms_l1.py")
        if args.half_decoder != "gated":
            files.extend(["train_mdta_half.py", "mdta_blocks.py"])
        if args.half_decoder == "restormer":
            files.append("restormer_blocks.py")
        if args.half_encoder_frequency != "none":
            files.extend(["fremlp_blocks.py", "train_fremlp.py"])
        if args.loss_mode in {"gt-mean-l1", "gt-mean-blur-l1"}:
            files.extend(["gt_mean_loss.py", "train_gt_mean.py"])
        if args.loss_mode == "gt-mean-blur-l1":
            files.extend(["blur_match_loss.py", "train_gt_mean_blur.py"])
        if args.geometric_augs:
            files.append("train_gt_mean_d4_30k.py")
        if args.bottleneck_prior == "lfpv":
            files.extend(["lfpv_blocks.py", "train_lfpv.py", "LFPV_SOURCE.md"])
            if (args.lfpv_channels, args.lfpv_updater_expansion) != (64, 4):
                files.append("train_lfpv_light.py")
        if args.illumination_guidance == "three-stage":
            files.extend(["illumination_blocks.py", "train_illumination.py", "ILLUMINATION_SOURCE.md"])
        if args.illumination_guidance == "bottom-v":
            files.extend(["illumination_blocks.py", "train_illumination_v.py", "ILLUMINATION_V_SOURCE.md"])
        if color_config is not None:
            files.extend(["color_prompt.py", "train_color_prompt.py"])
            shutil.copyfile(args.color_prompt_bundle, snapshot / "color_prompt_bundle.json")
        for name in dict.fromkeys(files):
            shutil.copyfile(Path(__file__).parent / name, snapshot / name)
    (run / "config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"run": str(run), "device": str(device), "amp": amp,
                      "parameters": sum(p.numel() for p in model.parameters()),
                      "train_images": len(dataset), "epochs": args.epochs or None,
                      "steps_per_epoch": steps_per_epoch, "crop_size": args.crop_size,
                      "geometric_augs": args.geometric_augs,
                      "start_step": start_step,
                      "stop_step": stop_step, "target_step": args.steps, "test_images_read": 0}), flush=True)
    write_status(run, "running", start_step, args.steps)
    model.train()
    with (run / "loss.jsonl").open("a", encoding="utf-8") as log:
        for step, batch in enumerate(loader, start=start_step + 1):
            started = time.perf_counter()
            low, target = batch[:2]
            cue = batch[2].to(device) if color_config is not None else None
            for group in optimizer.param_groups:
                group["lr"] = learning_rate(config, step)
            low, target = low.to(device), target.to(device)
            optimizer.zero_grad(set_to_none=True)
            lfpv_paths = None
            with torch.autocast(device_type=device.type, enabled=amp, dtype=torch.float16):
                if args.bottleneck_prior == "lfpv":
                    lfpv_paths = model(low, lfpv_training=True)
                    output = lfpv_paths["output"]
                elif color_config is not None:
                    output = model(low, color_prompt=cue)
                else:
                    output = model(low)
                if args.loss_mode == "l1":
                    loss = torch.nn.functional.l1_loss(output.float(), target)
            if args.loss_mode == "ms-l1":
                loss, scale_losses = multiscale_l1(output, target)
            elif args.loss_mode == "gt-mean-l1":
                loss, gt_mean_diagnostics = gt_mean_l1(output, target, sigma=args.gt_mean_sigma)
                if lfpv_paths is not None:
                    refined_loss = loss
                    base_loss, _ = gt_mean_l1(lfpv_paths["base_output"], target, sigma=args.gt_mean_sigma)
                    loss = 0.5 * base_loss + 0.5 * refined_loss
            elif args.loss_mode == "gt-mean-blur-l1":
                loss, gt_mean_diagnostics = gt_mean_l1(output, target, sigma=args.gt_mean_sigma)
                blur_loss, blur_diagnostics = blur_match_l1(
                    output, target, sigmas=tuple(args.blur_loss_sigmas),
                    weights=tuple(args.blur_loss_weights))
                loss = loss + blur_loss
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss at step {step}.")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip,
                                                       error_if_nonfinite=not amp)
            previous_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            skipped = bool(amp and scaler.get_scale() < previous_scale)
            if lfpv_paths is not None and not skipped:
                model.lfpv.commit(lfpv_paths["lfpv_update"])
            row = {"step": step, "epoch": (step - 1) // steps_per_epoch + 1,
                   "batch_in_epoch": (step - 1) % steps_per_epoch + 1, "loss": loss.item(),
                   "grad_norm_before_clip": float(grad_norm) if torch.isfinite(grad_norm) else None,
                   "amp_skipped_step": bool(amp and scaler.get_scale() < previous_scale),
                   "amp_scale": scaler.get_scale(),
                   "seconds": time.perf_counter() - started, "lr": optimizer.param_groups[0]["lr"]}
            if args.loss_mode == "ms-l1":
                row.update(zip(("loss_full", "loss_half", "loss_quarter"),
                               torch.stack(scale_losses).detach().cpu().tolist()))
            if args.loss_mode in {"gt-mean-l1", "gt-mean-blur-l1"}:
                row.update({key: value.item() for key, value in gt_mean_diagnostics.items()})
            if args.loss_mode == "gt-mean-blur-l1":
                row.update({key: value.item() for key, value in blur_diagnostics.items()})
            if lfpv_paths is not None:
                core = model.lfpv.core
                row.update(lfpv_bypass_loss=base_loss.item(), lfpv_guided_loss=refined_loss.item(),
                           lfpv_vector_std=core.common_feature.std(unbiased=False).item(),
                           lfpv_patch_std=core.common_feature_patch.std(unbiased=False).item())
            if args.geometric_augs:
                row["train_shape_hw"] = list(low.shape[-2:])
            improved = False
            if val_loader is not None and (step % args.val_every == 0 or step == args.steps):
                row.update(validate(model, val_loader, device))
                score = row["val_l1"] if config["best_metric"] == "l1" else row["val_psnr_rgb_float"]
                if best_val is None or (score < best_val if config["best_metric"] == "l1" else score > best_val):
                    best_val, improved = score, True
            log.write(json.dumps(row, allow_nan=False) + "\n")
            log.flush()
            if step == start_step + 1 or step % args.log_every == 0 or step == args.steps:
                print(json.dumps(row, allow_nan=False), flush=True)
                write_status(run, "running", step, args.steps, loss=row["loss"], lr=row["lr"],
                             peak_allocated_mib=torch.cuda.max_memory_allocated(device) / 2**20 if amp else None)
            if improved or step % args.save_every == 0 or step == stop_step:
                state = {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                         "scaler": scaler.state_dict(), "step": step, "config": config,
                         "rng": rng_state(), "best_val": best_val, "train_filenames": names}
                atomic_save(state, run / "last.pt")
                if step % args.save_every == 0 or step == args.steps:
                    atomic_save(state, run / f"step_{step:06d}.pt")
                if improved:
                    atomic_save(state, run / "best_val.pt")
    write_status(run, "complete" if stop_step == args.steps else "bounded_stop", stop_step, args.steps,
                 **{("best_val_l1" if config["best_metric"] == "l1" else "best_val_psnr"): best_val,
                    "best_metric": config["best_metric"]},
                 peak_allocated_mib=torch.cuda.max_memory_allocated(device) / 2**20 if amp else None)


@torch.no_grad()
def infer(args):
    source, destination = args.input.resolve(), args.output_dir.resolve()
    files = image_files(source) if source.is_dir() else [source]
    if not files or any(not p.is_file() or p.suffix.lower() not in IMAGE_SUFFIXES for p in files):
        raise ValueError("Input must be an image or a nonempty image directory.")
    check_output_location(destination, [source if source.is_dir() else source.parent, DEFAULT_DATA])
    outputs = [destination / (p.stem + ".png") for p in files]
    if len({str(p).casefold() for p in outputs}) != len(outputs):
        raise ValueError("Input names collide after conversion to PNG.")
    if any(p.exists() for p in outputs):
        raise ValueError("Output PNG already exists; use another output directory.")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    device = choose_device(args.device)
    model = build_model(checkpoint["config"]).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    destination.mkdir(parents=True, exist_ok=True)
    for source_file, output_file in zip(files, outputs):
        tensor = read_rgb(source_file).unsqueeze(0).to(device)
        output = model(tensor).float()
        if output.shape != tensor.shape or not torch.isfinite(output).all():
            raise RuntimeError(f"Invalid model output for {source_file}")
        array = output[0].clamp(0, 1).permute(1, 2, 0).cpu().numpy()
        Image.fromarray(np.rint(array * 255).astype(np.uint8)).save(output_file)
        print(str(output_file), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    training = commands.add_parser("train", help="Paired training with optional crop/flip augmentation")
    training.add_argument("--data-root", type=Path, default=DEFAULT_DATA)
    training.add_argument("--run-dir", type=Path, default=Path(__file__).parent / "runs" / "demo")
    training.add_argument("--steps", type=int, default=1000)
    training.add_argument("--epochs", type=int, default=0,
                          help="When positive, sets total steps to epochs * ceil(train_count / batch_size)")
    training.add_argument("--crop-size", type=int, nargs=2, metavar=("HEIGHT", "WIDTH"),
                          help="Paired training crop with horizontal/vertical flips, each p=0.5")
    training.add_argument("--flip-hv", action="store_true",
                          help="Horizontal/vertical flips each p=0.5, with or without crop")
    training.add_argument("--geometric-augs", action=argparse.BooleanOptionalAction, default=False,
                          help="Paired uniform D4; each batch shares whether to swap height/width")
    training.add_argument("--batch-size", type=int, default=2)
    training.add_argument("--lr", type=float, default=2e-4)
    training.add_argument("--lr-schedule", choices=["constant", "cosine"], default="constant")
    training.add_argument("--min-lr", type=float, default=2e-6)
    training.add_argument("--stop-after", type=int, default=0,
                          help="Run a bounded number of additional steps without changing the LR horizon")
    training.add_argument("--weight-decay", type=float, default=1e-4)
    training.add_argument("--grad-clip", type=float, default=1.0)
    training.add_argument("--seed", type=int, default=100)
    training.add_argument("--workers", type=int, default=0)
    training.add_argument("--width", type=int, default=24)
    training.add_argument("--spectral-mode", choices=["conditional", "static", "off"], default="conditional")
    training.add_argument("--bottleneck-attention", choices=["none", "mdta"], default="none")
    training.add_argument("--color-prompt", choices=["none", "zero", "ridge"], default="none",
                          help="Same 5-channel stem: zero control or two frozen predicted colour conditions")
    training.add_argument("--color-prompt-bundle", type=Path,
                          help="Portable frozen all-train ridge plus train OOF input predictions")
    training.add_argument("--bottleneck-spatial-attention", choices=["none", "shifted-window"], default="none",
                          help="Insert an attention-only window8/shift4 pair between Spatial Fusion and MDTA")
    training.add_argument("--half-decoder", choices=["gated", "mdta-gated", "restormer"],
                          default="gated", help="Half decoder: original, MDTA plus original, or full Restormer replacement")
    training.add_argument("--half-encoder-frequency", choices=["none", "fremlp"],
                          default="none", help="Append a zero-initialized FreMLP modulation after encoder1")
    training.add_argument("--bottleneck-prior", choices=["none", "lfpv"], default="none",
                          help="Full LFPV core with shared-decoder dual-path GT-Mean training")
    training.add_argument("--lfpv-channels", type=int, default=64,
                          help="LFPV internal channels; default published width 64")
    training.add_argument("--lfpv-updater-expansion", type=int, default=4,
                          help="SU/MU hidden channel multiplier; seven-layer depth retained")
    training.add_argument("--illumination-guidance", choices=["none", "three-stage", "bottom-v"], default="none",
                          help="Independent low-image brightness branch: three sites or bottom MDTA V only")
    training.add_argument("--brightness-prior-attention", choices=["none", "bottom-cross4", "multiscale-cross4"], default="none",
                          help="Four fixed brightness maps with bottom-only or five-site channel cross-attention")
    training.add_argument("--fusion-mode", choices=["gated", "additive", "none", "star"],
                          default="gated",
                          help="Bottleneck fusion; none removes it, star replaces it with one full StarBlock.")
    training.add_argument("--output-mode", choices=["structured", "direct", "rgb_interaction"],
                          default="structured",
                          help="Output parameterization. rgb_interaction is the per-colour gated residual head.")
    training.add_argument("--loss-mode", choices=["l1", "ms-l1", "gt-mean-l1", "gt-mean-blur-l1"], default="l1",
                          help="Final-output RGB L1, multi-scale L1, GT-Mean L1, or GT-Mean plus blurred matching")
    training.add_argument("--gt-mean-sigma", type=float, default=0.1,
                          help="Brightness distribution spread for GT-Mean L1")
    training.add_argument("--blur-loss-sigmas", type=float, nargs="+", default=[4.0, 8.0],
                          help="Gaussian sigmas for the blurred matching loss")
    training.add_argument("--blur-loss-weights", type=float, nargs="+", default=[0.05, 0.05],
                          help="Weights for each blurred matching scale")
    training.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    training.add_argument("--device", default="auto")
    training.add_argument("--resume", type=Path)
    training.add_argument("--val-every", type=int, default=0,
                          help="0 disables validation; positive values evaluate full val at this interval")
    training.add_argument("--best-metric", choices=["l1", "psnr"], default="l1",
                          help="Metric that decides best_val.pt independently of the training loss mode.")
    training.add_argument("--save-every", type=int, default=100)
    training.add_argument("--log-every", type=int, default=10)
    training.set_defaults(function=train)
    inference = commands.add_parser("infer", help="Infer images using a trained checkpoint")
    inference.add_argument("--checkpoint", type=Path, required=True)
    inference.add_argument("--input", type=Path, required=True)
    inference.add_argument("--output-dir", type=Path, required=True)
    inference.add_argument("--device", default="auto")
    inference.set_defaults(function=infer)
    args = parser.parse_args()
    if args.command == "train":
        for name in ("steps", "batch_size", "lr", "grad_clip", "width", "save_every", "log_every"):
            if getattr(args, name) <= 0:
                parser.error(f"--{name.replace('_', '-')} must be positive")
        if args.color_prompt != "none" and args.color_prompt_bundle is None:
            parser.error("--color-prompt requires --color-prompt-bundle")
        if min(args.lfpv_channels, args.lfpv_updater_expansion) < 1:
            parser.error("LFPV widths must be positive")
        if args.bottleneck_prior != "lfpv" and (args.lfpv_channels, args.lfpv_updater_expansion) != (64, 4):
            parser.error("Non-default LFPV widths require --bottleneck-prior lfpv")
        if args.bottleneck_prior == "lfpv" and args.loss_mode != "gt-mean-l1":
            parser.error("LFPV experiment requires --loss-mode gt-mean-l1")
        if args.loss_mode in {"gt-mean-l1", "gt-mean-blur-l1"} and args.gt_mean_sigma <= 0:
            parser.error("--gt-mean-sigma must be positive")
        if args.loss_mode == "gt-mean-blur-l1":
            if len(args.blur_loss_sigmas) != len(args.blur_loss_weights) or not args.blur_loss_sigmas:
                parser.error("--blur-loss-sigmas and --blur-loss-weights must have equal positive length")
            if min(args.blur_loss_sigmas) <= 0 or min(args.blur_loss_weights) < 0:
                parser.error("blur sigmas must be positive and blur weights nonnegative")
        if args.epochs < 0 or (args.crop_size is not None and min(args.crop_size) <= 0):
            parser.error("epochs must be nonnegative and crop dimensions positive")
        if args.workers < 0 or args.val_every < 0 or args.weight_decay < 0:
            parser.error("workers, val-every and weight-decay must be nonnegative")
        if args.stop_after < 0 or not 0 <= args.min_lr <= args.lr:
            parser.error("stop-after must be nonnegative; min-lr must be between 0 and lr")
    try:
        args.function(args)
    except BaseException as error:
        if args.command == "train" and (args.run_dir / "status.json").exists():
            prior = json.loads((args.run_dir / "status.json").read_text(encoding="utf-8"))
            if prior.get("pid") == os.getpid():
                write_status(args.run_dir, "failed", prior["step"], args.steps,
                             error_type=type(error).__name__, error=str(error))
        raise


if __name__ == "__main__":
    main()
