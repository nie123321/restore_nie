"""Training engine for this folder's installed experiment model.

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

import sys

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
for runtime_path in (PACKAGE_ROOT, PACKAGE_ROOT / "training" / "losses"):
    if str(runtime_path) not in sys.path:
        sys.path.insert(0, str(runtime_path))
from models.final_model import EnhancementDemo, MODEL_SPEC, validate_model_config
from wavelet_blocks import WAVELET_VARIANTS, FREQUENCY_VARIANTS
from frequency_candidate_blocks import CANDIDATE_VARIANTS


DEFAULT_DATA = (Path(r"M:\picture data\cholec80_t\train_test") if os.name == "nt"
                else Path("/root/autodl-tmp/datasets/cholec80/train_test"))
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


def saturation_parameters(seed, epoch, sample_id, probability, factor_range):
    """Independent per-image/epoch RNG; does not perturb crop, D4 or model RNG."""
    digest = hashlib.sha256(f"paired-saturation-v1:{seed}:{epoch}:{sample_id}".encode()).digest()
    generator = torch.Generator().manual_seed(int.from_bytes(digest[:8], "little") % (2**63 - 1))
    applied = bool(torch.rand((), generator=generator) < probability)
    factor = factor_range[0] + (factor_range[1] - factor_range[0]) * float(torch.rand((), generator=generator))
    return applied, factor


class PairedImages(Dataset):
    def __init__(self, root: Path, split: str, crop_size=None, seed=100, flip_hv=False,
                 geometric_augs=False, saturation_probability=0.0,
                 saturation_range=(0.9, 1.1)):
        self.crop_size = tuple(crop_size) if crop_size is not None and split == "train" else None
        self.flip_hv = bool(flip_hv) and split == "train"
        self.geometric_augs = bool(geometric_augs) and split == "train"
        self.seed = seed
        if not 0 <= saturation_probability <= 1 or not (0 < saturation_range[0] <= saturation_range[1]):
            raise ValueError("Invalid saturation probability or factor range")
        self.saturation_probability = float(saturation_probability) if split == "train" else 0.0
        self.saturation_range = tuple(saturation_range)
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
        if self.saturation_probability:
            applied, factor = saturation_parameters(self.seed, epoch, self.low[index].stem,
                                                   self.saturation_probability, self.saturation_range)
            if applied:
                from torchvision.transforms.functional import adjust_saturation
                low = adjust_saturation(low, factor)
                gt = adjust_saturation(gt, factor)
        return low.contiguous(), gt.contiguous()


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


def _cosine(start: float, end: float, progress: float) -> float:
    return end + 0.5 * (start - end) * (1 + math.cos(math.pi * progress))


def learning_rate(config: dict, step: int) -> float:
    """LR used by this 1-based optimizer step; resume needs no scheduler state."""
    if config["lr_schedule"] == "constant":
        return config["lr"]
    if config["lr_schedule"] == "two-stage-cosine":
        boundary = int(config["stage_steps"])
        if step <= boundary:
            progress = (step - 1) / max(1, boundary - 1)
            return _cosine(config["lr"], config["min_lr"], progress)
        span = int(config["steps"]) - boundary
        progress = (step - boundary - 1) / max(1, span - 1)
        return _cosine(config["stage2_lr"], config["stage2_min_lr"], progress)
    progress = (step - 1) / max(1, config["steps"] - 1)
    return config["min_lr"] + 0.5 * (config["lr"] - config["min_lr"]) * (1 + math.cos(math.pi * progress))


def rng_state():
    name, keys, position, has_gauss, cached = np.random.get_state()
    portable_numpy = {"algorithm": name, "keys": keys.tolist(), "position": int(position),
                      "has_gauss": int(has_gauss), "cached_gaussian": float(cached)}
    return {
        "python": random.getstate(), "numpy": portable_numpy,
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_rng(state):
    random.setstate(state["python"])
    numpy_state = state["numpy"]
    if isinstance(numpy_state, dict):
        numpy_state = (numpy_state["algorithm"], np.asarray(numpy_state["keys"], dtype=np.uint32),
                       numpy_state["position"], numpy_state["has_gauss"], numpy_state["cached_gaussian"])
    np.random.set_state(numpy_state)
    torch.set_rng_state(state["torch"])
    if state["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def _wang_channel_ssim_u8(target: np.ndarray, candidate: np.ndarray) -> float:
    """One uint8 channel. Same 11x11 Gaussian and constants as evaluate_four_metrics.channel_ssim."""
    import cv2
    c1 = (0.01 * 255.0) ** 2
    c2 = (0.03 * 255.0) ** 2
    target = target.astype(np.float64)
    candidate = candidate.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())
    mu1 = cv2.filter2D(target, -1, window)[5:-5, 5:-5]
    mu2 = cv2.filter2D(candidate, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(target ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(candidate ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(target * candidate, -1, window)[5:-5, 5:-5] - mu1_mu2
    result = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / (
        (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
    )
    return float(result.mean())


def _uint8_wang_ssim(output: torch.Tensor, target: torch.Tensor) -> float:
    """Mean over the batch. uint8 rounding matches the PNG saved by inference and test_best."""
    predicted = np.rint(
        output.detach().float().clamp(0, 1).permute(0, 2, 3, 1).cpu().numpy() * 255.0
    ).astype(np.uint8)
    reference = np.rint(
        target.detach().float().clamp(0, 1).permute(0, 2, 3, 1).cpu().numpy() * 255.0
    ).astype(np.uint8)
    scores = [
        float(np.mean([
            _wang_channel_ssim_u8(reference[index, :, :, channel], predicted[index, :, :, channel])
            for channel in range(3)
        ]))
        for index in range(predicted.shape[0])
    ]
    return float(np.mean(scores))


@torch.no_grad()
def validate(model, loader, device, lf_train_mean=None, preview_dir=None, preview_count=4,
             loss_mode="l1", gt_mean_sigma=0.1, fft_weight=0.1):
    model.eval()
    l1_sum, psnr_sum, loss_sum, ssim_sum, count = 0.0, 0.0, 0.0, 0.0, 0
    gt_mean_fft_l1 = None
    if loss_mode == "gt-mean-fft":
        from frequency_loss import gt_mean_fft_l1
    lf_sums = {}
    preview_indices = set()
    if model.lf_branch is not None:
        from lf_diagnostics import fixed_preview_indices, save_gain_preview
        if lf_train_mean is None:
            raise ValueError("LF validation needs the fixed training-only G mean")
        preview_indices = fixed_preview_indices(len(loader.dataset), preview_count)
    for low, target in loader:
        low, target = low.to(device), target.to(device)
        aux = model(low, return_aux=True) if model.lf_branch is not None else None
        output = (aux["output"] if aux is not None else model(low)).float()  # FP32 validation.
        if not torch.isfinite(output).all():
            raise RuntimeError("Non-finite validation output.")
        if aux is not None:
            operator = model.lf_branch.operator
            gain_target = operator.target(low, target)
            predicted = aux["lf_gain"]
            if not torch.isfinite(predicted).all():
                raise RuntimeError("Non-finite validation G prediction")
            low_luma, gt_luma = operator.luminance(low), operator.luminance(target)
            values = {
                "val_lf_gain_mae": (predicted - gain_target).abs().flatten(1).mean(1),
                "val_lf_mean_baseline_mae": (gain_target - lf_train_mean).abs().flatten(1).mean(1),
                "val_lf_zero_baseline_mae": gain_target.abs().flatten(1).mean(1),
                "val_lf_luma_mse": (operator(operator.luminance(output)) - operator(gt_luma)).square().flatten(1).mean(1),
            }
            for key, value in values.items():
                lf_sums[key] = lf_sums.get(key, 0.0) + value.sum().item()
            for key, value in aux["lf_diagnostics"].items():
                key = "val_" + key
                lf_sums[key] = lf_sums.get(key, 0.0) + value.item() * low.shape[0]
            if preview_dir is not None:
                for offset in range(low.shape[0]):
                    index = count + offset
                    if index in preview_indices:
                        name = loader.dataset.low[index].name
                        save_gain_preview(preview_dir / f"{index:04d}_{Path(name).stem}.png",
                                          low_luma[offset], gain_target[offset], predicted[offset],
                                          gt_luma[offset], name)
        l1 = (output - target).abs().flatten(1).mean(1)
        mse = (output.clamp(0, 1) - target).square().flatten(1).mean(1)
        psnr = -10.0 * torch.log10(mse.clamp_min(1e-12))
        l1_sum += l1.sum().item()
        psnr_sum += psnr.sum().item()
        if gt_mean_fft_l1 is not None:
            image_loss, _ = gt_mean_fft_l1(
                output, target, sigma=gt_mean_sigma, fft_weight=fft_weight)
            if not math.isfinite(float(image_loss.detach())):
                raise RuntimeError("Non-finite validation loss.")
            # Batch mean times batch size keeps each full image equally weighted.
            loss_sum += float(image_loss.detach()) * low.shape[0]
        ssim_sum += _uint8_wang_ssim(output, target) * low.shape[0]
        count += low.shape[0]
    model.train()
    metrics = {"val_count": count, "val_l1": l1_sum / count,
               "val_psnr_rgb_float": psnr_sum / count,
               "val_ssim": ssim_sum / count}
    if gt_mean_fft_l1 is not None:
        metrics["val_loss"] = loss_sum / count
    metrics.update({key: value / count for key, value in lf_sums.items()})
    return metrics


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
    if not amp:
        # Use full FP32 arithmetic, including CUDA convolutions and matmul.
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    model_config = dict(width=args.width, spectral_mode=args.spectral_mode, output_mode=args.output_mode)
    if args.fusion_mode != "gated":
        model_config["fusion_mode"] = args.fusion_mode
    if args.bottleneck_attention != "none":
        model_config["bottleneck_attention"] = args.bottleneck_attention
    if args.half_decoder != "gated":
        model_config["half_decoder"] = args.half_decoder
    if args.half_encoder_frequency != "none":
        model_config["half_encoder_frequency"] = args.half_encoder_frequency
    if args.bottleneck_prior != "none":
        model_config["bottleneck_prior"] = args.bottleneck_prior
    if args.illumination_guidance != "none":
        model_config["illumination_guidance"] = args.illumination_guidance
    if args.star_refinement != "none":
        model_config["star_refinement"] = args.star_refinement
    if args.star_variant != "original":
        model_config["star_variant"] = args.star_variant
    if args.lf_guidance != "none":
        model_config["lf_guidance"] = args.lf_guidance
    if args.wavelet_variant != "none":
        model_config["wavelet_variant"] = args.wavelet_variant
    model_config["experiment_variant"] = MODEL_SPEC["variant"]
    validate_model_config(model_config)
    dataset = PairedImages(data, "train", crop_size=args.crop_size, seed=args.seed,
                           flip_hv=args.flip_hv, geometric_augs=args.geometric_augs,
                           saturation_probability=args.saturation_probability,
                           saturation_range=args.saturation_range)
    steps_per_epoch = math.ceil(len(dataset) / args.batch_size)
    if args.epochs:
        args.steps = args.epochs * steps_per_epoch
    config = {
        "model": model_config, "data_root": str(data), "batch_size": args.batch_size,
        "seed": args.seed, "lr": args.lr, "weight_decay": args.weight_decay,
        "lr_schedule": args.lr_schedule, "min_lr": args.min_lr,
        **({"stage_steps": args.stage_steps, "stage2_lr": args.stage2_lr,
            "stage2_min_lr": args.stage2_min_lr,
            "lr_policy": (f"steps 1-{args.stage_steps}: cosine {args.lr:g} to {args.min_lr:g}; "
                          f"steps {args.stage_steps + 1}-{args.steps}: cosine {args.stage2_lr:g} to {args.stage2_min_lr:g}")}
           if args.lr_schedule == "two-stage-cosine" else {}),
        "amp": amp, "precision": "AMP-FP16" if amp else "FP32",
        "tf32_matmul": bool(torch.backends.cuda.matmul.allow_tf32),
        "tf32_cudnn": bool(torch.backends.cudnn.allow_tf32),
        "grad_clip": args.grad_clip, "steps": args.steps,
        "workers": args.workers, "val_every": args.val_every, "save_every": args.save_every,
        "device": str(device), "loss": "L1 on unclamped encoded RGB",
        "loss_mode": args.loss_mode,
        "epochs": args.epochs or None, "steps_per_epoch": steps_per_epoch,
        "crop_size": list(args.crop_size) if args.crop_size else None,
        "horizontal_flip_probability": 0.5 if (args.crop_size or args.flip_hv) else 0.0,
        "vertical_flip_probability": 0.5 if (args.crop_size or args.flip_hv) else 0.0,
        "flip_hv": bool(args.flip_hv),
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
    config.update(
        geometric_augs=bool(args.geometric_augs),
        saturation_probability=args.saturation_probability,
        saturation_range=list(args.saturation_range),
        saturation_recipe="paired-torchvision-saturation-independent-image-epoch-v1",
        saturation_policy="train only; same factor for LQ/GT; val/test unchanged",
        initialization="from scratch; shared A+MDTA seeded layers preserved",
    )
    if args.geometric_augs:
        image_policy = (f"paired random crop {args.crop_size[0]}x{args.crop_size[1]}"
                        if args.crop_size else "whole image")
        config.update(
            geometric_recipe="d4-batch-quarterturn-image-hv-v1",
            d4_transform_probability=0.125, d4_quarter_turn_probability=0.5,
            d4_batch_policy="shared 90-degree rotation bit; independent image horizontal/vertical flips",
            image_policy=f"train: {image_policy}, paired uniform D4; val/infer: whole image",
        )
    config["image_policy"] += (f"; train saturation p={args.saturation_probability}, "
                               f"factor range={list(args.saturation_range)}")
    if args.star_refinement != "none":
        config.update(
            star_recipe=args.star_refinement,
            star_bottleneck_blocks=2,
            star_ffn_expansion=3.0,
            star_stage_sites=(["after encoder0", "after encoder1", "after decoder1", "after decoder0"]
                              if args.star_refinement == "multiscale" else []),
            star_replacement="entire original Spatial Fusion replaced by two full StarBlocks",
            star_bottleneck_order="encoder2 -> StarBlock -> StarBlock -> unchanged MDTA",
            star_supervision="configured final RGB output loss",
        )
    if args.star_variant == "multiscale-amp":
        from multiscale_amp_blocks import MULTISCALE_AMP_RECIPE
        config.update(star_block_recipe=dict(MULTISCALE_AMP_RECIPE),
                      star_bottleneck_order="encoder2 -> new StarBlock -> new StarBlock -> unchanged MDTA")
    if args.wavelet_variant != "none":
        from wavelet_blocks import wavelet_recipe
        config["wavelet_recipe"] = wavelet_recipe(args.wavelet_variant)
        if args.wavelet_variant in {"encoder-mdta", "encoder-mdta-low-scale", "encoder-context-mdta"}:
            config["star_bottleneck_order"] = "wavelet encoder2 -> StarBlock -> StarBlock -> WaveletMDTAResidual"
        elif args.wavelet_variant == "mdta-only":
            config["star_bottleneck_order"] = "ordinary encoder2 -> StarBlock -> StarBlock -> WaveletMDTAResidual"
        elif args.wavelet_variant in FREQUENCY_VARIANTS:
            config["star_bottleneck_order"] = "multiscale frequency encoder2 -> StarBlock -> StarBlock -> WaveletMDTAResidual"
            if args.wavelet_variant in ("all-multiscale-mdta", *CANDIDATE_VARIANTS):
                config["decoder_frequency_order"] = {
                    "decoder1": "fuse1 -> multiscale frequency decoder1 -> original StarBlock",
                    "decoder0": "fuse0 -> multiscale frequency decoder0 -> original StarBlock",
                }
    if args.wavelet_variant == "all-shared-dffn-mdta":
        config["star_bottleneck_order"] = "multiscale frequency encoder2 -> StarBlock(shared spatial DFFN) x2 -> WaveletMDTAResidual"
        config["decoder_frequency_order"] = {
            "decoder1": "fuse1 -> frequency decoder1 -> StarBlock(shared spatial DFFN)",
            "decoder0": "fuse0 -> frequency decoder0 -> StarBlock(shared spatial DFFN)",
        }
    from structure_candidate_blocks import STRUCTURE_VARIANTS, structure_recipe
    if args.wavelet_variant in STRUCTURE_VARIANTS:
        recipe = structure_recipe(args.wavelet_variant)
        config["structure_recipe"] = recipe
        config["star_bottleneck_order"] = recipe["bottom_order"]
        if args.wavelet_variant == STRUCTURE_VARIANTS[3]:
            config["encoder_frequency_order"] = {"encoder1": recipe["encoder1_order"]}
            config["decoder_frequency_order"]["decoder1"] = recipe["decoder1_order"]
    if args.loss_mode == "ms-l1":
        config.update(
            loss="(L1_full + 0.5 * L1_half + 0.25 * L1_quarter) / 1.75 on unclamped encoded RGB",
            loss_scales=[1, 2, 4], loss_weights=[1.0, 0.5, 0.25],
            loss_normalizer=1.75, loss_downsample="avg_pool2d; kernel=stride=scale; FP32",
        )
    if args.loss_mode == "l1-fft":
        from frequency_loss import FFT_LOSS_RECIPE, pixel_fft_l1
        config.update(
            loss="raw RGB L1 + fft_loss_weight * complex FFT L1; no GT-Mean alignment",
            fft_loss_weight=args.fft_weight, fft_loss_recipe=dict(FFT_LOSS_RECIPE),
        )
    if args.loss_mode in {"gt-mean-l1", "gt-mean-fft"}:
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
    if args.loss_mode == "gt-mean-fft":
        from frequency_loss import FFT_LOSS_RECIPE, gt_mean_fft_l1
        config.update(
            loss="GT-Mean L1 + fft_loss_weight * complex FFT L1 on raw RGB",
            fft_loss_weight=args.fft_weight, fft_loss_recipe=dict(FFT_LOSS_RECIPE),
            fft_alignment="raw prediction and GT; no GT-Mean alignment in FFT term",
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
            lfpv_channels=64, lfpv_vectors=16, lfpv_patches=16, lfpv_patch_size=4,
            lfpv_updater_layers=7, lfpv_updater_hidden="4c x5, c x1; final c",
            lfpv_updater_activation="BatchNorm + ReLU; follows published code",
            lfpv_training="shared encoder/decoder; bypass and SU+MU+query outputs",
            lfpv_loss_weights=[0.5, 0.5],
            lfpv_precision="FP32 SU/MU/query/reference buffers; AMP backbone/projections",
            lfpv_reference_update="detached updater output committed after successful optimizer step",
            lfpv_eval="query saved references only; no SU/MU or reference updates",
            lfpv_source="https://github.com/xiaogang00/LFPVS_ICCV",
        )
        config["loss"] = "0.5 * GT-Mean L1(bypass) + 0.5 * GT-Mean L1(LFPV-guided)"
    if args.lf_guidance != "none":
        config.update(
            lf_recipe="supervised-log-luminance-gain-v1",
            lf_loss_weight=args.lf_loss_weight,
            lf_gaussian_sigma=2.0, lf_gaussian_radius=6, lf_downsample_factor=8,
            lf_target_eps=1 / 255, lf_luminance_weights=[0.2126, 0.7152, 0.0722],
            lf_target="log((D8(Y_GT)+eps)/(D8(Y_low)+eps)); after paired augmentation",
            lf_operator="FP32 separable Gaussian reflect (replicate for tiny inputs), then area to max(1, floor(HW/8))",
            lf_branch="blur/downsample low RGB -> Conv3x3 3:32 -> GCB32 x3 -> Conv1x1 32:1; unbounded G",
            lf_film="G -> Conv1x1 1:16 -> GELU -> Conv1x1 16:2C -> bilinear resize; F*(1+alpha)+beta",
            lf_sites=(["after fuse1, before decoder1"] if args.lf_guidance == "decoder1"
                      else ["after fuse1, before decoder1", "after fuse0, before decoder0"]),
            lf_initialization="all shared Star-Multiscale-A draws preserved; only final FiLM projections zero",
            lf_baseline_policy="fixed image-equal mean G over unaugmented training pairs; saved and reused on resume",
            lf_preview_count=args.lf_preview_count,
            lf_preview_scale=[-math.log(32), math.log(32)],
        )
        config["loss"] += f" + {args.lf_loss_weight} * MAE(G,G_target)"
        config["star_supervision"] = "final RGB restoration loss plus supervised LF G MAE"
    # These recipes describe installed runtime modules, replacing legacy construction metadata.
    config["experiment_recipe"] = MODEL_SPEC["recipe"]
    config["wavelet_recipe"] = MODEL_SPEC["recipe"]
    config["structure_recipe"] = MODEL_SPEC["recipe"]
    config["initialization"] = MODEL_SPEC["recipe"]["initialization"]
    config["star_bottleneck_order"] = MODEL_SPEC["recipe"]["bottom_order"]
    recipe = MODEL_SPEC["recipe"]
    config["decoder_frequency_order"] = recipe["decoder_frequency_order"]
    config["original_star_blocks"] = recipe["original_star_blocks"]
    config["own_block_count"] = recipe["own_block_count"]
    config["mdta_blocks"] = recipe["mdta_count"]
    config["dffn_blocks"] = recipe["dffn_count"]
    config["star_recipe"] = recipe["star_blocks"]
    config["star_bottleneck_blocks"] = 2 if recipe["original_star_blocks"] else 0
    config["star_stage_sites"] = (["after encoder0", "after encoder1", "after decoder1", "after decoder0"]
                                  if recipe["original_star_blocks"] else [])
    config["star_replacement"] = recipe["intervention"]
    if not recipe["dffn_count"]:
        config.pop("star_ffn_expansion", None)
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
        previous.setdefault("saturation_probability", 0.0)
        previous.setdefault("saturation_range", [0.9, 1.1])
        keys = ["model", "data_root", "batch_size", "seed", "lr", "weight_decay", "amp", "grad_clip", "lr_schedule", "best_metric", "crop_size", "flip_hv", "loss_mode"]
        keys.extend(["geometric_augs", "saturation_probability", "saturation_range", "experiment_recipe"])
        if args.geometric_augs:
            keys.extend(["geometric_recipe", "d4_transform_probability", "d4_batch_policy"])
        if args.loss_mode in {"gt-mean-l1", "gt-mean-fft"}:
            keys.extend(["gt_mean_sigma", "gt_mean_eps"])
        if args.loss_mode in {"l1-fft", "gt-mean-fft"}:
            keys.extend(["fft_loss_weight", "fft_loss_recipe"])
        if args.loss_mode == "gt-mean-fft":
            keys.append("fft_alignment")
        if args.star_variant == "multiscale-amp":
            keys.append("star_block_recipe")
        if args.wavelet_variant != "none":
            keys.append("wavelet_recipe")
        if args.wavelet_variant in STRUCTURE_VARIANTS:
            keys.append("structure_recipe")
        if args.illumination_guidance in {"three-stage", "bottom-v"}:
            keys.extend(["illumination_recipe", "illumination_initialization", "illumination_precision"])
        if args.bottleneck_prior == "lfpv":
            keys.extend(["lfpv_loss_weights", "lfpv_precision", "lfpv_reference_update"])
        if args.lf_guidance != "none":
            keys.extend(["lf_recipe", "lf_loss_weight", "lf_gaussian_sigma", "lf_gaussian_radius",
                         "lf_downsample_factor", "lf_target_eps", "lf_luminance_weights"])
        if args.lr_schedule == "cosine":
            keys.extend(["min_lr", "steps"])
        if args.lr_schedule == "two-stage-cosine":
            keys.extend(["min_lr", "steps", "stage_steps", "stage2_lr", "stage2_min_lr"])
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
    model = EnhancementDemo(**model_config).to(device=device, dtype=torch.float32)
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
    lf_train_mean = None
    if model.lf_branch is not None:
        if checkpoint is not None:
            lf_train_mean = checkpoint["config"]["lf_train_gain_mean"]
        else:
            from lf_diagnostics import training_gain_mean
            print(json.dumps({"lf_baseline": "computing from train pairs only", "train_images": len(dataset)}), flush=True)
            baseline_loader = DataLoader(PairedImages(data, "train"), batch_size=1, shuffle=False,
                                         num_workers=args.workers,
                                         generator=torch.Generator().manual_seed(args.seed + 300000))
            lf_train_mean = training_gain_mean(model.lf_branch.operator, baseline_loader, device)
        config["lf_train_gain_mean"] = lf_train_mean
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
    if not args.resume:
        snapshot = run / "code"
        snapshot.mkdir()
        for name in ("models", "training", "evaluation", "verification", "configs", "docs"):
            shutil.copytree(PACKAGE_ROOT / name, snapshot / name,
                            ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "reports"))
        for name in ("README.md", "requirements.txt"):
            shutil.copyfile(PACKAGE_ROOT / name, snapshot / name)
    (run / "config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"run": str(run), "device": str(device), "amp": amp,
                      "train_images": len(dataset), "epochs": args.epochs or None,
                      "steps_per_epoch": steps_per_epoch, "crop_size": args.crop_size,
                      "start_step": start_step,
                      "stop_step": stop_step, "target_step": args.steps, "test_images_read": 0}), flush=True)
    write_status(run, "running", start_step, args.steps)
    model.train()
    with (run / "loss.jsonl").open("a", encoding="utf-8") as log:
        for step, (low, target) in enumerate(loader, start=start_step + 1):
            started = time.perf_counter()
            for group in optimizer.param_groups:
                group["lr"] = learning_rate(config, step)
            low, target = low.to(device), target.to(device)
            optimizer.zero_grad(set_to_none=True)
            lfpv_paths = None
            lf_aux = None
            with torch.autocast(device_type=device.type, enabled=amp, dtype=torch.float16):
                if args.bottleneck_prior == "lfpv":
                    lfpv_paths = model(low, lfpv_training=True)
                    output = lfpv_paths["output"]
                elif model.lf_branch is not None:
                    lf_aux = model(low, return_aux=True)
                    output = lf_aux["output"]
                else:
                    output = model(low)
                if args.loss_mode == "l1":
                    loss = torch.nn.functional.l1_loss(output.float(), target)
            if args.loss_mode == "ms-l1":
                loss, scale_losses = multiscale_l1(output, target)
            elif args.loss_mode == "l1-fft":
                loss, fft_diagnostics = pixel_fft_l1(output, target, fft_weight=args.fft_weight)
            elif args.loss_mode == "gt-mean-fft":
                loss, combined_diagnostics = gt_mean_fft_l1(
                    output, target, sigma=args.gt_mean_sigma, fft_weight=args.fft_weight)
            elif args.loss_mode == "gt-mean-l1":
                loss, gt_mean_diagnostics = gt_mean_l1(output, target, sigma=args.gt_mean_sigma)
                if lfpv_paths is not None:
                    refined_loss = loss
                    base_loss, _ = gt_mean_l1(lfpv_paths["base_output"], target, sigma=args.gt_mean_sigma)
                    loss = 0.5 * base_loss + 0.5 * refined_loss
            if lf_aux is not None:
                restoration_loss = loss
                gain_target = model.lf_branch.operator.target(low, target)
                gain_loss = F.l1_loss(lf_aux["lf_gain"], gain_target)
                weighted_gain_loss = args.lf_loss_weight * gain_loss
                loss = restoration_loss + weighted_gain_loss
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
            if args.loss_mode == "gt-mean-l1":
                row.update({key: value.item() for key, value in gt_mean_diagnostics.items()})
            if args.loss_mode == "l1-fft":
                row.update({key: value.item() for key, value in fft_diagnostics.items()})
                row["fft_loss_weight"] = args.fft_weight
            if args.loss_mode == "gt-mean-fft":
                row.update({key: value.item() for key, value in combined_diagnostics.items()})
                row["fft_loss_weight"] = args.fft_weight
            if lf_aux is not None:
                row.update(loss_restoration=restoration_loss.item(), loss_lf_gain=gain_loss.item(),
                           loss_lf_gain_weighted=weighted_gain_loss.item(), lf_loss_weight=args.lf_loss_weight,
                           lf_gain_pred_mean=lf_aux["lf_gain"].detach().mean().item(),
                           lf_gain_pred_std=lf_aux["lf_gain"].detach().std(unbiased=False).item(),
                           lf_gain_target_mean=gain_target.mean().item(), lf_train_gain_mean=lf_train_mean)
                row.update({key: value.item() for key, value in lf_aux["lf_diagnostics"].items()})
            if lfpv_paths is not None:
                core = model.lfpv.core
                row.update(lfpv_bypass_loss=base_loss.item(), lfpv_guided_loss=refined_loss.item(),
                           lfpv_vector_std=core.common_feature.std(unbiased=False).item(),
                           lfpv_patch_std=core.common_feature_patch.std(unbiased=False).item())
            improved = False
            if val_loader is not None and (step % args.val_every == 0 or step == args.steps):
                preview_dir = run / "lf_previews" / f"step_{step:06d}" if lf_aux is not None else None
                row.update(validate(model, val_loader, device, lf_train_mean=lf_train_mean,
                                    preview_dir=preview_dir, preview_count=args.lf_preview_count,
                                    loss_mode=args.loss_mode, gt_mean_sigma=args.gt_mean_sigma,
                                    fft_weight=args.fft_weight))
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
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
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
    model = EnhancementDemo(**checkpoint["config"]["model"]).to(device)
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
                          help="Uniform paired D4; shared rectangle orientation bit per batch")
    training.add_argument("--saturation-probability", type=float, default=0.0)
    training.add_argument("--saturation-range", type=float, nargs=2, default=[0.9, 1.1])
    training.add_argument("--star-refinement", choices=["none", "bottleneck-two", "multiscale"],
                          default="none")
    training.add_argument("--star-variant", choices=["original", "multiscale-amp"],
                          default="original", help="Replace all six blocks with multiscale amplitude StarBlocks")
    training.add_argument("--wavelet-variant", choices=WAVELET_VARIANTS,
                          default="none", help="Replace ordinary gates; optionally replace bottom MDTA")
    training.add_argument("--lf-guidance", choices=["none", "decoder1", "decoder1-decoder0"], default="none",
                          help="Supervised 1/8-resolution G branch on Star-Multiscale-A; default injection decoder1")
    training.add_argument("--lf-loss-weight", type=float, default=0.05)
    training.add_argument("--lf-preview-count", type=int, default=4,
                          help="Fixed evenly spaced val examples at each validation; 0 disables previews")
    training.add_argument("--batch-size", type=int, default=2)
    training.add_argument("--lr", type=float, default=2e-4)
    training.add_argument("--lr-schedule", choices=["constant", "cosine", "two-stage-cosine"], default="constant")
    training.add_argument("--min-lr", type=float, default=2e-6)
    training.add_argument("--stage-steps", type=int, default=0,
                          help="First-stage length for two-stage-cosine; that step still uses the first cosine")
    training.add_argument("--stage2-lr", type=float, default=1e-4)
    training.add_argument("--stage2-min-lr", type=float, default=2e-6)
    training.add_argument("--stop-after", type=int, default=0,
                          help="Run a bounded number of additional steps without changing the LR horizon")
    training.add_argument("--weight-decay", type=float, default=1e-4)
    training.add_argument("--grad-clip", type=float, default=1.0)
    training.add_argument("--seed", type=int, default=100)
    training.add_argument("--workers", type=int, default=0)
    training.add_argument("--width", type=int, default=24)
    training.add_argument("--spectral-mode", choices=["conditional", "static", "off"], default="conditional")
    training.add_argument("--bottleneck-attention", choices=["none", "mdta"], default="none")
    training.add_argument("--half-decoder", choices=["gated", "mdta-gated", "restormer"],
                          default="gated", help="Half decoder: original, MDTA plus original, or full Restormer replacement")
    training.add_argument("--half-encoder-frequency", choices=["none", "fremlp"],
                          default="none", help="Append a zero-initialized FreMLP modulation after encoder1")
    training.add_argument("--bottleneck-prior", choices=["none", "lfpv"], default="none",
                          help="Full LFPV core with shared-decoder dual-path GT-Mean training")
    training.add_argument("--illumination-guidance", choices=["none", "three-stage", "bottom-v"], default="none",
                          help="Independent low-image brightness branch: three sites or bottom MDTA V only")
    training.add_argument("--fusion-mode", choices=["gated", "additive", "none", "star"],
                          default="gated",
                          help="Bottleneck fusion; none removes it, star replaces it with one full StarBlock.")
    training.add_argument("--output-mode", choices=["structured", "direct", "rgb_interaction"],
                          default="structured",
                          help="Output parameterization. rgb_interaction is the per-colour gated residual head.")
    training.add_argument("--loss-mode", choices=["l1", "ms-l1", "gt-mean-l1", "l1-fft", "gt-mean-fft"], default="l1",
                          help="RGB L1, multi-scale L1, GT-Mean L1, or either pixel loss plus complex FFT L1")
    training.add_argument("--fft-weight", type=float, default=0.1,
                          help="Frequency loss coefficient for l1-fft or gt-mean-fft (default: 0.1)")
    training.add_argument("--gt-mean-sigma", type=float, default=0.1,
                          help="Brightness distribution spread for GT-Mean L1")
    training.add_argument("--amp", action=argparse.BooleanOptionalAction, default=False,
                          help="Enable legacy AMP explicitly; default is full FP32")
    training.add_argument("--device", default="auto")
    training.add_argument("--resume", type=Path)
    training.add_argument("--val-every", type=int, default=0,
                          help="0 disables validation; positive values evaluate full val at this interval")
    training.add_argument("--best-metric", choices=["l1", "psnr"], default="l1",
                          help="Metric that decides best_val.pt independently of the training loss mode.")
    training.add_argument("--save-every", type=int, default=100)
    training.add_argument("--log-every", type=int, default=10)
    training.set_defaults(
        function=train, run_dir=PACKAGE_ROOT / "outputs" / "train", steps=100000,
        batch_size=8, crop_size=[128, 128], workers=4, lr_schedule="cosine",
        val_every=500, save_every=500, log_every=50, geometric_augs=True,
        saturation_probability=0.25, saturation_range=[0.9, 1.1],
        spectral_mode="off", output_mode="direct", star_refinement="multiscale",
        bottleneck_attention="mdta", wavelet_variant="all-subband-first-mdta", loss_mode="gt-mean-fft",
    )
    inference = commands.add_parser("infer", help="Infer images using a trained checkpoint")
    inference.add_argument("--checkpoint", type=Path, required=True)
    inference.add_argument("--input", type=Path, required=True)
    inference.add_argument("--output-dir", type=Path, required=True)
    inference.add_argument("--device", default="auto")
    inference.set_defaults(function=infer)
    args = parser.parse_args()
    if args.command == "train":
        if args.amp:
            parser.error("This experiment uses full FP32; use --no-amp.")
        fixed = dict(width=24, spectral_mode="off", output_mode="direct", fusion_mode="gated",
                     bottleneck_attention="mdta", star_refinement="multiscale", star_variant="original",
                     wavelet_variant="all-subband-first-mdta", half_decoder="gated", half_encoder_frequency="none",
                     bottleneck_prior="none", illumination_guidance="none", lf_guidance="none",
                     loss_mode="gt-mean-fft", fft_weight=0.1, gt_mean_sigma=0.1)
        if any(getattr(args, key) != value for key, value in fixed.items()):
            parser.error("This package fixes its experiment architecture and GT-Mean + 0.1 FFT loss.")
        if args.wavelet_variant != "none" and (
                args.star_variant != "original" or args.star_refinement != "multiscale"
                or args.spectral_mode != "off" or args.output_mode != "direct"
                or args.fusion_mode != "gated" or args.bottleneck_attention != "mdta"
                or args.half_decoder != "gated" or args.half_encoder_frequency != "none"
                or args.bottleneck_prior != "none" or args.illumination_guidance != "none"
                or args.lf_guidance != "none" or args.loss_mode != "gt-mean-fft"
                or args.fft_weight != 0.1 or args.gt_mean_sigma != 0.1):
            parser.error("wavelet experiments require the unchanged six-Star direct A+MDTA flags and GT-Mean L1 + 0.1 * FFT L1")
        if args.star_variant == "multiscale-amp" and (args.star_refinement != "multiscale"
                                                    or args.lf_guidance != "none" or args.width % 2):
            parser.error("multiscale-amp requires --star-refinement multiscale, --lf-guidance none, and even width")
        for name in ("steps", "batch_size", "lr", "grad_clip", "width", "save_every", "log_every"):
            if getattr(args, name) <= 0:
                parser.error(f"--{name.replace('_', '-')} must be positive")
        if args.bottleneck_prior == "lfpv" and args.loss_mode != "gt-mean-l1":
            parser.error("LFPV experiment requires --loss-mode gt-mean-l1")
        if not math.isfinite(args.lf_loss_weight) or args.lf_loss_weight < 0 or args.lf_preview_count < 0:
            parser.error("lf-loss-weight must be finite/nonnegative; lf-preview-count must be nonnegative")
        if args.lf_guidance != "none" and args.star_refinement != "multiscale":
            parser.error("LF guidance requires --star-refinement multiscale and its unchanged A flags")
        if args.loss_mode in {"gt-mean-l1", "gt-mean-fft"} and args.gt_mean_sigma <= 0:
            parser.error("--gt-mean-sigma must be positive")
        if args.loss_mode in {"l1-fft", "gt-mean-fft"} and (not math.isfinite(args.fft_weight) or args.fft_weight < 0):
            parser.error("--fft-weight must be finite and nonnegative")
        if not 0 <= args.saturation_probability <= 1 or not (0 < args.saturation_range[0] <= args.saturation_range[1]):
            parser.error("Saturation probability must be [0,1] and factor range positive/ordered")
        if args.epochs < 0 or (args.crop_size is not None and min(args.crop_size) <= 0):
            parser.error("epochs must be nonnegative and crop dimensions positive")
        if args.workers < 0 or args.val_every < 0 or args.weight_decay < 0:
            parser.error("workers, val-every and weight-decay must be nonnegative")
        if args.stop_after < 0 or not 0 <= args.min_lr <= args.lr:
            parser.error("stop-after must be nonnegative; min-lr must be between 0 and lr")
        if args.lr_schedule == "two-stage-cosine" and (
                not 1 <= args.stage_steps < args.steps or not 0 <= args.stage2_min_lr <= args.stage2_lr):
            parser.error("two-stage-cosine needs 1 <= stage-steps < steps and 0 <= stage2-min-lr <= stage2-lr")
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
