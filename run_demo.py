"""Minimal whole-image paired training and checkpoint-only inference.

Training reads train and optionally val; it never opens test. RGB values stay
in encoded [0, 1] space, with no resizing, cropping or automatic white balance.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import json
import math
import os
from pathlib import Path
import random
import tempfile
import time

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset, Sampler

from model import EnhancementDemo


DEFAULT_DATA = Path(r"M:\picture data\cholec80_t\train_test")
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def read_rgb(path: Path) -> torch.Tensor:
    with Image.open(path) as image:
        array = np.asarray(image.convert("RGB"), dtype=np.float32).copy() / 255.0
    return torch.from_numpy(array).permute(2, 0, 1)


def image_files(directory: Path) -> list[Path]:
    return sorted(p for p in directory.iterdir()
                  if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)


class PairedImages(Dataset):
    def __init__(self, root: Path, split: str):
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

    def __getitem__(self, index):
        low, gt = read_rgb(self.low[index]), read_rgb(self.gt[index])
        if low.shape != gt.shape:
            raise ValueError(f"Mismatched paired image shapes: {self.low[index]}")
        # Full images are deliberate: global colour statistics must match
        # the full-image input used at validation and inference.
        return low, gt


class StepBatches(Sampler):
    """Recreate each epoch's shuffle from seed; resumes at an exact batch.

    Dataset loading has no random augmentation, so worker prefetch does not
    change the sample sequence. A final short batch is retained each epoch.
    """
    def __init__(self, count: int, batch: int, seed: int, start: int, stop: int):
        self.count, self.batch, self.seed = count, batch, seed
        self.start, self.stop = start, stop

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
            yield order[offset:offset + self.batch]


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
    config = {
        "model": model_config, "data_root": str(data), "batch_size": args.batch_size,
        "seed": args.seed, "lr": args.lr, "weight_decay": args.weight_decay,
        "lr_schedule": args.lr_schedule, "min_lr": args.min_lr,
        "amp": amp, "grad_clip": args.grad_clip, "steps": args.steps,
        "workers": args.workers, "val_every": args.val_every, "save_every": args.save_every,
        "device": str(device), "loss": "L1 on unclamped encoded RGB",
        "image_policy": "whole image, no resize/crop/augmentation",
        "val_policy": "image-equal raw L1 and clamped RGB float PSNR (MSE floor 1e-12); no GT mean",
        "resume_policy": "exact epoch/batch shuffle and saved RNG; backend bitwise determinism not guaranteed",
    }
    checkpoint = None
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        previous = dict(checkpoint["config"])
        previous.setdefault("lr_schedule", "constant")
        previous.setdefault("min_lr", 2e-6)
        keys = ["model", "data_root", "batch_size", "seed", "lr", "weight_decay", "amp", "grad_clip", "lr_schedule"]
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
    model = EnhancementDemo(**model_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=amp, init_scale=1024.0)
    start_step, best_val = 0, None
    if checkpoint is not None:
        model.load_state_dict(checkpoint["model"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        scaler.load_state_dict(checkpoint["scaler"])
        start_step, best_val = checkpoint["step"], checkpoint.get("best_val")
        restore_rng(checkpoint["rng"])
    dataset = PairedImages(data, "train")
    names = [p.name for p in dataset.low]
    if checkpoint is not None and checkpoint["train_filenames"] != names:
        raise ValueError("Training filename list changed since checkpoint.")
    stop_step = min(args.steps, start_step + args.stop_after) if args.stop_after else args.steps
    sampler = StepBatches(len(dataset), args.batch_size, args.seed, start_step, stop_step)
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
    (run / "config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"run": str(run), "device": str(device), "amp": amp,
                      "parameters": sum(p.numel() for p in model.parameters()),
                      "train_images": len(dataset), "start_step": start_step,
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
            with torch.autocast(device_type=device.type, enabled=amp, dtype=torch.float16):
                output = model(low)
                loss = torch.nn.functional.l1_loss(output.float(), target)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss at step {step}.")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip,
                                                       error_if_nonfinite=not amp)
            previous_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            row = {"step": step, "loss": loss.item(),
                   "grad_norm_before_clip": float(grad_norm) if torch.isfinite(grad_norm) else None,
                   "amp_skipped_step": bool(amp and scaler.get_scale() < previous_scale),
                   "amp_scale": scaler.get_scale(),
                   "seconds": time.perf_counter() - started, "lr": optimizer.param_groups[0]["lr"]}
            improved = False
            if val_loader is not None and (step % args.val_every == 0 or step == args.steps):
                row.update(validate(model, val_loader, device))
                if best_val is None or row["val_l1"] < best_val:
                    best_val, improved = row["val_l1"], True
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
                 best_val_l1=best_val,
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
    training = commands.add_parser("train", help="Whole-image paired training; no test access")
    training.add_argument("--data-root", type=Path, default=DEFAULT_DATA)
    training.add_argument("--run-dir", type=Path, default=Path(__file__).parent / "runs" / "demo")
    training.add_argument("--steps", type=int, default=1000)
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
    training.add_argument("--output-mode", choices=["structured", "direct"], default="structured")
    training.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    training.add_argument("--device", default="auto")
    training.add_argument("--resume", type=Path)
    training.add_argument("--val-every", type=int, default=0,
                          help="0 disables validation; positive values evaluate full val at this interval")
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
