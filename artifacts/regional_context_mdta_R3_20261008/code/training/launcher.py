"""Training CLI for the controlled experiment in this folder."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = json.loads((PACKAGE_ROOT / "configs" / "training.json").read_text(encoding="utf-8"))
SPEC = json.loads((PACKAGE_ROOT / "configs" / "model.json").read_text(encoding="utf-8"))
DEFAULT_DATA = (Path(r"M:\picture data\cholec80_t\train_test") if os.name == "nt"
                else Path("/root/autodl-tmp/datasets/cholec80/train_test"))


def training_command(args):
    command = [
        sys.executable, "-B", "-u", "-X", "utf8", str(PACKAGE_ROOT / "training" / "engine.py"), "train",
        "--data-root", str(args.data_root), "--run-dir", str(args.run_dir),
        "--steps", str(args.steps), "--epochs", "0", "--batch-size", str(args.batch_size),
        "--seed", str(args.seed), "--workers", str(args.workers),
        "--lr", str(args.lr), "--lr-schedule", args.lr_schedule, "--min-lr", str(args.min_lr),
        "--weight-decay", str(PROTOCOL["weight_decay"]), "--grad-clip", str(PROTOCOL["grad_clip"]),
        "--val-every", str(args.val_every), "--save-every", str(args.save_every), "--log-every", str(args.log_every),
        "--best-metric", "l1", "--width", "24", "--spectral-mode", "off", "--output-mode", "direct",
        "--fusion-mode", "gated", "--bottleneck-attention", "mdta", "--star-refinement", "multiscale",
        "--star-variant", "original", "--wavelet-variant", "all-subband-first-mdta", "--lf-guidance", "none",
        "--loss-mode", "gt-mean-fft", "--fft-weight", "0.1", "--gt-mean-sigma", "0.1",
        "--crop-size", *(str(size) for size in args.crop_size), "--geometric-augs",
        "--saturation-probability", str(args.saturation_probability),
        "--saturation-range", *(str(value) for value in args.saturation_range),
        "--device", args.device, "--amp" if args.amp else "--no-amp",
        "--stop-after", str(args.stop_after),
    ]
    if args.lr_schedule == "two-stage-cosine":
        command.extend(["--stage-steps", str(args.stage_steps), "--stage2-lr", str(args.stage2_lr),
                        "--stage2-min-lr", str(args.stage2_min_lr)])
    if args.resume is not None:
        command.extend(["--resume", str(args.resume)])
    return command


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--run-dir", type=Path, default=PACKAGE_ROOT / "outputs" / "train")
    parser.add_argument("--steps", type=int, default=PROTOCOL["steps"])
    parser.add_argument("--batch-size", type=int, default=PROTOCOL["batch_size"])
    parser.add_argument("--crop-size", type=int, nargs=2, default=PROTOCOL["crop_size"])
    parser.add_argument("--seed", type=int, default=PROTOCOL["seed"])
    parser.add_argument("--workers", type=int, default=PROTOCOL["workers"])
    parser.add_argument("--lr", type=float, default=PROTOCOL["lr"])
    parser.add_argument("--min-lr", type=float, default=PROTOCOL["min_lr"])
    parser.add_argument("--lr-schedule", choices=("cosine", "two-stage-cosine"), default="cosine")
    parser.add_argument("--stage-steps", type=int, default=30000)
    parser.add_argument("--stage2-lr", type=float, default=1e-4)
    parser.add_argument("--stage2-min-lr", type=float, default=2e-6)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=PROTOCOL["amp"])
    parser.add_argument("--val-every", type=int, default=PROTOCOL["val_every"])
    parser.add_argument("--save-every", type=int, default=PROTOCOL["save_every"])
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--saturation-probability", type=float, default=PROTOCOL["saturation_probability"])
    parser.add_argument("--saturation-range", type=float, nargs=2, default=PROTOCOL["saturation_range"])
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--stop-after", type=int, default=0,
                        help="Run this many additional steps without changing the cosine horizon")
    parser.add_argument("--print-command", action="store_true")
    args = parser.parse_args()
    for name in ("steps", "batch_size", "save_every", "log_every"):
        if getattr(args, name) <= 0:
            parser.error(f"{name} must be positive")
    if args.workers < 0 or args.stop_after < 0 or args.val_every < 0 or min(args.crop_size) <= 0:
        parser.error("Invalid workers, stop-after, val-every or crop size")
    if args.amp:
        parser.error("This experiment uses FP32; use --no-amp.")
    command = training_command(args)
    print(json.dumps({"command": command, "variant": SPEC["variant"],
                      "architecture": SPEC["recipe"],
                      "loss": "GT-Mean L1(sigma=0.1) + 0.1 * complex FFT L1 on raw RGB"},
                     ensure_ascii=False, indent=2), flush=True)
    if not args.print_command:
        subprocess.run(command, check=True)
