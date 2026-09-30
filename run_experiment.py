"""Train one of three matched seed-100 variants, then test raw-val-L1 best once."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from run_demo import DEFAULT_DATA

CODE = Path(__file__).resolve().parent
VARIANTS = {"a": "none", "star_bottom2": "bottleneck-two", "star_multiscale": "multiscale"}


def training_command(variant, seed, run, data):
    return [sys.executable, "-B", "-u", "-X", "utf8", str(CODE / "run_demo.py"), "train",
            "--data-root", str(data), "--run-dir", str(run),
            "--steps", "30000", "--epochs", "0", "--batch-size", "8", "--seed", str(seed),
            "--lr", "2e-4", "--lr-schedule", "cosine", "--min-lr", "2e-6",
            "--weight-decay", "1e-4", "--grad-clip", "1",
            "--val-every", "500", "--save-every", "500", "--log-every", "50",
            "--width", "24", "--spectral-mode", "off", "--output-mode", "direct",
            "--fusion-mode", "gated", "--bottleneck-attention", "mdta",
            "--star-refinement", VARIANTS[variant],
            "--loss-mode", "gt-mean-l1", "--gt-mean-sigma", "0.1",
            "--crop-size", "192", "384", "--geometric-augs",
            "--saturation-probability", "0.25", "--saturation-range", "0.9", "1.1",
            "--workers", "0", "--device", "cuda", "--amp"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--pipeline-status", type=Path, required=True)
    args = parser.parse_args()
    def record(stage, **details):
        args.pipeline_status.write_text(json.dumps(dict(variant=args.variant, seed=args.seed,
            stage=stage, wrapper_pid=os.getpid(), time_unix=time.time(),
            run_dir=str(args.run_dir), **details), indent=2) + "\n", encoding="utf-8")
    def execute(command, stage):
        child = subprocess.Popen(command, cwd=CODE)
        record(stage, child_pid=child.pid, command=command)
        result = child.wait()
        if result != 0:
            raise subprocess.CalledProcessError(result, command)
    try:
        execute(training_command(args.variant, args.seed, args.run_dir, args.data_root), "training")
        execute([sys.executable, "-B", "-u", "-X", "utf8", str(CODE / "test_best.py"),
                 "--checkpoint", str(args.run_dir / "best_val.pt"),
                 "--data-root", str(args.data_root), "--output-dir", str(args.run_dir / "test_best"),
                 "--method-label", f"{args.variant}-seed{args.seed}-D4-sat025-30k-valL1best-test",
                 "--device", "cuda"], "evaluating")
        record("complete")
    except BaseException as error:
        record("failed", error_type=type(error).__name__, error=str(error))
        raise


if __name__ == "__main__":
    main()
