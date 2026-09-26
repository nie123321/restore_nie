"""Explicit train/resume/eval commands. With no command, prints help only."""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
import sys
import time
import torch
from torch.utils.data import DataLoader

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from endo_prior_a_v2.data import DEFAULT_DATA, PairedImages, StepBatches, inspect_split
from endo_prior_a_v2.model import ARCH_VERSION, PriorAV2, default_config
from endo_prior_a_v2.runtime import (HERE, atomic_save, build_model, check_resume, compute_loss,
    ensure_run_location, learning_rate, load_checkpoint, restore_rng, rng_state, seed_all,
    validate, write_json, write_status)


def choose_device(name):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if name == "auto" else torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA unavailable")
    return device


def training_config(args, split, device):
    steps_per_epoch = math.ceil(split["counts"]["train"] / args.batch_size)
    return dict(arch_version=ARCH_VERSION, model=default_config(), data_root=str(args.data_root.resolve()),
        manifest_sha256=split["manifest_sha256"], counts=split["counts"], batch_size=args.batch_size,
        crop=[args.crop_height, args.crop_width], horizontal_flip=.5, vertical_flip=.5,
        augmentation="sha256_seed_epoch_sample_v1", seed=args.seed, epochs=args.epochs,
        steps=steps_per_epoch * args.epochs, steps_per_epoch=steps_per_epoch,
        lr=args.lr, min_lr=args.min_lr, lr_schedule="cosine", weight_decay=args.weight_decay,
        optimizer="AdamW", betas=[.9, .999], grad_clip=args.grad_clip,
        amp=bool(args.amp and device.type == "cuda"), device=str(device), workers=args.workers,
        val_every=args.val_every, save_every=args.save_every, log_every=args.log_every,
        loss="rawRGB_L1(output,GT)+0.1*L1(area_Down4(Rec709(Ic)),area_Down4(Rec709(GT)))",
        best_metric="minimum_final_output_rawRGB_L1", val_policy="full_image_batch1_FP32_clampedfloat_RGB_PSNR_no_GTmean",
        initialization="from_scratch", sampler="seed_plus_epoch_randperm_v1")


def train(args):
    data, run = args.data_root.resolve(), ensure_run_location(args.run_dir)
    device = choose_device(args.device)
    if device.type != "cuda":
        raise ValueError("Training protocol requires CUDA AMP; use verify for CPU checks")
    split = inspect_split(data)
    config = training_config(args, split, device)
    if not config["amp"]:
        raise ValueError("This training protocol requires AMP")
    checkpoint = load_checkpoint(args.resume) if args.resume else None
    start = checkpoint["step"] if checkpoint else 0
    if start >= config["steps"]:
        raise ValueError("Checkpoint has completed the fixed training horizon")
    if run.exists() and any(run.iterdir()) and checkpoint is None:
        raise ValueError("Use an empty run directory or explicitly resume")
    if checkpoint is not None:
        if args.resume.resolve().parent != run:
            raise ValueError("Resume checkpoint must belong to the same run directory")
        latest = run / "last.pt"
        if not latest.is_file() or load_checkpoint(latest)["step"] != start:
            raise ValueError("Resume must use the latest saved step; forked/mixed runs are rejected")
    filenames = {s: [r["lowlight_relpath"] for r in split["splits"][s]] for s in ("train", "val", "test")}
    stop = min(config["steps"], start + args.stop_after) if args.stop_after else config["steps"]
    sampler = StepBatches(1500, args.batch_size, args.seed, start, stop)
    if checkpoint:
        check_resume(checkpoint, config, sampler, filenames)
    seed_all(args.seed)
    model = PriorAV2(config["model"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(.9, .999))
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    best_val, best_step = None, None
    if checkpoint:
        model.load_state_dict(checkpoint["model"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        scaler.load_state_dict(checkpoint["scaler"])
        restore_rng(checkpoint["rng"])
        best_val, best_step = checkpoint.get("best_val"), checkpoint.get("best_step")
    dataset = PairedImages(data, split["splits"]["train"], training=True, crop=config["crop"], seed=args.seed)
    loader = DataLoader(dataset, batch_sampler=sampler, num_workers=args.workers, pin_memory=True,
                        generator=torch.Generator().manual_seed(args.seed + 100000))
    val_loader = DataLoader(PairedImages(data, split["splits"]["val"]), batch_size=1,
                        num_workers=args.workers, generator=torch.Generator().manual_seed(args.seed + 200000))
    run.mkdir(parents=True, exist_ok=True)
    write_json(run / "config.json", config)
    del checkpoint

    def state(step):
        return dict(model=model.state_dict(), optimizer=optimizer.state_dict(), scaler=scaler.state_dict(),
                    step=step, config=config, rng=rng_state(), sampler=sampler.state(step),
                    filenames=filenames, best_val=best_val, best_step=best_step)

    model.train()
    step = start
    write_status(run, "running", step, config["steps"], best_val=best_val, best_step=best_step)
    print(json.dumps(dict(run=str(run), start_step=start, stop_step=stop, target_step=config["steps"],
        parameters=sum(p.numel() for p in model.parameters()), manifest_sha256=split["manifest_sha256"],
        train_count=1500, val_count=200, test_images_opened=0)), flush=True)
    stopped = False
    torch.cuda.reset_peak_memory_stats(device)
    try:
        with (run / "loss.jsonl").open("a", encoding="utf-8") as log:
            for low, target in loader:
                if (run / "STOP_REQUESTED").exists():
                    stopped = True
                    break
                next_step = step + 1
                started = time.perf_counter()
                for group in optimizer.param_groups:
                    group["lr"] = learning_rate(config, next_step)
                low, target = low.to(device, non_blocking=True), target.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", dtype=torch.float16):
                    packed = model(low, return_aux=True)
                    loss, loss_main, loss_aux = compute_loss(packed, target)
                if not torch.isfinite(loss):
                    raise RuntimeError(f"Non-finite loss at step {next_step}")
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                prior_scale = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                step = next_step
                row = dict(step=step, epoch=step / config["steps_per_epoch"], loss=float(loss.detach()),
                    loss_main=float(loss_main.detach()), loss_aux=float(loss_aux.detach()), aux_weight=.1,
                    grad_norm_before_clip=float(grad_norm) if torch.isfinite(grad_norm) else None,
                    amp_skipped_step=scaler.get_scale() < prior_scale, amp_scale=scaler.get_scale(),
                    lr=optimizer.param_groups[0]["lr"], seconds=time.perf_counter()-started)
                # Release train graph references before full-resolution FP32 val.
                del packed, loss, loss_main, loss_aux, low, target
                optimizer.zero_grad(set_to_none=True)
                improved = False
                if step % args.val_every == 0 or step == config["steps"]:
                    row.update(validate(model, val_loader, device))
                    improved = best_val is None or row["val_l1"] < best_val
                    if improved:
                        best_val, best_step = row["val_l1"], step
                    row.update(best_val=best_val, best_step=best_step, improved=improved)
                log.write(json.dumps(row, allow_nan=False) + "\n")
                log.flush()
                if step == start+1 or step % args.log_every == 0 or "val_l1" in row:
                    print(json.dumps(row, allow_nan=False), flush=True)
                    write_status(run, "running", step, config["steps"], best_val=best_val, best_step=best_step,
                        peak_allocated_mib=torch.cuda.max_memory_allocated(device)/2**20)
                stop_flag = (run / "STOP_REQUESTED").exists()
                if improved or step % args.save_every == 0 or step == stop or stop_flag:
                    current = state(step)
                    atomic_save(current, run / "last.pt")
                    if step % args.save_every == 0 or step == config["steps"]:
                        atomic_save(current, run / f"step_{step:06d}.pt")
                    if improved:
                        atomic_save(current, run / "best_val.pt")
                    del current
                if stop_flag:
                    stopped = True
                    break
        # Covers a stop flag found before the first new optimizer step as well.
        atomic_save(state(step), run / "last.pt")
        final = "stopped" if stopped else "complete" if step == config["steps"] else "bounded_stop"
        write_status(run, final, step, config["steps"], best_val=best_val, best_step=best_step,
                     peak_allocated_mib=torch.cuda.max_memory_allocated(device)/2**20)
    except BaseException as error:
        write_status(run, "failed", step, config["steps"], error_type=type(error).__name__, error=str(error))
        raise


def safe_resume(args):
    checkpoint = load_checkpoint(args.checkpoint)
    c = checkpoint["config"]
    run = ensure_run_location(args.checkpoint.resolve().parent)
    if (run / "STOP_REQUESTED").exists():
        raise ValueError("Remove this run's STOP_REQUESTED flag explicitly before resume")
    # Reconstruct all training settings from the checkpoint, avoiding flag drift.
    training = argparse.Namespace(data_root=Path(c["data_root"]), run_dir=run, resume=args.checkpoint,
        epochs=c["epochs"], batch_size=c["batch_size"], crop_height=c["crop"][0], crop_width=c["crop"][1],
        seed=c["seed"], lr=c["lr"], min_lr=c["min_lr"], weight_decay=c["weight_decay"],
        grad_clip=c["grad_clip"], amp=c["amp"], device=c["device"], workers=c["workers"],
        val_every=c["val_every"], save_every=c["save_every"], log_every=c["log_every"], stop_after=args.stop_after)
    train(training)


def evaluate(args):
    checkpoint = load_checkpoint(args.checkpoint)
    split = inspect_split(args.data_root)
    if checkpoint["config"]["manifest_sha256"] != split["manifest_sha256"]:
        raise ValueError("Evaluation manifest differs from training")
    device = choose_device(args.device)
    model = build_model(checkpoint["config"]["model"]).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    loader = DataLoader(PairedImages(args.data_root, split["splits"][args.split]), batch_size=1,
                        num_workers=0, generator=torch.Generator().manual_seed(200100))
    result = validate(model, loader, device, limit=args.limit)
    print(json.dumps(dict(split=args.split, checkpoint_step=checkpoint["step"],
                         subset=bool(args.limit), **result)), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command")
    training = commands.add_parser("train", help="Explicit fresh CUDA AMP training")
    training.add_argument("--data-root", type=Path, default=DEFAULT_DATA)
    training.add_argument("--run-dir", type=Path, default=HERE / "runs" / "prior_a_v2_seed100")
    training.add_argument("--epochs", type=int, default=40)
    training.add_argument("--batch-size", type=int, default=4)
    training.add_argument("--crop-height", type=int, default=192)
    training.add_argument("--crop-width", type=int, default=384)
    training.add_argument("--seed", type=int, default=100)
    training.add_argument("--lr", type=float, default=2e-4)
    training.add_argument("--min-lr", type=float, default=2e-6)
    training.add_argument("--weight-decay", type=float, default=1e-4)
    training.add_argument("--grad-clip", type=float, default=1.)
    training.add_argument("--workers", type=int, default=0)
    training.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    training.add_argument("--device", default="auto")
    training.add_argument("--val-every", type=int, default=1000)
    training.add_argument("--save-every", type=int, default=1000)
    training.add_argument("--log-every", type=int, default=50)
    training.add_argument("--stop-after", type=int, default=0, help="Bound extra steps without changing the LR horizon")
    training.add_argument("--resume", type=Path, help="Strict resume; flags must match original training")
    training.set_defaults(function=train)
    resuming = commands.add_parser("resume", help="Safe resume using the checkpoint's saved settings")
    resuming.add_argument("--checkpoint", type=Path, required=True)
    resuming.add_argument("--stop-after", type=int, default=0)
    resuming.set_defaults(function=safe_resume)
    evaluation = commands.add_parser("eval", help="Explicit complete-image FP32 val or test evaluation")
    evaluation.add_argument("--checkpoint", type=Path, required=True)
    evaluation.add_argument("--data-root", type=Path, default=DEFAULT_DATA)
    evaluation.add_argument("--split", choices=("val", "test"), default="val")
    evaluation.add_argument("--device", default="auto")
    evaluation.add_argument("--limit", type=int, default=0, help="Optional subset; 0 means complete split")
    evaluation.set_defaults(function=evaluate)
    verification = commands.add_parser("verify", help="CPU checks; GPU smoke requires an explicit flag")
    verification.add_argument("--gpu-smoke", action="store_true")
    verification.add_argument("--gpu-steps", type=int, choices=(3, 4, 5), default=3)
    verification.set_defaults(function=lambda args: __import__("endo_prior_a_v2.verify", fromlist=["verify"]).verify(args))
    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        return
    if args.command == "train":
        positive = ("epochs", "batch_size", "crop_height", "crop_width", "lr", "grad_clip", "val_every", "save_every", "log_every")
        if any(getattr(args, key) <= 0 for key in positive):
            parser.error("epochs, batch, crop, lr, grad clip and intervals must be positive")
        if args.workers < 0 or args.weight_decay < 0 or not 0 <= args.min_lr <= args.lr:
            parser.error("Invalid workers, weight decay or min LR")
    if getattr(args, "stop_after", 0) < 0 or getattr(args, "limit", 0) < 0:
        parser.error("stop-after/limit must be nonnegative")
    args.function(args)


if __name__ == "__main__":
    main()
