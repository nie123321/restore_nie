import argparse
import json
import math
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import PairedImages, StepBatches
from losses import gt_mean_fft_l1
from models import EnhancementNet
from metrics import ssim


def learning_rate(config, step):
    progress = (step - 1) / max(1, config["steps"] - 1)
    return config["min_lr"] + .5 * (config["lr"] - config["min_lr"]) * (1 + math.cos(math.pi * progress))


def rng_state():
    name, keys, position, has_gauss, cached = np.random.get_state()
    return {"python": random.getstate(), "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "numpy": {"algorithm": name, "keys": keys.tolist(), "position": int(position),
                      "has_gauss": int(has_gauss), "cached_gaussian": float(cached)}}


def restore_rng(state):
    random.setstate(state["python"])
    value = state["numpy"]
    if isinstance(value, dict):
        value = (value["algorithm"], np.asarray(value["keys"], dtype=np.uint32),
                 value["position"], value["has_gauss"], value["cached_gaussian"])
    np.random.set_state(value)
    torch.set_rng_state(state["torch"])
    if state["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def save_checkpoint(state, path):
    temporary = path.with_suffix(".tmp")
    torch.save(state, temporary)
    temporary.replace(path)


@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    psnr_sum, ssim_sum = 0.0, 0.0
    for low, target in loader:
        target = target.to(device)
        output = model(low.to(device)).clamp(0, 1)
        mse = float((output - target).square().mean())
        psnr_sum += -10 * math.log10(max(mse, 1e-12))
        reference = target[0].permute(1, 2, 0).cpu().numpy().astype(np.float64) * 255.0
        prediction = output[0].permute(1, 2, 0).cpu().numpy().astype(np.float64) * 255.0
        ssim_sum += ssim(reference, prediction)
    model.train()
    return {"val_psnr": psnr_sum / len(loader), "val_ssim": ssim_sum / len(loader)}


def check_resume(checkpoint, config):
    previous = dict(checkpoint["config"])
    previous["fft_weight"] = previous.get("fft_weight", previous.get("fft_loss_weight", 0.1))
    keys = ("data_root", "steps", "batch_size", "seed", "lr", "min_lr", "lr_schedule",
            "weight_decay", "grad_clip", "crop_size", "geometric_augs", "saturation_probability",
            "saturation_range", "gt_mean_sigma", "fft_weight")
    for key in keys:
        if previous.get(key) != config[key]:
            raise ValueError(f"Resume configuration differs: {key}; repeat the original setting")
    if previous.get("amp", False) or previous.get("loss_mode", "gt-mean-fft") != "gt-mean-fft":
        raise ValueError("Resume requires FP32 and the GT-Mean + FFT loss")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("outputs/train"))
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.json"))
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--stop-after", type=int, default=0, help="Additional steps; leaves the cosine horizon unchanged")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    for key in ("batch_size", "workers"):
        if getattr(args, key) is not None:
            config[key] = getattr(args, key)
    config.update(data_root=str(args.data_root.resolve()), amp=False, loss_mode="gt-mean-fft")
    config.pop("best_metric", None)
    if config["batch_size"] < 1 or config["workers"] < 0 or config["steps"] < 1 or args.stop_after < 0:
        parser.error("Invalid batch size, workers or steps")
    if config["lr_schedule"] != "cosine":
        parser.error("This release uses cosine learning rate scheduling")
    if args.output.exists() and any(args.output.iterdir()) and not args.resume:
        parser.error("Use an empty output directory or --resume")
    device = torch.device(args.device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    model = EnhancementNet().to(device)
    config["model"] = model.config
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    start = 0
    checkpoint = None
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        check_resume(checkpoint, config)
        from models.checkpoint import validate_model_config
        validate_model_config(checkpoint["config"]["model"])
        model.load_state_dict(checkpoint["model"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        start = checkpoint["step"]
        restore_rng(checkpoint["rng"])
    if start >= config["steps"]:
        parser.error("Training already reached the configured number of steps")
    train_set = PairedImages(args.data_root, "train", crop_size=config["crop_size"], seed=config["seed"],
                             geometric_augs=config["geometric_augs"],
                             saturation_probability=config["saturation_probability"],
                             saturation_range=config["saturation_range"])
    names = [path.name for path in train_set.low]
    if checkpoint is not None and checkpoint["train_filenames"] != names:
        raise ValueError("Training image list changed since the checkpoint")
    stop = min(config["steps"], start + args.stop_after) if args.stop_after else config["steps"]
    sampler = StepBatches(len(train_set), config["batch_size"], config["seed"], start, stop,
                           geometric_augs=config["geometric_augs"])
    loader = DataLoader(train_set, batch_sampler=sampler, num_workers=config["workers"],
                        generator=torch.Generator().manual_seed(config["seed"] + 100000),
                        pin_memory=device.type == "cuda")
    val_loader = DataLoader(PairedImages(args.data_root, "val"), batch_size=1, num_workers=config["workers"],
                            generator=torch.Generator().manual_seed(config["seed"] + 200000)) if config["val_every"] else None
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(f"start={start}, stop={stop}, cosine horizon={config['steps']}", flush=True)
    model.train()
    with (args.output / "train.jsonl").open("a", encoding="utf-8") as log:
        for step, (low, target) in enumerate(loader, start=start + 1):
            began = time.perf_counter()
            lr = learning_rate(config, step)
            for group in optimizer.param_groups:
                group["lr"] = lr
            optimizer.zero_grad(set_to_none=True)
            loss, _ = gt_mean_fft_l1(model(low.to(device)), target.to(device),
                                     sigma=config["gt_mean_sigma"], fft_weight=config["fft_weight"])
            if not torch.isfinite(loss):
                raise RuntimeError(f"Nonfinite loss at step {step}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip"], error_if_nonfinite=True)
            optimizer.step()
            row = {"step": step, "loss": float(loss.detach()), "lr": lr, "seconds": time.perf_counter() - began}
            if val_loader is not None and step % config["val_every"] == 0:
                row.update(validate(model, val_loader, device))
            log.write(json.dumps(row) + "\n")
            log.flush()
            if step == start + 1 or step % 50 == 0 or step == stop:
                print(json.dumps(row), flush=True)
            if step % config["save_every"] == 0:
                state = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "scaler": {},
                         "step": step, "config": config, "rng": rng_state(), "train_filenames": names,
                         "val_metrics": {key: row[key] for key in ("val_psnr", "val_ssim") if key in row}}
                save_checkpoint(state, args.output / f"step_{step:06d}.pt")


if __name__ == "__main__":
    main()
