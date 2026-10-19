import argparse
import csv
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image
import torch

from dataset import read_rgb, image_files
from models import load_checkpoint
from runtime_utils import (RunLock, atomic_json, configure_fp32, output_entries,
                           sha256_file, split_manifest_info)


def test_pairs(root):
    info = split_manifest_info(root)
    if info is not None:
        with Path(info["path"]).open(encoding="utf-8-sig", newline="") as stream:
            rows = [row for row in csv.DictReader(stream) if row["split"] == "test"]
        rows.sort(key=lambda row: int(row["sample_id"]))
        return [(root / row["lowlight_relpath"], root / row["gt_relpath"]) for row in rows]
    return [(path, root / "test/gt" / path.name) for path in image_files(root / "test/lowlight")]


def run(args, device):
    with RunLock(args.output):
        if output_entries(args.output):
            raise ValueError("Use an empty output directory; existing inference outputs are never overwritten")
        model, checkpoint = load_checkpoint(args.checkpoint, device)
        model = model.to(device=device, dtype=torch.float32).eval()
        pairs = test_pairs(args.data_root) if args.data_root else [
            (path, None) for path in (image_files(args.input) if args.input.is_dir() else [args.input])]
        if not pairs or len({path.stem.casefold() for path, _ in pairs}) != len(pairs):
            raise ValueError("Empty input or duplicate output filenames")
        inputs = []
        for path, target in pairs:
            if not path.is_file() or (target is not None and not target.is_file()):
                raise FileNotFoundError(f"Input or paired target is missing: {path}, {target}")
            with Image.open(path) as image:
                width, height = image.size
            if target is not None:
                with Image.open(target) as image:
                    if image.size != (width, height):
                        raise ValueError(f"Paired dimensions differ: {path}")
                if min(width, height) < 11:
                    raise ValueError("Four-metric evaluation requires images at least 11x11 for the original SSIM")
            row = {"input": str(path.resolve()), "sha256": sha256_file(path),
                   "width": width, "height": height, "output": f"{path.stem}.png"}
            if target is not None:
                row.update(target=str(target.resolve()), target_sha256=sha256_file(target))
            inputs.append(row)
        input_hash = hashlib.sha256(json.dumps(inputs, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        if args.data_root:
            import lpips
            from metrics import psnr, ssim, lpips_tensor, ciede2000
            perceptual = lpips.LPIPS(net="alex").to(device).eval()
        metadata = {"state": "running", "variant": model.config["experiment_variant"],
                    "checkpoint": str(args.checkpoint.resolve()),
                    "checkpoint_sha256": sha256_file(args.checkpoint), "step": checkpoint.get("step"),
                    "input": str((args.data_root or args.input).resolve()),
                    "images": len(pairs), "test_images": len(pairs) if args.data_root else None,
                    "processed": 0, "device": str(device),
                    "precision": {"dtype": "float32", "amp": False, "tf32": False},
                    "protocol": "RGB full-resolution FP32; clamp [0,1]; np.rint(x*255); uint8 PNG; no target tuning",
                    "input_manifest_sha256": input_hash, "split_manifest": split_manifest_info(args.data_root),
                    "metrics": ["PSNR", "SSIM", "LPIPS", "CIEDE2000"] if args.data_root else [],
                    "metric_domain": "saved uint8 PNG RGB" if args.data_root else None}
        atomic_json(args.output / "inputs.json", {"sha256": input_hash, "files": inputs})
        atomic_json(args.output / "inference.json", metadata)
        enhanced = args.output / "enhanced"
        enhanced.mkdir()
        rows = []
        began = time.perf_counter()
        try:
            with torch.inference_mode():
                for index, (path, target_path) in enumerate(pairs, 1):
                    output = model(read_rgb(path).unsqueeze(0).to(device))
                    if not torch.isfinite(output).all():
                        raise RuntimeError(f"Nonfinite output: {path}")
                    array = output[0].clamp(0, 1).permute(1, 2, 0).cpu().numpy()
                    candidate = np.rint(array * 255).astype(np.uint8)
                    with (enhanced / f"{path.stem}.png").open("xb") as stream:
                        Image.fromarray(candidate).save(stream, format="PNG")
                    if target_path is not None:
                        with Image.open(target_path) as image:
                            target = np.asarray(image.convert("RGB"), dtype=np.uint8)
                        if target.shape != candidate.shape:
                            raise ValueError(f"Paired dimensions differ: {path}")
                        row = {"image": path.name, "PSNR": psnr(target, candidate),
                               "SSIM": ssim(target, candidate),
                               "LPIPS": float(perceptual(lpips_tensor(target, device), lpips_tensor(candidate, device)).item()),
                               "CIEDE2000": ciede2000(target, candidate)}
                        if not all(np.isfinite(value) for key, value in row.items() if key != "image"):
                            raise RuntimeError(f"Nonfinite metric: {path}")
                        rows.append(row)
                    metadata["processed"] = index
                    if index % 50 == 0 or index == len(pairs):
                        print(f"{index}/{len(pairs)}", flush=True)
                        atomic_json(args.output / "inference.json", metadata)
            if rows:
                with (args.output / "metrics.csv").open("w", encoding="utf-8-sig", newline="") as stream:
                    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                    writer.writeheader()
                    writer.writerows(rows)
                summary = {key: float(np.mean([row[key] for row in rows])) for key in rows[0] if key != "image"}
                summary.update(step=checkpoint.get("step"), images=len(rows), variant=metadata["variant"])
                atomic_json(args.output / "summary.json", summary)
                print(json.dumps(summary, indent=2))
            metadata.update(state="complete", elapsed_seconds=time.perf_counter() - began)
            atomic_json(args.output / "inference.json", metadata)
        except BaseException as error:
            metadata.update(state="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                            error_type=type(error).__name__, error=str(error))
            atomic_json(args.output / "inference.json", metadata)
            raise


def main():
    parser = argparse.ArgumentParser(description="Strict-variant full-resolution FP32 PNG inference and paired four-metric evaluation")
    parser.add_argument("--checkpoint", type=Path, required=True)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--input", type=Path, help="Image or image directory")
    group.add_argument("--data-root", type=Path, help="Paired dataset with a test split")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "outputs" / "test")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--amp", action="store_true", help="Unsupported; protocol requires FP32")
    args = parser.parse_args()
    if args.amp:
        parser.error("--amp is not supported; AMP and TF32 must remain disabled")
    if args.input is None and args.data_root is None:
        config = json.loads(Path(__file__).with_name('config.json').read_text(encoding='utf-8-sig'))
        args.data_root = Path(config['data_root'])
    configure_fp32()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA requested but unavailable; explicitly use --device cpu for a CPU probe")
    args.output = args.output.resolve()
    if args.data_root is not None:
        args.data_root = args.data_root.resolve()
    if args.input is not None:
        args.input = args.input.resolve()
    run(args, device)


if __name__ == "__main__":
    main()
