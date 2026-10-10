import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from dataset import read_rgb, image_files
from models import load_checkpoint


def test_pairs(root):
    manifest = root / "split_manifest.csv"
    if manifest.exists():
        with manifest.open(encoding="utf-8-sig", newline="") as stream:
            rows = [row for row in csv.DictReader(stream) if row["split"] == "test"]
        rows.sort(key=lambda row: int(row["sample_id"]))
        return [(root / row["lowlight_relpath"], root / row["gt_relpath"]) for row in rows]
    return [(path, root / "test/gt" / path.name) for path in image_files(root / "test/lowlight")]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--input", type=Path, help="Image or image directory")
    group.add_argument("--data-root", type=Path, help="Paired dataset with a test split")
    parser.add_argument("--output", type=Path, default=Path("outputs/test"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Use an empty output directory")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device(args.device)
    model, checkpoint = load_checkpoint(args.checkpoint, device)
    pairs = test_pairs(args.data_root) if args.data_root else [
        (path, None) for path in (image_files(args.input) if args.input.is_dir() else [args.input])]
    if not pairs or len({path.stem for path, _ in pairs}) != len(pairs):
        raise ValueError("Empty input or duplicate output filenames")
    if args.data_root:
        import lpips
        from metrics import psnr, ssim, lpips_tensor, ciede2000
        perceptual = lpips.LPIPS(net="alex").to(device).eval()
    enhanced = args.output / "enhanced"
    enhanced.mkdir(parents=True)
    rows = []
    with torch.inference_mode():
        for index, (path, target_path) in enumerate(pairs, 1):
            output = model(read_rgb(path).unsqueeze(0).to(device))
            if not torch.isfinite(output).all():
                raise RuntimeError(f"Nonfinite output: {path}")
            array = output[0].clamp(0, 1).permute(1, 2, 0).cpu().numpy()
            candidate = np.rint(array * 255).astype(np.uint8)
            Image.fromarray(candidate).save(enhanced / f"{path.stem}.png")
            if target_path is not None:
                with Image.open(target_path) as image:
                    target = np.asarray(image.convert("RGB"), dtype=np.uint8)
                if target.shape != candidate.shape:
                    raise ValueError(f"Paired dimensions differ: {path}")
                rows.append({"image": path.name, "PSNR": psnr(target, candidate),
                             "SSIM": ssim(target, candidate),
                             "LPIPS": float(perceptual(lpips_tensor(target, device), lpips_tensor(candidate, device)).item()),
                             "CIEDE2000": ciede2000(target, candidate)})
            if index % 50 == 0 or index == len(pairs):
                print(f"{index}/{len(pairs)}", flush=True)
    if rows:
        with (args.output / "metrics.csv").open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        summary = {key: float(np.mean([row[key] for row in rows])) for key in rows[0] if key != "image"}
        summary.update(step=checkpoint.get("step"), images=len(rows))
        (args.output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
