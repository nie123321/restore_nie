"""Test-split evaluation for A variants, including frozen LF correction.

Inference is FP32 on full images; enhanced PNGs are clamp [0,1] + np.rint
uint8, then scored by the shared four-metric evaluator.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch
from PIL import Image

from model import EnhancementDemo
from run_demo import DEFAULT_DATA, build_model, read_rgb


EVALUATOR = Path(r"M:\picture data\cholec80_t\code\cut\endo_enhancement_demo\runs"
                 r"\A_spatial_fusion_on_off_10k_seed100_20260924\code\evaluate_four_metrics.py")
MANIFEST_SHA256 = "d969938d27c82e72bd5dce875979e4c2a6c1094c25ae1d7d1267b66e87a68797"
FIELDS = ["sample_id", "video", "lowlight_relpath", "gt_relpath", "output_filename"]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def test_rows(manifest: Path) -> list[dict]:
    with manifest.open(encoding="utf-8-sig", newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row["split"] == "test"]
    rows.sort(key=lambda row: int(row["sample_id"]))
    if len(rows) != 300:
        raise ValueError(f"Expected 300 test rows, got {len(rows)}")
    names = [Path(row["lowlight_relpath"]).stem + ".png" for row in rows]
    if len(set(names)) != len(names):
        raise ValueError("Test output filenames are not unique")
    return [{"sample_id": row["sample_id"], "video": row["video"],
             "lowlight_relpath": row["lowlight_relpath"], "gt_relpath": row["gt_relpath"],
             "output_filename": name} for row, name in zip(rows, names)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--method-label", default="A-crop192x384-test")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    data = args.data_root.resolve()
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Output directory is not empty: {output}")
    manifest = data / "split_manifest.csv"
    recorded = file_sha256(manifest)
    if recorded != MANIFEST_SHA256:
        raise ValueError(f"split_manifest.csv changed: {recorded}")
    if not EVALUATOR.is_file():
        raise FileNotFoundError(f"Four-metric script missing: {EVALUATOR}")
    rows = test_rows(manifest)
    checkpoint = torch.load(args.checkpoint.resolve(), map_location="cpu", weights_only=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else torch.device(args.device)
    config = checkpoint["config"]
    if config.get("model_kind") == "frozen-low-frequency-v1":
        from low_freq_blocks import FrozenLowFrequency
        model = FrozenLowFrequency(config["model"], **config["low_frequency"]).to(device)
    else:
        model = build_model(config).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    enhanced = output / "enhanced"
    enhanced.mkdir(parents=True, exist_ok=True)
    with (output / "manifest.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    l1_sum, psnr_sum = 0.0, 0.0
    with torch.no_grad():
        for index, row in enumerate(rows, start=1):
            low = read_rgb(data / row["lowlight_relpath"]).unsqueeze(0).to(device)
            target = read_rgb(data / row["gt_relpath"]).unsqueeze(0).to(device)
            result = model(low).float()
            if result.shape != low.shape or not torch.isfinite(result).all():
                raise RuntimeError(f"Invalid model output for {row['lowlight_relpath']}")
            l1_sum += float((result - target).abs().mean())
            mse = float((result.clamp(0, 1) - target).square().mean())
            psnr_sum += -10.0 * float(np.log10(max(mse, 1e-12)))
            array = result[0].clamp(0, 1).permute(1, 2, 0).cpu().numpy()
            Image.fromarray(np.rint(array * 255).astype(np.uint8)).save(enhanced / row["output_filename"])
            if index % 50 == 0 or index == len(rows):
                print(f"[{index}/{len(rows)}] sample={row['sample_id']}", flush=True)
    inference = {
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_step": checkpoint.get("step"),
        "checkpoint_sha256": file_sha256(args.checkpoint.resolve()),
        "manifest_sha256": recorded,
        "split": "test",
        "samples": len(rows),
        "precision": "FP32",
        "quantization": "clamp [0,1], np.rint RGB uint8",
        "float_psnr": psnr_sum / len(rows),
        "raw_l1": l1_sum / len(rows),
    }
    (output / "inference.json").write_text(json.dumps(inference, indent=2) + "\n", encoding="utf-8")
    command = [sys.executable, "-u", "-X", "utf8", str(EVALUATOR),
               "--experiment-root", str(output), "--method-label", args.method_label,
               "--dataset-root", str(data)]
    completed = subprocess.run(command, check=False)
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)
    print(json.dumps(inference, indent=2), flush=True)
    print("TEST_EVAL_PASS", flush=True)


if __name__ == "__main__":
    main()
