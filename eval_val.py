"""Manual validation four-metric entry. Does not train and does not touch test.

Builds a val-only manifest, runs ColorWaveletNet inference, then calls the
existing HVI_CIDNet/evaluate_four_metrics.py. Start this script yourself.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys

from run import DEFAULT_DATA, build_model, choose_device


METRICS = ["output_psnr", "output_ssim", "output_lpips_alex", "output_ciede2000"]
EVALUATOR = Path(r"M:\picture data\cholec80_t\train_test\HVI_CIDNet\evaluate_four_metrics.py")
FIELDS = ["sample_id", "video", "lowlight_relpath", "gt_relpath", "output_filename"]


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def val_rows(data: Path) -> list[dict]:
    manifest = data / "split_manifest.csv"
    if not manifest.is_file():
        raise FileNotFoundError(f"Required split manifest missing: {manifest}")
    rows = []
    with manifest.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if (row.get("split") or "").strip().lower() != "val":
                continue
            rel = (row.get("lowlight_relpath") or "").replace("\\", "/")
            gt = (row.get("gt_relpath") or "").replace("\\", "/")
            sample = (row.get("sample_id") or "").strip()
            if not rel or not gt or not sample:
                raise ValueError(f"Val row missing sample_id or paths in {manifest}")
            rows.append({
                "sample_id": sample,
                "video": row.get("video") or "",
                "lowlight_relpath": rel,
                "gt_relpath": gt,
                "output_filename": Path(rel).name,
            })
    if len(rows) != 200:
        raise ValueError(f"Expected 200 val rows in {manifest}, got {len(rows)}")
    ids = [row["sample_id"] for row in rows]
    names = [row["output_filename"] for row in rows]
    if len(set(ids)) != 200:
        raise ValueError(f"Val sample_id values are not unique in {manifest}")
    if len(set(names)) != 200:
        raise ValueError(f"Val output filenames are not unique in {manifest}")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--method-label", default="ColorWaveletNet-val")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    args = parser.parse_args()
    data = args.data_root.resolve()
    experiment = args.experiment_root.resolve()
    enhanced = experiment / "enhanced"
    if not EVALUATOR.is_file():
        raise FileNotFoundError(f"Four-metric script missing: {EVALUATOR}")
    checkpoint = args.checkpoint.resolve()
    payload = __import__("torch").load(checkpoint, map_location="cpu", weights_only=False)
    if payload["config"].get("arch_version") != "color_wavelet_v1":
        raise ValueError("Checkpoint is not a color_wavelet_v1 model.")
    _ = choose_device(args.device)
    _ = build_model(payload["config"]["model"])
    rows = val_rows(data)
    experiment.mkdir(parents=True, exist_ok=True)
    with (experiment / "manifest.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    infer = [
        str(args.python), "-u", "-X", "utf8", str(Path(__file__).with_name("run.py")),
        "infer", "--checkpoint", str(checkpoint),
        "--input", str(data / "val" / "lowlight"),
        "--output-dir", str(enhanced), "--device", args.device,
    ]
    completed = subprocess.run(infer, check=False)
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)
    predicted = {path.name for path in enhanced.glob("*.png")}
    expected = {row["output_filename"] for row in rows}
    if predicted != expected:
        raise RuntimeError(f"Prediction set mismatch: got {len(predicted)}, expected {len(expected)}")
    metrics_cmd = [
        str(args.python), "-u", "-X", "utf8", str(EVALUATOR),
        "--experiment-root", str(experiment),
        "--method-label", args.method_label,
        "--dataset-root", str(data),
    ]
    completed = subprocess.run(metrics_cmd, check=False)
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)
    metrics_path = experiment / "metrics_per_sample.csv"
    if not metrics_path.is_file():
        raise FileNotFoundError(f"Evaluator did not write {metrics_path}")
    with metrics_path.open(encoding="utf-8-sig", newline="") as handle:
        metrics = list(csv.DictReader(handle))
    ids = {row["sample_id"] for row in metrics}
    expected_ids = {row["sample_id"] for row in rows}
    if len(metrics) != 200 or ids != expected_ids:
        raise RuntimeError("metrics_per_sample.csv must cover the same 200 unique val sample_id values.")
    for row in metrics:
        for key in METRICS:
            value = float(row[key])
            if not math.isfinite(value):
                raise RuntimeError(f"Non-finite {key} for sample {row['sample_id']}")
    means = {key: sum(float(row[key]) for row in metrics) / len(metrics) for key in METRICS}
    summary = {
        "state": "complete",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": digest(checkpoint),
        "checkpoint_step": payload.get("step"),
        "samples": len(metrics),
        "split": "val",
        "metrics": means,
        "experiment": str(experiment),
    }
    (experiment / "summary_metrics.txt").write_text(
        "\n".join([f"method={args.method_label}", f"samples={len(metrics)}"]
                  + [f"{k}={v:.6f}" for k, v in means.items()]) + "\n",
        encoding="utf-8",
    )
    (experiment / "evaluation_contract.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8",
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
