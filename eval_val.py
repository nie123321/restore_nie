"""Manual validation four-metric entry. Does not train and does not touch test."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys

from run import DEFAULT_DATA, build_model, choose_device, file_sha256, split_rows


METRICS = ["output_psnr", "output_ssim", "output_lpips_alex", "output_ciede2000"]
EVALUATOR = Path(r"M:\picture data\cholec80_t\train_test\HVI_CIDNet\evaluate_four_metrics.py")
FIELDS = ["sample_id", "video", "lowlight_relpath", "gt_relpath", "output_filename"]
VAL_COUNT = 200


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def val_rows(manifest: Path) -> list[dict]:
    if not manifest.is_file():
        raise FileNotFoundError(f"Required split manifest missing: {manifest}")
    raw = split_rows(manifest, "val")
    rows = []
    with manifest.open(encoding="utf-8-sig", newline="") as handle:
        by_id = {(row.get("sample_id") or "").strip(): row for row in csv.DictReader(handle)
                 if (row.get("split") or "").strip().lower() == "val"}
    if len(by_id) != len(raw):
        raise ValueError("Val sample_id values are not unique in the manifest")
    for item in raw:
        source = by_id[item["sample_id"]]
        rows.append({
            "sample_id": item["sample_id"],
            "video": source.get("video") or "",
            "lowlight_relpath": item["lowlight_relpath"],
            "gt_relpath": item["gt_relpath"],
            "output_filename": item["output_filename"],
        })
    if len(rows) != VAL_COUNT:
        raise ValueError(f"Expected {VAL_COUNT} val rows, got {len(rows)}")
    names = [row["output_filename"] for row in rows]
    if len(set(names)) != VAL_COUNT:
        raise ValueError("Val output filenames are not unique")
    return rows


def assert_unique_ids(rows: list[dict], label: str) -> None:
    ids = [row["sample_id"] for row in rows]
    if len(ids) != VAL_COUNT or len(set(ids)) != VAL_COUNT:
        raise ValueError(f"{label} must contain exactly {VAL_COUNT} unique sample_id values")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--method-label", default="DeepA-val")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    args = parser.parse_args()
    data = args.data_root.resolve()
    experiment = args.experiment_root.resolve()
    if experiment.exists() and any(experiment.iterdir()):
        raise ValueError(f"Evaluation directory is not empty: {experiment}")
    enhanced = experiment / "enhanced"
    if not EVALUATOR.is_file():
        raise FileNotFoundError(f"Four-metric script missing: {EVALUATOR}")
    checkpoint = args.checkpoint.resolve()
    payload = __import__("torch").load(checkpoint, map_location="cpu", weights_only=False)
    if payload["config"].get("arch_version") != "deep_a_v1":
        raise ValueError("Checkpoint is not a deep_a_v1 model.")
    _ = choose_device(args.device)
    _ = build_model(payload["config"]["model"])
    manifest = data / "split_manifest.csv"
    used_hash = file_sha256(manifest)
    recorded = payload["config"].get("manifest_sha256")
    if recorded and recorded != used_hash:
        raise ValueError("split_manifest.csv changed since the checkpoint was trained")
    rows = val_rows(manifest)
    assert_unique_ids(rows, "val manifest")
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
    predicted = [path.name for path in enhanced.glob("*.png")]
    if len(predicted) != VAL_COUNT or len(set(predicted)) != VAL_COUNT:
        raise RuntimeError("Prediction PNG set must be 200 unique filenames")
    if set(predicted) != {row["output_filename"] for row in rows}:
        raise RuntimeError("Prediction filenames do not match the val manifest")
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
    metric_ids = [row["sample_id"] for row in metrics]
    if len(metric_ids) != VAL_COUNT or len(set(metric_ids)) != VAL_COUNT:
        raise RuntimeError("metrics_per_sample.csv must contain exactly 200 unique sample_id rows")
    if set(metric_ids) != {row["sample_id"] for row in rows}:
        raise RuntimeError("metrics_per_sample.csv IDs do not match the val manifest")
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
        "best_metric": payload["config"].get("best_metric"),
        "manifest_sha256": used_hash,
        "samples": len(metrics),
        "split": "val",
        "metrics": means,
        "protocol": "HVI_CIDNet/evaluate_four_metrics.py on quantized RGB PNG",
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
