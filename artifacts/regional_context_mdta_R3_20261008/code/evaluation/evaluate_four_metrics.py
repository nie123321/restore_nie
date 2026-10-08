from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

import cv2
import lpips
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from skimage.color import deltaE_ciede2000, rgb2lab
from tqdm import tqdm


DEFAULT_DATASET_ROOT = Path(r"M:\picture data\cholec80_t\最终结果")
METRIC_KEYS = [
    "input_psnr",
    "output_psnr",
    "delta_psnr",
    "input_ssim",
    "output_ssim",
    "delta_ssim",
    "input_lpips_alex",
    "output_lpips_alex",
    "delta_lpips_alex",
    "input_ciede2000",
    "output_ciede2000",
    "delta_ciede2000",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the shared four metrics.")
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--method-label", required=True)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    return parser.parse_args()


def load_u8(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


def psnr(target: np.ndarray, candidate: np.ndarray) -> float:
    target_f = target.astype(np.float32) / 255.0
    candidate_f = candidate.astype(np.float32) / 255.0
    mse = float(np.mean((target_f - candidate_f) ** 2))
    return 100.0 if mse == 0.0 else 10.0 * math.log10(1.0 / mse)


def channel_ssim(target: np.ndarray, candidate: np.ndarray) -> float:
    c1 = (0.01 * 255.0) ** 2
    c2 = (0.03 * 255.0) ** 2
    target = target.astype(np.float64)
    candidate = candidate.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())
    mu1 = cv2.filter2D(target, -1, window)[5:-5, 5:-5]
    mu2 = cv2.filter2D(candidate, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(target**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(candidate**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(target * candidate, -1, window)[5:-5, 5:-5] - mu1_mu2
    result = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / (
        (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
    )
    return float(result.mean())


def ssim(target: np.ndarray, candidate: np.ndarray) -> float:
    return float(
        np.mean([channel_ssim(target[:, :, i], candidate[:, :, i]) for i in range(3)])
    )


def lpips_tensor(image: np.ndarray, device: torch.device) -> torch.Tensor:
    value = (
        torch.from_numpy(image.transpose(2, 0, 1).copy()).unsqueeze(0).float() / 255.0
    )
    return (value * 2.0 - 1.0).to(device)


def ciede2000(target: np.ndarray, candidate: np.ndarray) -> float:
    target_f = target.astype(np.float32) / 255.0
    candidate_f = candidate.astype(np.float32) / 255.0
    return float(np.mean(deltaE_ciede2000(rgb2lab(target_f), rgb2lab(candidate_f))))


def mean(rows: list[dict[str, str | float]], key: str) -> float:
    return float(np.mean([float(row[key]) for row in rows]))


def write_csv(path: Path, rows: list[dict[str, str | float]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def make_grid(
    experiment_root: Path,
    dataset_root: Path,
    method_label: str,
    rows: list[dict[str, str | float]],
) -> None:
    first_by_video: dict[str, dict[str, str | float]] = {}
    for row in rows:
        first_by_video.setdefault(str(row["video"]), row)
    chosen = [
        first_by_video[key]
        for key in sorted(first_by_video, key=lambda value: int(value.removeprefix("video")))
    ]
    images: list[tuple[dict[str, str | float], np.ndarray, np.ndarray, np.ndarray]] = []
    for row in chosen:
        low = load_u8(dataset_root / str(row["lowlight_relpath"]))
        output = load_u8(experiment_root / "enhanced" / str(row["output_filename"]))
        target = load_u8(dataset_root / str(row["gt_relpath"]))
        images.append((row, low, output, target))
    height, width = images[0][1].shape[:2]
    label_height = 30
    canvas = Image.new("RGB", (width * 3, (height + label_height) * len(images)), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default(size=18)
    for row_index, (row, low, output, target) in enumerate(images):
        y = row_index * (height + label_height)
        labels = [
            f"Input | sample {row['sample_id']} {row['video']}",
            f"{method_label} | PSNR {float(row['output_psnr']):.2f}",
            "Ground truth",
        ]
        for column, (label, array) in enumerate(zip(labels, [low, output, target])):
            x = column * width
            draw.text((x + 6, y + 5), label, fill="black", font=font)
            canvas.paste(Image.fromarray(array), (x, y + label_height))
    canvas.save(experiment_root / "comparison_grid.png")


def main() -> None:
    args = parse_args()
    manifest_path = args.experiment_root / "manifest.csv"
    with manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
        manifest_rows = list(csv.DictReader(handle))
    if not manifest_rows:
        raise RuntimeError(f"Empty manifest: {manifest_path}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lpips_model = lpips.LPIPS(net="alex", version="0.1").to(device).eval()
    metric_rows: list[dict[str, str | float]] = []
    with torch.inference_mode():
        for row in tqdm(manifest_rows, desc=f"{args.method_label} four metrics"):
            low = load_u8(args.dataset_root / row["lowlight_relpath"])
            output = load_u8(args.experiment_root / "enhanced" / row["output_filename"])
            target = load_u8(args.dataset_root / row["gt_relpath"])
            if low.shape != output.shape or low.shape != target.shape:
                raise ValueError(
                    f"Shape mismatch for sample {row['sample_id']}: "
                    f"{low.shape}, {output.shape}, {target.shape}"
                )
            input_psnr = psnr(target, low)
            output_psnr = psnr(target, output)
            input_ssim = ssim(target, low)
            output_ssim = ssim(target, output)
            target_tensor = lpips_tensor(target, device)
            input_lpips = float(
                lpips_model(lpips_tensor(low, device), target_tensor).item()
            )
            output_lpips = float(
                lpips_model(lpips_tensor(output, device), target_tensor).item()
            )
            input_ciede = ciede2000(target, low)
            output_ciede = ciede2000(target, output)
            metric_rows.append(
                {
                    **row,
                    "input_psnr": input_psnr,
                    "output_psnr": output_psnr,
                    "delta_psnr": output_psnr - input_psnr,
                    "input_ssim": input_ssim,
                    "output_ssim": output_ssim,
                    "delta_ssim": output_ssim - input_ssim,
                    "input_lpips_alex": input_lpips,
                    "output_lpips_alex": output_lpips,
                    "delta_lpips_alex": output_lpips - input_lpips,
                    "input_ciede2000": input_ciede,
                    "output_ciede2000": output_ciede,
                    "delta_ciede2000": output_ciede - input_ciede,
                }
            )
    write_csv(args.experiment_root / "metrics_per_sample.csv", metric_rows)
    grouped: dict[str, list[dict[str, str | float]]] = defaultdict(list)
    for row in metric_rows:
        grouped[str(row["video"])].append(row)
    video_rows: list[dict[str, str | float]] = []
    for video in sorted(grouped, key=lambda value: int(value.removeprefix("video"))):
        rows = grouped[video]
        video_rows.append(
            {
                "video": video,
                "sample_count": len(rows),
                **{key: mean(rows, key) for key in METRIC_KEYS},
            }
        )
    write_csv(args.experiment_root / "metrics_per_video.csv", video_rows)
    improved = {
        "psnr": sum(float(row["delta_psnr"]) > 0 for row in metric_rows),
        "ssim": sum(float(row["delta_ssim"]) > 0 for row in metric_rows),
        "lpips": sum(float(row["delta_lpips_alex"]) < 0 for row in metric_rows),
        "ciede2000": sum(float(row["delta_ciede2000"]) < 0 for row in metric_rows),
    }
    summary = [
        f"method={args.method_label}",
        f"samples={len(metric_rows)}",
        f"videos={len(video_rows)}",
        *[f"{key}={mean(metric_rows, key):.6f}" for key in METRIC_KEYS],
        f"psnr_improved={improved['psnr']}/{len(metric_rows)}",
        f"ssim_improved={improved['ssim']}/{len(metric_rows)}",
        f"lpips_improved={improved['lpips']}/{len(metric_rows)}",
        f"ciede2000_improved={improved['ciede2000']}/{len(metric_rows)}",
    ]
    (args.experiment_root / "summary_metrics.txt").write_text(
        "\n".join(summary) + "\n", encoding="utf-8"
    )
    make_grid(args.experiment_root, args.dataset_root, args.method_label, metric_rows)
    print("\n".join(summary))


if __name__ == "__main__":
    main()
