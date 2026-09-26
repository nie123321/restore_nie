"""Manifest-bound pairs, deterministic paired augmentation and step sampler."""
from __future__ import annotations
import csv
import hashlib
import math
from pathlib import Path
import torch
from torch.utils.data import Dataset, Sampler
from PIL import Image
import numpy as np

DEFAULT_DATA = Path(r"M:\picture data\cholec80_t\train_test")
MANIFEST_SHA256 = "d969938d27c82e72bd5dce875979e4c2a6c1094c25ae1d7d1267b66e87a68797"
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def sha256(path):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def read_rgb(path):
    with Image.open(path) as image:
        array = np.array(image.convert("RGB"), dtype=np.float32) / 255
    return torch.from_numpy(array).permute(2, 0, 1)


def inspect_split(root: Path):
    """Read all manifest rows, check train/val paths; never open test images."""
    root = root.resolve()
    manifest = root / "split_manifest.csv"
    actual_hash = sha256(manifest)
    if actual_hash != MANIFEST_SHA256:
        raise ValueError(f"Manifest SHA256 differs: expected {MANIFEST_SHA256}, got {actual_hash}")
    with manifest.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    splits = {split: [] for split in ("train", "val", "test")}
    for row in rows:
        split = row["split"].strip().lower()
        if split not in splits:
            raise ValueError(f"Unknown manifest split: {split}")
        for key, folder in (("lowlight_relpath", "lowlight"), ("gt_relpath", "gt")):
            path = Path(row[key].replace("\\", "/"))
            if path.is_absolute() or ".." in path.parts or path.parts[:2] != (split, folder) or len(path.parts) != 3:
                raise ValueError(f"Unsafe or invalid manifest path: {path}")
            if split != "test" and not (root / path).is_file():
                raise FileNotFoundError(root / path)
        if Path(row["lowlight_relpath"]).name != Path(row["gt_relpath"]).name:
            raise ValueError("Paired filenames differ")
        splits[split].append(row)
    all_ids = [r["sample_id"] for r in rows]
    if len(set(all_ids)) != len(all_ids):
        raise ValueError("Manifest sample IDs are not globally unique")
    for split, expected in (("train", 1500), ("val", 200), ("test", 300)):
        if len(splits[split]) != expected:
            raise ValueError(f"Expected {expected} {split} rows")
        names = [Path(r["lowlight_relpath"]).name for r in splits[split]]
        if len(set(names)) != len(names):
            raise ValueError(f"Duplicate filenames in {split}")
        if split != "test":
            for folder in ("lowlight", "gt"):
                files = {p.name for p in (root / split / folder).iterdir()
                         if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES}
                if files != set(names):
                    raise ValueError(f"{split}/{folder} files differ from manifest")
    video_sets = [{r["video"] for r in splits[s]} for s in ("train", "val", "test")]
    if any(video_sets[i] & video_sets[j] for i in range(3) for j in range(i+1, 3)):
        raise ValueError("Video leakage across splits")
    return dict(manifest_sha256=actual_hash, splits=splits,
                counts={s: len(r) for s, r in splits.items()}, test_images_opened=0)


def augment_pair(low, target, crop, seed, epoch, sample_id):
    if low.shape != target.shape:
        raise ValueError("Paired dimensions differ")
    ch, cw = crop
    h, w = low.shape[-2:]
    if h < ch or w < cw:
        raise ValueError(f"Training image {h}x{w} smaller than crop {ch}x{cw}")
    digest = hashlib.sha256(f"prior-a-v2:{seed}:{epoch}:{sample_id}".encode()).digest()
    local_seed = int.from_bytes(digest[:8], "little") % (2**63 - 1)
    generator = torch.Generator().manual_seed(local_seed)
    y = int(torch.randint(h - ch + 1, (), generator=generator))
    x = int(torch.randint(w - cw + 1, (), generator=generator))
    horizontal = bool(torch.rand((), generator=generator) < .5)
    vertical = bool(torch.rand((), generator=generator) < .5)
    low, target = low[..., y:y+ch, x:x+cw], target[..., y:y+ch, x:x+cw]
    axes = ([-1] if horizontal else []) + ([-2] if vertical else [])
    if axes:
        low, target = low.flip(axes), target.flip(axes)
    return low.contiguous(), target.contiguous()


class PairedImages(Dataset):
    def __init__(self, root, rows, training=False, crop=(192, 384), seed=100):
        self.root, self.rows = Path(root), rows
        self.training, self.crop, self.seed = training, tuple(crop), seed

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, key):
        index, epoch = key if isinstance(key, tuple) else (key, 0)
        row = self.rows[index]
        low, target = read_rgb(self.root / row["lowlight_relpath"]), read_rgb(self.root / row["gt_relpath"])
        if low.shape != target.shape:
            raise ValueError(f"Pair dimensions differ for sample {row['sample_id']}")
        if self.training:
            low, target = augment_pair(low, target, self.crop, self.seed, epoch, row["sample_id"])
        return low, target


class StepBatches(Sampler):
    """Epoch seed+epoch shuffle; keys carry epoch so prefetch is resume-safe."""
    def __init__(self, count, batch, seed, start, stop):
        self.count, self.batch, self.seed = count, batch, seed
        self.start, self.stop = start, stop
        self.per_epoch = math.ceil(count / batch)

    def __len__(self):
        return max(0, self.stop - self.start)

    def state(self, next_step):
        epoch, batch_index = divmod(next_step, self.per_epoch)
        return dict(count=self.count, batch=self.batch, seed=self.seed, next_step=next_step,
                    epoch=epoch, batch_index=batch_index, algorithm="seed_plus_epoch_randperm_v1")

    def __iter__(self):
        previous, order = -1, []
        for step in range(self.start, self.stop):
            epoch, batch_index = divmod(step, self.per_epoch)
            if epoch != previous:
                generator = torch.Generator().manual_seed(self.seed + epoch)
                order = torch.randperm(self.count, generator=generator).tolist()
                previous = epoch
            offset = batch_index * self.batch
            yield [(index, epoch) for index in order[offset:offset+self.batch]]
