from pathlib import Path
import hashlib
import math

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, Sampler

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def read_rgb(path: Path) -> torch.Tensor:
    with Image.open(path) as image:
        array = np.asarray(image.convert('RGB'), dtype=np.float32).copy() / 255.0
    return torch.from_numpy(array).permute(2, 0, 1)


def image_files(directory: Path) -> list[Path]:
    return sorted((p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES))


def augment_pair(low, target, crop, seed, epoch, sample_id, quarter_turn=False):
    if low.shape != target.shape:
        raise ValueError('Paired dimensions differ')
    digest = hashlib.sha256(f'prior-a-v2:{seed}:{epoch}:{sample_id}'.encode()).digest()
    local_seed = int.from_bytes(digest[:8], 'little') % (2 ** 63 - 1)
    generator = torch.Generator().manual_seed(local_seed)
    if crop is not None:
        (ch, cw) = crop
        (h, w) = low.shape[-2:]
        if h < ch or w < cw:
            raise ValueError(f'Training image {h}x{w} smaller than crop {ch}x{cw}')
        y = int(torch.randint(h - ch + 1, (), generator=generator))
        x = int(torch.randint(w - cw + 1, (), generator=generator))
        (low, target) = (low[..., y:y + ch, x:x + cw], target[..., y:y + ch, x:x + cw])
    horizontal = bool(torch.rand((), generator=generator) < 0.5)
    vertical = bool(torch.rand((), generator=generator) < 0.5)
    axes = ([-1] if horizontal else []) + ([-2] if vertical else [])
    if axes:
        (low, target) = (low.flip(axes), target.flip(axes))
    if quarter_turn:
        low = torch.rot90(low, 1, dims=(-2, -1))
        target = torch.rot90(target, 1, dims=(-2, -1))
    return (low.contiguous(), target.contiguous())


def saturation_parameters(seed, epoch, sample_id, probability, factor_range):
    digest = hashlib.sha256(f'paired-saturation-v1:{seed}:{epoch}:{sample_id}'.encode()).digest()
    generator = torch.Generator().manual_seed(int.from_bytes(digest[:8], 'little') % (2 ** 63 - 1))
    applied = bool(torch.rand((), generator=generator) < probability)
    factor = factor_range[0] + (factor_range[1] - factor_range[0]) * float(torch.rand((), generator=generator))
    return (applied, factor)


class PairedImages(Dataset):

    def __init__(self, root: Path, split: str, crop_size=None, seed=100, flip_hv=False, geometric_augs=False, saturation_probability=0.0, saturation_range=(0.9, 1.1)):
        self.crop_size = tuple(crop_size) if crop_size is not None and split == 'train' else None
        self.flip_hv = bool(flip_hv) and split == 'train'
        self.geometric_augs = bool(geometric_augs) and split == 'train'
        self.seed = seed
        if not 0 <= saturation_probability <= 1 or not 0 < saturation_range[0] <= saturation_range[1]:
            raise ValueError('Invalid saturation probability or factor range')
        self.saturation_probability = float(saturation_probability) if split == 'train' else 0.0
        self.saturation_range = tuple(saturation_range)
        if split not in {'train', 'val'}:
            raise ValueError('Only train and val splits are supported.')
        (low, gt) = (root / split / 'lowlight', root / split / 'gt')
        self.low = image_files(low)
        targets = {p.name: p for p in image_files(gt)}
        if not self.low or {p.name for p in self.low} != set(targets):
            raise ValueError(f'Empty or unmatched lowlight/gt filenames: {split}')
        self.gt = [targets[p.name] for p in self.low]

    def __len__(self):
        return len(self.low)

    def __getitem__(self, key):
        quarter_turn = False
        if isinstance(key, tuple):
            (index, epoch, *orientation) = key
            if orientation:
                quarter_turn = bool(orientation[0])
        else:
            (index, epoch) = (key, 0)
        if self.geometric_augs and (not isinstance(key, tuple) or len(key) != 3):
            raise ValueError('D4 requires StepBatches with geometric_augs=True for shared batch orientation')
        (low, gt) = (read_rgb(self.low[index]), read_rgb(self.gt[index]))
        if low.shape != gt.shape:
            raise ValueError(f'Mismatched paired image shapes: {self.low[index]}')
        if self.crop_size is not None or self.flip_hv or self.geometric_augs:
            (low, gt) = augment_pair(low, gt, self.crop_size, self.seed, epoch, self.low[index].stem, quarter_turn=quarter_turn and self.geometric_augs)
        if self.saturation_probability:
            (applied, factor) = saturation_parameters(self.seed, epoch, self.low[index].stem, self.saturation_probability, self.saturation_range)
            if applied:
                from torchvision.transforms.functional import adjust_saturation
                low = adjust_saturation(low, factor)
                gt = adjust_saturation(gt, factor)
        return (low.contiguous(), gt.contiguous())


class StepBatches(Sampler):
    def __init__(self, count: int, batch: int, seed: int, start: int, stop: int, geometric_augs=False):
        (self.count, self.batch, self.seed) = (count, batch, seed)
        (self.start, self.stop) = (start, stop)
        self.geometric_augs = bool(geometric_augs)

    def __len__(self):
        return max(0, self.stop - self.start)

    def __iter__(self):
        per_epoch = math.ceil(self.count / self.batch)
        (previous_epoch, order) = (-1, [])
        for step in range(self.start, self.stop):
            (epoch, batch_index) = divmod(step, per_epoch)
            if epoch != previous_epoch:
                generator = torch.Generator().manual_seed(self.seed + epoch)
                order = torch.randperm(self.count, generator=generator).tolist()
                previous_epoch = epoch
            offset = batch_index * self.batch
            indices = order[offset:offset + self.batch]
            if self.geometric_augs:
                digest = hashlib.sha256(f'prior-a-d4:{self.seed}:{step}'.encode()).digest()
                quarter_turn = bool(digest[0] & 1)
                yield [(index, epoch, quarter_turn) for index in indices]
            else:
                yield [(index, epoch) for index in indices]
