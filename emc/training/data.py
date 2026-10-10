"""Prepared datasets: <root>/<split>/images and <root>/<split>/masks with identical file names.

LB-UNet additionally reads boundary-point targets from <root>/train/points_boundary2
(scripts/lbunet_points.py). Masks are read as grey levels / 255.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, Sampler


def _listing(directory: Path) -> list:
    return sorted(p.name for p in directory.iterdir() if not p.name.startswith("."))


class FolderDataset(Dataset):
    def __init__(self, root, split: str, transform, points: bool = False):
        root = Path(root)
        self.images_dir, self.masks_dir = root / split / "images", root / split / "masks"
        self.names = _listing(self.images_dir)
        if self.names != _listing(self.masks_dir):
            raise RuntimeError(f"{root / split}: image and mask file names differ")
        self.points_dir = root / split / "points_boundary2" if points else None
        if self.points_dir is not None and _listing(self.points_dir) != self.names:
            raise RuntimeError(f"{self.points_dir}: missing boundary points; run scripts/lbunet_points.py")
        self.transform = transform

    def __len__(self):
        return len(self.names)

    @staticmethod
    def _gray(path):
        return np.expand_dims(np.array(Image.open(path).convert("L")), axis=2) / 255.0

    def __getitem__(self, i):
        name = self.names[i]
        img = np.array(Image.open(self.images_dir / name).convert("RGB"))
        sample = (img, self._gray(self.masks_dir / name))
        if self.points_dir is not None:
            sample += (self._gray(self.points_dir / name),)
        return self.transform(sample)

    def empty_masks(self) -> list[bool]:
        return [Image.open(self.masks_dir / n).convert("L").getextrema()[1] == 0 for n in self.names]


class BalancedEpochSampler(Sampler):
    """SIIM-ACR training stream: every lesion-bearing image plus an equal number of
    lesion-free images drawn without replacement each epoch (Appendix A.2).
    The draw depends only on (seed, epoch)."""

    def __init__(self, empty: list[bool], seed: int, neg_per_pos: float = 1.0):
        self.pos = [i for i, e in enumerate(empty) if not e]
        self.neg = [i for i, e in enumerate(empty) if e]
        self.n_neg = min(int(round(len(self.pos) * neg_per_pos)), len(self.neg))
        self.seed, self.epoch = int(seed), 0

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed * 100003 + self.epoch)
        picked = torch.randperm(len(self.neg), generator=g)[:self.n_neg].tolist()
        indices = self.pos + [self.neg[i] for i in picked]
        order = torch.randperm(len(indices), generator=g).tolist()
        return iter([indices[i] for i in order])

    def __len__(self):
        return len(self.pos) + self.n_neg
