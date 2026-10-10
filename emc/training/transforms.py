"""Paired image/mask transforms (one implementation for both benchmarks).

Samples are tuples (image, mask) or (image, mask, points). Image and masks enter as
H x W x C numpy arrays (image 0-255, mask 0-1) and leave as C x H x W tensors.

Two behaviours differ between the benchmarks and are therefore parameters:
  * Rotation. PolypGen runs draw ONE angle in [0, 360) per run, when the transform
    is built, and rotate the selected images by that angle; SIIM-ACR runs draw a
    new angle in [-15, 15] for every image (``per_sample``).
  * Resize. PolypGen resizes masks with the same bilinear filter as images
    (torchvision default); SIIM-ACR resizes masks with nearest neighbour.
"""
from __future__ import annotations

import random

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF

# Per-dataset intensity statistics of the training split (computed once, fixed).
NORMALISATION = {
    "polypgen": (89.867, 68.3348),
    "polypgen_c4": (90.0917, 68.6167),
    "siim": (129.1255, 65.1556),
}


class Compose:
    def __init__(self, transforms):
        self.transforms = list(transforms)

    def __call__(self, data):
        for t in self.transforms:
            data = t(data)
        return data


class Normalize:
    """Standardise with the dataset statistics, then min-max rescale each image to [0, 255]."""

    def __init__(self, dataset: str):
        self.mean, self.std = NORMALISATION[dataset]

    def __call__(self, data):
        img = (data[0] - self.mean) / self.std
        lo, hi = np.min(img), np.max(img)
        img = np.zeros_like(img, dtype=np.float64) if not hi > lo else (img - lo) / (hi - lo) * 255.0
        return (img,) + tuple(data[1:])


class ToTensor:
    def __call__(self, data):
        return tuple(torch.tensor(a).permute(2, 0, 1) for a in data)


class ColorJitter:
    """Brightness, contrast and saturation jitter of the image only."""

    def __init__(self, p=0.5, brightness=0.2, contrast=0.2, saturation=0.2):
        self.p, self.b, self.c, self.s = p, float(brightness), float(contrast), float(saturation)

    def _jitter(self, img):
        if self.b > 0:
            img = img * (1.0 + (random.random() * 2 - 1) * self.b)
        if self.c > 0:
            f = 1.0 + (random.random() * 2 - 1) * self.c
            mean = img.mean()
            img = (img - mean) * f + mean
        if self.s > 0:
            f = 1.0 + (random.random() * 2 - 1) * self.s
            gray = (0.299 * img[0] + 0.587 * img[1] + 0.114 * img[2]).unsqueeze(0)
            img = (img - gray) * f + gray
        return img.clamp(0, 255)

    def __call__(self, data):
        if random.random() > self.p:
            return data
        return (self._jitter(data[0]),) + tuple(data[1:])


class GaussianNoise:
    def __init__(self, p=0.3, std=8.0):
        self.p, self.std = p, float(std)

    def __call__(self, data):
        if random.random() > self.p:
            return data
        img = (data[0] + torch.randn_like(data[0]) * self.std).clamp(0, 255)
        return (img,) + tuple(data[1:])


class HorizontalFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, data):
        return tuple(TF.hflip(a) for a in data) if random.random() < self.p else data


class VerticalFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, data):
        return tuple(TF.vflip(a) for a in data) if random.random() < self.p else data


class Rotation:
    def __init__(self, p=0.5, degree=(0, 360), per_sample=False):
        self.p, self.degree, self.per_sample = p, (float(degree[0]), float(degree[1])), per_sample
        self.angle = None if per_sample else random.uniform(*self.degree)

    def __call__(self, data):
        if random.random() >= self.p:
            return data
        angle = random.uniform(*self.degree) if self.per_sample else self.angle
        return tuple(TF.rotate(a, angle) for a in data)


class ElasticDeform:
    """Smooth random displacement field (control grid, bilinear upsampling)."""

    def __init__(self, p=0.5, alpha=6.0, control_grid=8):
        self.p, self.alpha, self.cg = p, float(alpha), int(control_grid)

    @staticmethod
    def _warp(x, grid, mode, padding_mode):
        return F.grid_sample(x.unsqueeze(0).float(), grid, mode=mode, padding_mode=padding_mode,
                             align_corners=True).squeeze(0)

    def __call__(self, data):
        if random.random() > self.p:
            return data
        img = data[0]
        _, H, W = img.shape
        dx = F.interpolate((torch.rand(1, 1, self.cg, self.cg) - 0.5) * 2 * self.alpha, size=(H, W),
                           mode="bilinear", align_corners=True).squeeze()
        dy = F.interpolate((torch.rand(1, 1, self.cg, self.cg) - 0.5) * 2 * self.alpha, size=(H, W),
                           mode="bilinear", align_corners=True).squeeze()
        yy, xx = torch.meshgrid(torch.linspace(-1.0, 1.0, H), torch.linspace(-1.0, 1.0, W), indexing="ij")
        grid = torch.stack([xx + 2.0 * dx / max(W - 1, 1), yy + 2.0 * dy / max(H - 1, 1)], dim=-1).unsqueeze(0)
        out = [self._warp(img, grid, "bilinear", "reflection")]
        out += [self._warp(m, grid, "nearest", "zeros") for m in data[1:]]
        return tuple(out)


class ScaleCrop:
    """Random rescaling, then centre crop (zoom in) or reflection padding (zoom out)."""

    def __init__(self, p=0.3, scale_range=(0.85, 1.15)):
        self.p, self.lo, self.hi = p, float(scale_range[0]), float(scale_range[1])

    def __call__(self, data):
        if random.random() > self.p:
            return data
        img = data[0]
        _, H, W = img.shape
        scale = random.uniform(self.lo, self.hi)
        nH, nW = max(int(round(H * scale)), 4), max(int(round(W * scale)), 4)
        out = [TF.resize(img, [nH, nW], interpolation=TF.InterpolationMode.BILINEAR, antialias=True)]
        out += [TF.resize(m, [nH, nW], interpolation=TF.InterpolationMode.NEAREST) for m in data[1:]]
        if scale >= 1.0:
            top, left = (nH - H) // 2, (nW - W) // 2
            return tuple(a[:, top:top + H, left:left + W] for a in out)
        pad_h, pad_w = H - nH, W - nW
        pad = (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2)
        res = [F.pad(out[0].unsqueeze(0), pad, mode="reflect").squeeze(0)]
        res += [F.pad(m.unsqueeze(0), pad, mode="constant", value=0).squeeze(0) for m in out[1:]]
        return tuple(res)


class Resize:
    def __init__(self, size=256, nearest_masks=False):
        self.size, self.nearest_masks = [size, size], nearest_masks

    def __call__(self, data):
        if not self.nearest_masks:
            return tuple(TF.resize(a, self.size) for a in data)
        img = TF.resize(data[0], self.size, interpolation=TF.InterpolationMode.BILINEAR, antialias=True)
        return (img,) + tuple(TF.resize(m, self.size, interpolation=TF.InterpolationMode.NEAREST)
                              for m in data[1:])
