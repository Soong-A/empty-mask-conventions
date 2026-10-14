"""Training loop pieces shared by every model and protocol."""
from __future__ import annotations

import copy
import os
import random

import numpy as np
import torch

from emc import models


def set_seed(seed: int, deterministic: bool = True) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cudnn.deterministic = deterministic


class EMA:
    """Exponential moving average of the model weights, updated after every optimiser step."""

    def __init__(self, model, decay: float = 0.999):
        self.module = copy.deepcopy(model).eval()
        for p in self.module.parameters():
            p.requires_grad_(False)
        self.decay = float(decay)

    @torch.no_grad()
    def update(self, model):
        msd, esd = model.state_dict(), self.module.state_dict()
        for k, ev in esd.items():
            mv = msd[k].to(ev.device, non_blocking=True)
            if ev.dtype.is_floating_point:
                ev.mul_(self.decay).add_(mv, alpha=1.0 - self.decay)
            else:
                ev.copy_(mv)


def optimiser_and_schedule(model, protocol):
    opt = torch.optim.AdamW(model.parameters(), lr=protocol.lr, betas=(0.9, 0.999), eps=1e-8,
                            weight_decay=protocol.weight_decay, amsgrad=False)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=protocol.t_max, eta_min=protocol.eta_min)
    return opt, sch


def train_one_epoch(loader, model, objective, optimiser, ema, device) -> float:
    model.train()
    losses = []
    for batch in loader:
        images = batch[0].to(device, non_blocking=True).float()
        targets = batch[1].to(device, non_blocking=True).float()
        points = batch[2].to(device, non_blocking=True).float() if len(batch) > 2 else None
        optimiser.zero_grad()
        loss = objective(model(images), targets, points)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite training loss {float(loss)}")
        loss.backward()
        optimiser.step()
        ema.update(model)
        losses.append(float(loss.detach()))
    return float(np.mean(losses))


@torch.no_grad()
def evaluate(loader, model, key: str, device, threshold: float = 0.5) -> dict:
    """Per-image overlap on a split, as used for checkpoint selection.

    miou / dice: per-image IoU and Dice with additive smoothing 1e-5 (two empty masks
    score 1), averaged over all images; dice_pos: Dice over images with a non-empty
    reference; q and r0 as in Section 2.4.
    """
    model.eval()
    eps = 1e-5
    iou, dice, dice_pos, n_miss, n_fa, n_pos, n_neg = [], [], [], 0, 0, 0, 0
    for img, msk in loader:
        prob = models.probability(key, model(img.to(device).float())).cpu().numpy()
        ref = msk.numpy()
        for p, r in zip(prob, ref):
            pred, gt = p.reshape(-1) >= threshold, r.reshape(-1) >= 0.5
            inter, union, ps, gs = (pred & gt).sum(), (pred | gt).sum(), pred.sum(), gt.sum()
            iou.append((inter + eps) / (union + eps))
            dice.append((2 * inter + eps) / (ps + gs + eps))
            if gs:
                n_pos += 1
                n_miss += ps == 0
                dice_pos.append(2 * inter / (ps + gs))
            else:
                n_neg += 1
                n_fa += ps > 0
    return {"miou": float(np.mean(iou)), "dice": float(np.mean(dice)),
            "dice_pos": float(np.mean(dice_pos)) if dice_pos else float("nan"),
            "q": n_miss / n_pos if n_pos else float("nan"), "r0": n_fa / n_neg if n_neg else float("nan"),
            "n": len(iou), "n_pos": n_pos}


def selection_score(metrics: dict, selection: str) -> float:
    if selection == "miou_plus_dice":
        return metrics["miou"] + metrics["dice"]
    if selection == "dice_pos":
        return metrics["dice_pos"] if np.isfinite(metrics["dice_pos"]) else -1.0
    raise ValueError(selection)
