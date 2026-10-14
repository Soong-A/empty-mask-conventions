"""Training objectives: each architecture keeps its established objective (Section 2.2).

``build_objective(model, protocol, batch_dice)`` returns a module called as
``objective(output, target, points)`` with the raw model output. The objectives
are those of the original training code; where the two benchmarks used different
versions of an objective, both are reproduced:

    model       PolypGen (protocol "polypgen")            SIIM-ACR (protocol "siim")
    UNet++      mean of BceDice(1,1) over the 4 heads     same, predictions clamped
    UNeXt-S     0.5 BCE-with-logits + Dice (smooth 1e-5)  0.5 BCE + Dice on sigmoid (smooth 1)
    MALUNet,    BceDice(1,1)                              same, predictions clamped
    MobileUNet,
    UNet-Tiny
    EGE-UNet    BceDice(1,1) main + 0.1..0.5 x BceDice(1,1) deep supervision
                                                          BceDice(1,2) main + 0.1..0.5 x BceDice(0.5,1)
    LB-UNet     BceDice(1,2) main + 0.1..0.5 x BceDice(0.5,1) deep supervision
                + 0.1..0.3 x BceDice(0.5,1) on boundary points (both benchmarks)
    PRISM       sum_s lambda_s BceDice(0.5,0.5)(m_s) + boundary-weighted BCE + Dice on m6,
                lambda = (0.10, 0.15, 0.20, 0.25, 0.30, 0.40) (Equation A.1)

BceDice(wb, wd) = wb * BCE + wd * (1 - Dice). Dice is per image and averaged over
the batch ("per-image Dice"); on SIIM-ACR the main runs aggregate it over the batch
(``batch_dice=True``), the stress test of Appendix D keeps it per image.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class BCE(nn.Module):
    def __init__(self, clamp: bool):
        super().__init__()
        self.clamp = clamp

    def forward(self, pred, target):
        if self.clamp:
            pred, target = pred.clamp(1e-6, 1.0 - 1e-6), target.clamp(0.0, 1.0)
        b = pred.size(0)
        return F.binary_cross_entropy(pred.view(b, -1), target.view(b, -1))


class Dice(nn.Module):
    """Soft Dice loss, per image (mean over the batch) or over the whole batch."""

    def __init__(self, smooth: float = 1.0, batch: bool = False, clamp: bool = False):
        super().__init__()
        self.smooth, self.batch, self.clamp = smooth, batch, clamp

    def forward(self, pred, target):
        if self.clamp:
            pred, target = pred.clamp(0.0, 1.0), target.clamp(0.0, 1.0)
        rows = 1 if self.batch else pred.size(0)
        p, t = pred.reshape(rows, -1), target.reshape(rows, -1)
        dice = (2 * (p * t).sum(1) + self.smooth) / (p.sum(1) + t.sum(1) + self.smooth)
        return 1 - dice.mean()


class BceDice(nn.Module):
    def __init__(self, wb=1.0, wd=1.0, clamp=False, batch=False, smooth=1.0):
        super().__init__()
        self.wb, self.wd = wb, wd
        self.bce = BCE(clamp)
        self.dice = Dice(smooth, batch, clamp)

    def forward(self, pred, target):
        return self.wd * self.dice(pred, target) + self.wb * self.bce(pred, target)


class SingleHead(nn.Module):
    """One output map (MALUNet, MobileUNet, UNet-Tiny)."""

    def __init__(self, term, clamp_pred=False):
        super().__init__()
        self.term, self.clamp_pred = term, clamp_pred

    def forward(self, out, target, points=None):
        if self.clamp_pred:
            out = out.clamp(1e-6, 1.0 - 1e-6)
        return self.term(out, target)


class NestedHeads(nn.Module):
    """UNet++: equal-weight mean over the three auxiliary heads and the final head."""

    def __init__(self, term, clamp_pred=False):
        super().__init__()
        self.term, self.clamp_pred = term, clamp_pred

    def forward(self, output, target, points=None):
        aux, out = output
        heads = list(aux) + [out]
        if self.clamp_pred:
            heads = [h.clamp(1e-6, 1.0 - 1e-6) for h in heads]
        return torch.stack([self.term(h, target) for h in heads]).mean()


class DeepSupervision(nn.Module):
    """EGE-UNet: main head plus five deep-supervision heads weighted 0.1 ... 0.5."""

    def __init__(self, main, aux, clamp_pred=False, weights=(0.1, 0.2, 0.3, 0.4, 0.5)):
        super().__init__()
        self.main, self.aux, self.clamp_pred, self.weights = main, aux, clamp_pred, weights

    def forward(self, output, target, points=None):
        gt_pre, out = output
        if self.clamp_pred:
            out, gt_pre = out.clamp(1e-6, 1.0 - 1e-6), [g.clamp(0.0, 1.0) for g in gt_pre]
        loss = self.main(out, target)
        for g, w in zip(gt_pre, self.weights):
            loss = loss + w * self.aux(g, target)
        return loss


class LBUNetObjective(nn.Module):
    """LB-UNet: main BceDice(1,2), deep supervision and boundary-point heads BceDice(0.5,1)."""

    def __init__(self, batch=False):
        super().__init__()
        self.main = BceDice(1, 2, clamp=True, batch=batch)
        self.aux = BceDice(0.5, 1, clamp=True, batch=batch)

    def forward(self, output, target, points):
        gt_pre, key_points, out = output
        if points is None:
            raise RuntimeError("LB-UNet training needs boundary-point targets (scripts/lbunet_points.py)")
        target, points = target.clamp(0.0, 1.0), points.clamp(0.0, 1.0)
        main = self.main(out.clamp(1e-6, 1.0 - 1e-6), target)
        boundary = sum(w * self.aux(k, points) for k, w in zip(key_points, (0.1, 0.2, 0.3)))
        deep = sum(w * self.aux(g.clamp(0.0, 1.0), target)
                   for g, w in zip(gt_pre[:5], (0.1, 0.2, 0.3, 0.4, 0.5)))
        return main + deep + boundary


class UNeXtLogits(nn.Module):
    """UNeXt's published BCEDiceLoss on logits: 0.5 BCE-with-logits + soft Dice (smooth 1e-5)."""

    def forward(self, logits, target, points=None):
        bce = F.binary_cross_entropy_with_logits(logits, target)
        return 0.5 * bce + Dice(smooth=1e-5)(torch.sigmoid(logits), target)


class UNeXtProbabilities(nn.Module):
    """SIIM-ACR version: the same weights on sigmoid probabilities, with the shared Dice term."""

    def __init__(self, batch=False):
        super().__init__()
        self.term = BceDice(0.5, 1.0, clamp=True, batch=batch)

    def forward(self, logits, target, points=None):
        return self.term(torch.sigmoid(logits).clamp(1e-6, 1.0 - 1e-6), target.clamp(0.0, 1.0))


class BoundaryWeightedBCE(nn.Module):
    """BCE weighted 1 + 5 b_3(y), b_3 = dilate_3(y) - erode_3(y) over a 7 x 7 neighbourhood."""

    def __init__(self, band_width=3, w_edge=5.0):
        super().__init__()
        self.band_width, self.w_edge = band_width, w_edge

    def forward(self, pred, target):
        with torch.no_grad():
            t = (target > 0.5).float()
            k = 2 * self.band_width + 1
            band = F.max_pool2d(t, k, 1, self.band_width) + F.max_pool2d(-t, k, 1, self.band_width)
            weight = 1.0 + self.w_edge * band
        bce = F.binary_cross_entropy(pred.clamp(1e-6, 1 - 1e-6), target, reduction="none")
        return (bce * weight).mean()


class PRISMObjective(nn.Module):
    def __init__(self, batch=False, clamp_final=False,
                 deep_weights=(0.10, 0.15, 0.20, 0.25, 0.30, 0.40)):
        super().__init__()
        self.deep_weights, self.clamp_final = deep_weights, clamp_final
        self.deep = BceDice(0.5, 0.5, clamp=True, batch=batch)
        self.bwbce = BoundaryWeightedBCE()
        self.dice = Dice(batch=batch)

    def forward(self, output, target, points=None):
        gt_pre, _, out = output
        target = target.clamp(0.0, 1.0)
        total = 0.0
        for w, m in zip(self.deep_weights, gt_pre):
            t = target if target.shape[-2:] == m.shape[-2:] else \
                F.interpolate(target, size=m.shape[-2:], mode="bilinear", align_corners=False).clamp(0.0, 1.0)
            total = total + w * self.deep(m, t)
        if self.clamp_final:
            out = out.clamp(1e-6, 1.0 - 1e-6)
        return total + (self.bwbce(out, target) + self.dice(out, target))


def build_objective(model: str, protocol: str, batch_dice: bool = False) -> nn.Module:
    if protocol == "polypgen":
        if batch_dice:
            raise ValueError("the PolypGen objectives use per-image Dice")
        plain = BceDice(1, 1)
        return {
            "unetpp": lambda: NestedHeads(plain),
            "unext_s": lambda: UNeXtLogits(),
            "malunet": lambda: SingleHead(plain),
            "mobileunet": lambda: SingleHead(plain),
            "tinyunet": lambda: SingleHead(plain),
            "egeunet": lambda: DeepSupervision(plain, plain),
            "lbunet": lambda: LBUNetObjective(),
            "prism": lambda: PRISMObjective(),
        }[model]()
    if protocol == "siim":
        b = batch_dice
        plain = BceDice(1, 1, clamp=True, batch=b)
        return {
            "unetpp": lambda: NestedHeads(plain, clamp_pred=True),
            "unext_s": lambda: UNeXtProbabilities(b),
            "malunet": lambda: SingleHead(plain, clamp_pred=True),
            "mobileunet": lambda: SingleHead(plain, clamp_pred=True),
            "tinyunet": lambda: SingleHead(plain, clamp_pred=True),
            "egeunet": lambda: DeepSupervision(BceDice(1, 2, clamp=True, batch=b),
                                               BceDice(0.5, 1, clamp=True, batch=b), clamp_pred=True),
            "lbunet": lambda: LBUNetObjective(b),
            "prism": lambda: PRISMObjective(b, clamp_final=True),
        }[model]()
    raise ValueError(f"unknown protocol {protocol!r}")
