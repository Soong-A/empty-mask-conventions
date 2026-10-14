"""Empty-mask conventions as functions of a set of per-image records.

Notation follows Section 2.4 of the paper. For one model run (model, seed):

    N, N+, N0        all images, non-empty references, empty references
    q                fraction of non-empty references with an empty prediction
                     (complete misses)
    r0               fraction of empty references with any predicted foreground
    n_det            images on which both masks are non-empty, N+ (1 - q)

Overlap (Dice, Boundary IoU): a one-sided-empty image scores 0 and two empty
masks score the constant c (c_empty = 0 or 1). ``*_pos`` is the mean over
non-empty references only and carries no convention.

Distance (HD95):
    P(d)   substitution: one-sided empty -> d, both empty -> 0
    Z      substitution with d = 0
    X      deletion (MONAI): every image with an empty mask on either side
           leaves the mean, so the value is the mean over the n_det images
    X'     one-sided-empty images leave the mean, two empty masks score 0

``run_metrics`` evaluates all of them for one run; ``summarise`` averages runs
over training seeds, the replication unit of the paper.
"""
from __future__ import annotations

import itertools
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

DIAGONAL = float(np.hypot(256, 256))      # 362.04 px, image diagonal at 256 x 256
PENALTIES = {"P128": 128.0, "P256": 256.0, "P362": DIAGONAL}


def run_metrics(rows: pd.DataFrame, d_pos: float = DIAGONAL) -> dict:
    """Every count, rate and convention for the records of ONE model run."""
    gt = rows["gt_area"].to_numpy() > 0
    pr = rows["pred_area"].to_numpy() > 0
    inter = rows["inter"].to_numpy(float)
    denom = rows["gt_area"].to_numpy(float) + rows["pred_area"].to_numpy(float)
    hd = rows["hd95_det"].to_numpy(float)
    bi = rows["biou_inter"].to_numpy(float)
    bu = rows["biou_union"].to_numpy(float)

    N, n_pos, n_neg = len(rows), int(gt.sum()), int((~gt).sum())
    det = gt & pr
    both_empty = ~gt & ~pr
    n_miss, n_fa = int((gt & ~pr).sum()), int((~gt & pr).sum())
    n_1s = n_miss + n_fa
    hd_sum = float(hd[det].sum())

    with np.errstate(invalid="ignore", divide="ignore"):
        dice = np.where(denom > 0, 2.0 * inter / denom, np.nan)
        biou = np.where(bu > 0, bi / bu, np.nan)

    out = {
        "N": N, "N_pos": n_pos, "N_neg": n_neg,
        "n_det": int(det.sum()), "n_miss": n_miss, "n_fa": n_fa,
        "n_both_empty": int(both_empty.sum()),
        "q": n_miss / n_pos if n_pos else np.nan,
        "r0": n_fa / n_neg if n_neg else np.nan,
        "dice_pos": float(np.nanmean(dice[gt])) if n_pos else np.nan,
        "biou_pos": float(np.nan_to_num(biou[gt]).mean()) if n_pos else np.nan,
        "hd95_det": hd_sum / det.sum() if det.any() else np.nan,
        "hd95_pos": float(np.where(det, hd, d_pos)[gt].mean()) if n_pos else np.nan,
        "hd95_X": hd_sum / det.sum() if det.any() else np.nan,
        "hd95_Xprime": (hd_sum / (det.sum() + both_empty.sum())
                        if det.any() or both_empty.any() else np.nan),
        "hd95_Z": hd_sum / N,
    }
    for c in (0, 1):
        out[f"dice_c{c}"] = float(np.where(both_empty, c, np.nan_to_num(dice)).mean())
        out[f"biou_c{c}"] = float(np.where(bu == 0, c, np.nan_to_num(biou)).mean())
    for name, d in PENALTIES.items():
        out[f"hd95_{name}"] = (hd_sum + n_1s * d) / N
    return out


def per_run(df: pd.DataFrame) -> pd.DataFrame:
    """``run_metrics`` for every (model, seed) in a records frame."""
    rows = []
    for (model, seed), g in df.groupby(["model", "seed"], sort=False):
        rows.append({"model": model, "seed": int(seed), **run_metrics(g)})
    return pd.DataFrame(rows)


def summarise(runs: pd.DataFrame, models: Sequence[str] | None = None) -> pd.DataFrame:
    """Seed means and seed standard deviations (ddof=1), one row per model."""
    num = [c for c in runs.columns if c not in ("model", "seed")]
    g = runs.groupby("model", sort=False)[num]
    out = g.mean().join(g.std(ddof=1).fillna(0.0), rsuffix="_sd")
    out["n_seeds"] = runs.groupby("model", sort=False).size()
    return out.reindex(list(models)) if models is not None else out


# ---------------------------------------------------------------------------
# Identities of Section 2.4, checked numerically against the aggregates
# ---------------------------------------------------------------------------

def identity_residuals(m: dict) -> dict:
    """Absolute residuals of Equations (1)-(3) for one run (should be ~0)."""
    N, Np, Nn, q, r = m["N"], m["N_pos"], m["N_neg"], m["q"], m["r0"]
    res = {}
    for c in (0, 1):
        res[f"eq1_dice_c{c}"] = abs(Np / N * m["dice_pos"] + Nn / N * (1 - r) * c - m[f"dice_c{c}"])
    res["eq1_hd95_P362"] = abs(Np / N * m["hd95_pos"] + Nn / N * r * DIAGONAL - m["hd95_P362"])
    slope = Nn / N * r + Np / N * q
    for name, d in PENALTIES.items():
        res[f"eq2_{name}"] = abs(m["hd95_Z"] + slope * d - m[f"hd95_{name}"])
    if m["n_det"]:
        num = Np * (1 - q) * m["hd95_det"]
        den = Np * (1 - q) + Nn * (1 - r)
        res["eq3_Xprime"] = abs(num / den - m["hd95_Xprime"])
    res["X_denominator"] = abs(m["n_det"] - Np * (1 - q))
    return res


# ---------------------------------------------------------------------------
# Rankings
# ---------------------------------------------------------------------------

def ranks(values: dict, higher_is_better: bool) -> dict:
    """Rank 1 = best."""
    order = sorted(values, key=lambda m: -values[m] if higher_is_better else values[m])
    return {m: i + 1 for i, m in enumerate(order)}


def reordered_pairs(a: dict, b: dict) -> list:
    """Model pairs whose order differs between two columns (same direction of goodness)."""
    models = list(a)
    return [(m, n) for m, n in itertools.combinations(models, 2)
            if np.sign(a[m] - a[n]) * np.sign(b[m] - b[n]) < 0]


def kendall_tau_b(a: dict, b: dict) -> float:
    from scipy.stats import kendalltau
    models = list(a)
    return float(kendalltau([a[m] for m in models], [b[m] for m in models]).statistic)


def seed_reversals(runs: pd.DataFrame, pairs: Iterable, col_a: str, col_b: str) -> list:
    """For each pair, the number of training seeds in which the two columns order the
    pair differently when that seed is scored on its own (seeds paired by value).

    The order may flip in either direction within a seed; on the PolypGen mixed
    partition the LB-UNet/PRISM pair flips the other way in seed 1337.
    """
    v = runs.set_index(["model", "seed"])
    seeds = sorted(runs["seed"].unique())
    return [sum(1 for s in seeds
                if np.sign(v.at[(m, s), col_a] - v.at[(n, s), col_a])
                * np.sign(v.at[(m, s), col_b] - v.at[(n, s), col_b]) < 0)
            for m, n in pairs]


def crossings(lines: dict, lo: float, hi: float) -> list:
    """Where two straight lines y = a + b x (``lines``: name -> (a, b)) cross in [lo, hi]."""
    out = []
    for m, n in itertools.combinations(list(lines), 2):
        (a0, a1), (b0, b1) = lines[m], lines[n]
        if abs(a1 - b1) > 1e-12:
            x = (b0 - a0) / (a1 - b1)
            if lo <= x <= hi:
                out.append({"pair": (m, n), "x": float(x), "y": float(a0 + a1 * x)})
    return sorted(out, key=lambda c: c["x"])
