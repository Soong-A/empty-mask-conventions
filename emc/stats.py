"""Resampling intervals used in the paper."""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from .conventions import DIAGONAL


def clopper_pearson(k: int, n: int, level: float = 0.95) -> tuple[float, float]:
    """Exact two-sided binomial interval."""
    from scipy.stats import beta
    a = (1 - level) / 2
    lo = 0.0 if k == 0 else float(beta.ppf(a, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(1 - a, k + 1, n - k))
    return lo, hi


def frame_bootstrap_reorders(df: pd.DataFrame, models: list, B: int = 10000,
                             seed: int = 20260801, d: float = DIAGONAL) -> np.ndarray:
    """Reordered model pairs between deletion (X) and P(d) under frame resampling.

    Frames are resampled with replacement, paired across models and seeds (the
    same frame indices for every run); each replicate averages the seed values of
    every model, as the point estimate does. Returns the count per replicate.
    """
    seeds = sorted(df["seed"].unique())
    n = int(df.groupby(["model", "seed"]).size().iloc[0])
    rng = np.random.default_rng(seed)
    idx = np.stack([rng.choice(n, size=n, replace=True) for _ in range(B)])

    x = np.zeros((len(models), B))
    p = np.zeros((len(models), B))
    for i, m in enumerate(models):
        for s in seeds:
            g = df[(df["model"] == m) & (df["seed"] == s)].sort_values("idx")
            det = ((g["gt_area"] > 0) & (g["pred_area"] > 0)).to_numpy()
            one_sided = ((g["gt_area"] > 0) ^ (g["pred_area"] > 0)).to_numpy()
            hd = np.where(det, g["hd95_det"].to_numpy(float), 0.0)
            s_hd, k_det, k_1s = hd[idx].sum(1), det[idx].sum(1), one_sided[idx].sum(1)
            with np.errstate(invalid="ignore", divide="ignore"):
                x[i] += s_hd / k_det / len(seeds)
            p[i] += (s_hd + k_1s * d) / n / len(seeds)
    counts = np.zeros(B, int)
    for i, j in itertools.combinations(range(len(models)), 2):
        counts += (np.sign(x[i] - x[j]) * np.sign(p[i] - p[j]) < 0)
    return counts


def paired_seed_bootstrap(diffs: dict, B: int = 100000, seed: int = 20260727,
                          level: float = 0.95) -> dict:
    """Percentile intervals for paired-seed mean differences.

    ``diffs`` maps a column name to the per-seed differences (seeds in a fixed
    order). One set of resampled seed indices is drawn and shared by every column,
    so the columns of one contrast are resampled jointly.
    """
    lengths = {len(v) for v in diffs.values()}
    if len(lengths) != 1:
        raise ValueError("all columns need the same seeds")
    n = lengths.pop()
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(B, n))
    q = [(1 - level) / 2, 1 - (1 - level) / 2]
    out = {}
    for name, v in diffs.items():
        v = np.asarray(v, float)
        lo, hi = np.quantile(v[idx].mean(axis=1), q)
        out[name] = (float(v.mean()), float(lo), float(hi))
    return out
