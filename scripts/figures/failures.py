"""Figure B.1: centre, attribute and failure analysis of PRISM over eight seeds.

    python -m scripts.figures.failures --data datasets/polypgen_mixed

Uses the records of the complete model (arm M, prism_dn) in records/polypgen_ablation.csv.gz
and the 307 validation frames of the mixed partition (images and masks, for the
boundary contrast and the number of lesions). All statistics use the 276
lesion-bearing frames; per-image Dice is computed from the records at threshold 0.5.

  (a) per-image Dice averaged over the eight seeds, by acquisition centre (box),
      with the eight seed-wise centre means (dots);
  (b) mean Dice by joint quartiles of lesion area and boundary contrast, where the
      boundary contrast is |mean grey level inside - outside| in a 7 x 7 band
      around the reference contour;
  (c) frames sorted by mean Dice, with the seed min-max range and mean +/- SD;
  (d) attributes of the failure frames (mean Dice < 0.5); a frame can carry several:
      prediction empty in >= 4 of 8 seeds, lesion < 1% of the image, more than one
      lesion, contrast below the median of the lesion-bearing frames.

Writes fig_analysis_8seed.pdf/.png and failure_frames.csv to outputs/figures/.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage

from emc import records as R
from emc.plotting import FIGURES, plt
from emc.report import SPLITS

SEEDS = (42, 1337, 2024, 7, 3407, 12345, 271828, 314159)
FAIL, TINY, MISS_RATE, BAND = 0.5, 0.01, 0.5, 3


def boundary_contrast(gray: np.ndarray, gt: np.ndarray) -> float:
    k = 2 * BAND + 1
    m = gt.astype(np.float32)
    band = (ndimage.maximum_filter(m, size=k) - ndimage.minimum_filter(m, size=k)) > 0
    inner, outer = band & gt, band & ~gt
    if inner.sum() < 5 or outer.sum() < 5:
        return np.nan
    return float(abs(gray[inner].mean() - gray[outer].mean()))


def quartile(values: pd.Series) -> pd.Series:
    """Q1-Q4 by rank (stable order), equal-sized up to one frame."""
    order = np.argsort(values.to_numpy(), kind="mergesort")
    q = np.empty(len(values), int)
    q[order] = np.minimum(3, (4 * np.arange(len(values))) // len(values))
    return pd.Series(q + 1, index=values.index)


def frame_table(data: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per (seed, frame) Dice and per-frame attributes of the lesion-bearing frames."""
    rec = R.load("polypgen_ablation", models=["prism_dn"])
    rec = rec[(rec.threshold == 0.5) & (rec.min_area == 0)]
    assert sorted(rec.seed.unique()) == sorted(SEEDS), "the analysis needs all eight seeds"
    rec = rec[rec.gt_area > 0].copy()
    rec["dice"] = 2 * rec.inter / (rec.gt_area + rec.pred_area)
    rec["empty_pred"] = rec.pred_area == 0

    centre = pd.read_csv(SPLITS / "polypgen_frames.csv", dtype=str).set_index("frame").centre
    attrs = []
    for name, area in rec.groupby("image").gt_area.first().items():
        gt = np.asarray(Image.open(data / "val" / "masks" / name).convert("L")) / 255.0 >= 0.5
        assert int(gt.sum()) == area, f"{name}: mask does not match the records"
        gray = np.asarray(Image.open(data / "val" / "images" / name).convert("L")
                          .resize(gt.shape[::-1]), np.float32) / 255.0
        attrs.append({"image": name, "centre": centre[name], "gt_area": area,
                      "area_fraction": area / gt.size, "lesions": int(ndimage.label(gt)[1]),
                      "contrast": boundary_contrast(gray, gt)})
    attrs = pd.DataFrame(attrs).set_index("image")

    g = rec.groupby("image")
    attrs["dice_mean"] = g.dice.mean()
    attrs["dice_sd"] = g.dice.std(ddof=1)
    attrs["dice_min"], attrs["dice_max"] = g.dice.min(), g.dice.max()
    attrs["empty_rate"] = g.empty_pred.mean()
    attrs["area_q"] = quartile(attrs.area_fraction)
    attrs["contrast_q"] = quartile(attrs.contrast)
    return rec.merge(attrs[["centre"]], left_on="image", right_index=True), attrs


def failures(attrs: pd.DataFrame) -> pd.DataFrame:
    f = attrs[attrs.dice_mean < FAIL].copy()
    f["miss"] = f.empty_rate >= MISS_RATE
    f["tiny"] = f.area_fraction < TINY
    f["multiple"] = f.lesions > 1
    f["low_contrast"] = f.contrast < attrs.contrast.median()
    return f


def figure(rec: pd.DataFrame, attrs: pd.DataFrame, fail: pd.DataFrame) -> Path:
    with plt.rc_context(plt.rcParamsDefault):
        fig, ((a, b), (c, d)) = plt.subplots(2, 2, figsize=(13.5, 9.5))
        centres = sorted(attrs.centre.unique(), key=lambda s: int(s[1:]))

        a.boxplot([attrs.dice_mean[attrs.centre == k] for k in centres], tick_labels=centres,
                  showfliers=False)
        seed_means = rec.groupby(["seed", "centre"]).dice.mean()
        rng = np.random.default_rng(0)
        for i, k in enumerate(centres, 1):
            v = [seed_means[(s, k)] for s in SEEDS]
            a.scatter(i + rng.uniform(-0.08, 0.08, len(v)), v, s=22, zorder=3)
        a.set(ylim=(0, 1), ylabel="Per-image Dice", xlabel="Acquisition center")
        a.set_title("(a) Center-wise performance over eight seeds", loc="left")

        joint = attrs.groupby(["area_q", "contrast_q"]).dice_mean.agg(["mean", "size"])
        mat = np.array([[joint.loc[(i, j), "mean"] for j in range(1, 5)] for i in range(1, 5)])
        im = b.imshow(mat, vmin=0, vmax=1, origin="lower", aspect="auto")
        for i in range(4):
            for j in range(4):
                b.text(j, i, f"{mat[i, j]:.2f}\nn={joint.loc[(i + 1, j + 1), 'size']}",
                       ha="center", va="center", fontsize=9)
        q = ["Q1", "Q2", "Q3", "Q4"]
        b.set_xticks(range(4), q)
        b.set_yticks(range(4), q)
        b.set(xlabel="Boundary-contrast quartile (low to high)", ylabel="Lesion-area quartile (small to large)")
        b.set_title("(b) Mean Dice by lesion area and boundary contrast", loc="left")
        fig.colorbar(im, ax=b, fraction=0.046, pad=0.04, label="Eight-seed mean Dice")

        s = attrs.sort_values("dice_mean", kind="mergesort")
        x = np.arange(len(s))
        c.fill_between(x, s.dice_min, s.dice_max, alpha=0.20, label="Seed min--max")
        c.fill_between(x, np.clip(s.dice_mean - s.dice_sd, 0, 1), np.clip(s.dice_mean + s.dice_sd, 0, 1),
                       alpha=0.30, label="Mean +/- sample SD")
        c.plot(x, s.dice_mean, linewidth=1.4, label="Eight-seed mean")
        c.axhline(FAIL, linestyle="--", linewidth=1.0, label=f"Failure threshold ({FAIL:g})")
        c.set(ylim=(0, 1), xlim=(0, len(x) - 1), xlabel="Positive frames sorted by mean Dice",
              ylabel="Per-image Dice")
        c.legend(fontsize=8, loc="upper left")
        c.set_title("(c) Per-image performance and seed-wise variation", loc="left")

        labels = {"miss": f"Miss in >={MISS_RATE * 100:g}% seeds", "tiny": "Tiny lesion",
                  "multiple": "Multiple polyps", "low_contrast": "Low boundary contrast"}
        counts = np.array([int(fail[k].sum()) for k in labels])
        pct = 100 * counts / len(fail)
        pos = np.arange(len(labels))
        d.barh(pos, pct)
        d.set_yticks(pos, list(labels.values()))
        d.invert_yaxis()
        d.set_xlabel("Percentage of failure frames (%)")
        d.set_title(f"(d) Overlapping attributes among failures (n={len(fail)})", loc="left")
        for y, n, p in zip(pos, counts, pct):
            d.text(p + 0.8, y, f"{n} ({p:.0f}%)", va="center", fontsize=9)
        d.set_xlim(0, max(100, pct.max() + 15))

        fig.suptitle("PolypGen center, attribute, and failure analysis over eight seeds", fontsize=14)
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        FIGURES.mkdir(parents=True, exist_ok=True)
        fig.savefig(FIGURES / "fig_analysis_8seed.png", dpi=220, bbox_inches="tight")
        fig.savefig(FIGURES / "fig_analysis_8seed.pdf", bbox_inches="tight")
        plt.close(fig)
    return FIGURES / "fig_analysis_8seed.pdf"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=str(R.REPO / "datasets" / "polypgen_mixed"))
    a = ap.parse_args()
    rec, attrs = frame_table(Path(a.data))
    fail = failures(attrs)
    cols = ["centre", "gt_area", "area_fraction", "lesions", "contrast", "dice_mean", "dice_sd", "empty_rate",
            "miss", "tiny", "multiple", "low_contrast"]
    FIGURES.mkdir(parents=True, exist_ok=True)
    fail[cols].to_csv(FIGURES / "failure_frames.csv")
    print(f"{len(attrs)} lesion-bearing frames, {len(fail)} failures (mean Dice < {FAIL}); "
          + ", ".join(f"{k} {int(fail[k].sum())}" for k in ("miss", "tiny", "multiple", "low_contrast")))
    print(figure(rec, attrs, fail))


if __name__ == "__main__":
    main()
