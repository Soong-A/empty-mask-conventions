"""Convention-free per-image evaluation records.

A record describes one image, scored for one trained model (model, seed) at one
binarisation setting (threshold, min_area). It stores only quantities that are
defined for every image, whatever the masks contain:

    gt_area, pred_area, inter   foreground pixels of the reference, of the
                                prediction, and of their intersection
    hd95_det                    95th-percentile boundary distance, stored only
                                when BOTH masks are non-empty (empty otherwise)
    biou_inter, biou_union      intersection and union of the two boundary bands

Every empty-mask convention studied in the paper (substitution families P and Z,
exclusion families X and X') is then a function of the record set; see
``emc.conventions``. No convention is applied when a record is written.

All released records share one schema (``COLUMNS``) and live in
``records/<partition>.csv.gz``, one file per evaluation partition.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
RECORDS_DIR = REPO / "records"

COLUMNS = [
    "partition", "model", "seed", "idx", "image", "threshold", "min_area",
    "gt_area", "pred_area", "inter", "hd95_det", "biou_inter", "biou_union",
]


@dataclass(frozen=True)
class Partition:
    """An evaluation partition and how its released records were computed."""
    name: str
    description: str
    n_frames: int
    geometry: str          # key of GEOMETRIES used for hd95_det and the boundary band


PARTITIONS = {
    p.name: p for p in (
        Partition("polypgen_mixed", "PolypGen mixed reporting partition (Table 4, Panel A)",
                  307, "mask"),
        Partition("polypgen_ablation", "PolypGen mixed partition, PRISM ablation arms over eight seeds "
                  "(Appendix B.4, Figure B.1)", 307, "mask"),
        Partition("polypgen_c4", "PolypGen with centre C4 held out (Table 4, Panel B)",
                  227, "surface"),
        Partition("polypgen_archive", "PolypGen 1537-frame archive (Table B.4)",
                  1537, "surface"),
        Partition("siim", "SIIM-ACR held-out test partition (Table 4, Panel C)",
                  2135, "mask"),
    )
}


# ---------------------------------------------------------------------------
# Per-image geometry
# ---------------------------------------------------------------------------
# Two boundary definitions were used when the released records were computed.
#
#   "mask"     The metric logged during training. HD95 pools the
#              distances from each mask's surface pixels to the other MASK; the
#              boundary band is dilate(3) - erode(3), a 7-pixel band across the
#              contour. Used for the PolypGen mixed partition and SIIM-ACR.
#   "surface"  HD95 pools surface-to-SURFACE distances; the boundary band is the
#              inner band mask - erode^w(mask) with w = round(0.02 * diagonal) = 7,
#              the dilation ratio of Cheng et al.'s Boundary IoU. Used for the
#              centre-held-out partition and the 1537-frame archive.
#
# Both definitions agree on everything a convention acts on: which images have
# an empty reference or prediction, and therefore every count, rate and
# denominator in the paper. They differ only in the distance and boundary
# values of images on which both masks are non-empty.

GEOMETRIES = ("mask", "surface")


def _cross():
    from scipy import ndimage
    return ndimage.generate_binary_structure(2, 1)


def _surface(mask: np.ndarray) -> np.ndarray:
    from scipy import ndimage
    return mask & ~ndimage.binary_erosion(mask, structure=_cross(), border_value=0)


def hd95_det(pred: np.ndarray, gt: np.ndarray, geometry: str = "mask") -> Optional[float]:
    """95th-percentile symmetric boundary distance, or None if a mask is empty.

    Returning None is the only place an empty mask enters: the value is left
    undefined here and each convention decides what to do with it.
    """
    from scipy import ndimage
    if not pred.any() or not gt.any():
        return None
    p_surf, t_surf = _surface(pred), _surface(gt)
    if not p_surf.any() or not t_surf.any():
        return 0.0
    if geometry == "mask":
        to_gt = ndimage.distance_transform_edt(~gt)
        to_pred = ndimage.distance_transform_edt(~pred)
    elif geometry == "surface":
        to_gt = ndimage.distance_transform_edt(~t_surf)
        to_pred = ndimage.distance_transform_edt(~p_surf)
    else:
        raise ValueError(f"unknown geometry {geometry!r}")
    distances = np.concatenate([to_gt[p_surf], to_pred[t_surf]])
    return float(np.percentile(distances, 95))


def boundary_band(mask: np.ndarray, geometry: str = "mask") -> np.ndarray:
    """Boundary band used by Boundary IoU (see the geometry note above)."""
    from scipy import ndimage
    if not mask.any():
        return np.zeros_like(mask, dtype=bool)
    if geometry == "mask":
        # cv2.dilate - cv2.erode with a 7 x 7 kernel; OpenCV's default border leaves
        # the image edge out of both, hence border_value=1 for the erosion
        kernel = np.ones((7, 7), bool)
        return (ndimage.binary_dilation(mask, kernel, border_value=0)
                & ~ndimage.binary_erosion(mask, kernel, border_value=1))
    if geometry == "surface":
        width = max(1, int(round(0.02 * float(np.hypot(*mask.shape)))))
        eroded = ndimage.binary_erosion(mask, structure=np.ones((3, 3), np.uint8),
                                        iterations=width, border_value=0)
        return mask & ~eroded
    raise ValueError(f"unknown geometry {geometry!r}")


def binarize(prob: np.ndarray, threshold: float, min_area: int = 0) -> np.ndarray:
    """Threshold a probability map and drop 8-connected components below min_area."""
    pred = prob >= threshold
    if min_area <= 1 or not pred.any():
        return pred
    from scipy import ndimage
    labels, _ = ndimage.label(pred, structure=np.ones((3, 3), np.uint8))
    keep = np.bincount(labels.ravel()) >= min_area
    keep[0] = False
    return keep[labels]


def make_record(prob: np.ndarray, gt: np.ndarray, threshold: float, min_area: int,
                geometry: str = "mask") -> dict:
    """Score one probability map against one binary reference (no convention applied)."""
    pred = binarize(prob, threshold, min_area)
    gt = gt.astype(bool)
    pb, gb = boundary_band(pred, geometry), boundary_band(gt, geometry)
    return {
        "threshold": float(threshold),
        "min_area": int(min_area),
        "gt_area": int(gt.sum()),
        "pred_area": int(pred.sum()),
        "inter": int((pred & gt).sum()),
        "hd95_det": hd95_det(pred, gt, geometry),
        "biou_inter": int((pb & gb).sum()),
        "biou_union": int((pb | gb).sum()),
    }


# ---------------------------------------------------------------------------
# Reading and writing
# ---------------------------------------------------------------------------

def records_path(partition: str, root: Path | str | None = None) -> Path:
    if partition not in PARTITIONS:
        raise KeyError(f"unknown partition {partition!r}; choose from {sorted(PARTITIONS)}")
    return Path(root or RECORDS_DIR) / f"{partition}.csv.gz"


def write(df: pd.DataFrame, path: Path | str) -> Path:
    """Write records in the released schema (full float precision, gzip)."""
    missing = [c for c in COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"records are missing columns {missing}")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out = df[COLUMNS].sort_values(["model", "seed", "threshold", "min_area", "idx"],
                                  kind="mergesort")
    out.to_csv(path, index=False, compression={"method": "gzip", "mtime": 0})
    return path


def load(partition: str, threshold: float | None = 0.5, min_area: int | None = 0,
         models: Sequence[str] | None = None, seeds: Iterable[int] | None = None,
         root: Path | str | None = None) -> pd.DataFrame:
    """Load the records of one partition, filtered to one binarisation cell.

    Pass threshold=None / min_area=None to keep every cell (robustness grid).
    Each (model, seed, cell) is checked to cover every frame of the partition
    exactly once, so a truncated or duplicated file fails loudly.
    """
    df = pd.read_csv(records_path(partition, root), float_precision="round_trip",
                     dtype={"model": str, "image": str, "partition": str})
    if threshold is not None:
        df = df[np.isclose(df["threshold"], threshold)]
    if min_area is not None:
        df = df[df["min_area"] == min_area]
    if models is not None:
        df = df[df["model"].isin(list(models))]
    if seeds is not None:
        df = df[df["seed"].isin(list(seeds))]
    if df.empty:
        raise SystemExit(f"no {partition} records for threshold={threshold}, "
                         f"min_area={min_area}, models={models}, seeds={seeds}")
    n = PARTITIONS[partition].n_frames
    for key, g in df.groupby(["model", "seed", "threshold", "min_area"]):
        if len(g) != n or g["idx"].nunique() != n:
            raise ValueError(f"{partition} {key}: {len(g)} rows, expected {n} distinct frames")
    return df.reset_index(drop=True)
