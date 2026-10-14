"""Score your own masks under every empty-mask convention.

    emc-score --pred PRED_DIR --ref REF_DIR          # masks or probability maps
    emc-score --records records.csv                  # per-image records (released schema)

Predictions and references are paired by file name without the extension
(.png, .jpg, .bmp, .tif or .npy). An 8-bit image is divided by 255, so a
probability map saved as an image and a 0/255 mask both work; a 0/1 mask and a
.npy array are used as they are. A prediction is foreground where it is at
least ``--threshold`` (default 0.5), a reference where it is at least 0.5.

The output reports the counts and rates that decide every convention (q, r0,
image-level sensitivity 1 - q and specificity 1 - r0), Dice and Boundary IoU
over non-empty references and with two empty masks scoring c = 0 or 1, and
HD95 under deletion (X, MONAI's default), X', d = 0 (Z) and each one-sided-empty
penalty d (P; by default the image diagonal). See emc.conventions for the
definitions and Section 2 of the paper for the decomposition behind them.

From Python::

    from emc.score import score_pairs, summarise_records
    records = score_pairs({"case1": (pred, ref), ...})
    print(summarise_records(records, penalties={"Pdiag": 362.04}))
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import pandas as pd

from . import conventions as C
from . import records as R

SUFFIXES = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".npy")
RECORD_COLUMNS = ["gt_area", "pred_area", "inter", "hd95_det", "biou_inter", "biou_union"]


def read_array(path: Path | str) -> np.ndarray:
    """A 2-D float array in [0, 1] from an image or a .npy file."""
    path = Path(path)
    if path.suffix.lower() == ".npy":
        a = np.load(path)
    else:
        from PIL import Image
        with Image.open(path) as im:
            a = np.asarray(im.convert("L"))
    a = np.squeeze(a)
    if a.ndim != 2:
        raise ValueError(f"{path}: expected a 2-D mask, got shape {a.shape}")
    if a.dtype == np.uint8 and a.max() > 1:
        return a / 255.0
    return a.astype(float)


def pair_files(pred_dir: Path | str, ref_dir: Path | str) -> dict:
    """{stem: (prediction path, reference path)}; every file must have a partner."""
    def index(d):
        files = {}
        for p in sorted(Path(d).iterdir()):
            if p.suffix.lower() in SUFFIXES:
                if p.stem in files:
                    raise ValueError(f"two files named {p.stem!r} in {d}")
                files[p.stem] = p
        return files
    pred, ref = index(pred_dir), index(ref_dir)
    if not ref:
        raise ValueError(f"no masks in {ref_dir}")
    unmatched = sorted(set(pred) ^ set(ref))
    if unmatched:
        raise ValueError(f"{len(unmatched)} file(s) without a partner, e.g. {unmatched[:5]}")
    return {k: (pred[k], ref[k]) for k in ref}


def score_pairs(pairs: Mapping[str, tuple], threshold: float = 0.5, min_area: int = 0,
                geometry: str = "mask") -> pd.DataFrame:
    """Per-image records for {name: (prediction, reference)}, arrays or paths.

    No convention is applied here; ``summarise_records`` applies all of them.
    """
    rows = []
    for name, (pred, ref) in pairs.items():
        p = read_array(pred) if isinstance(pred, (str, Path)) else np.asarray(pred, float)
        g = read_array(ref) if isinstance(ref, (str, Path)) else np.asarray(ref, float)
        if p.shape != g.shape:
            raise ValueError(f"{name}: prediction {p.shape} and reference {g.shape} differ in shape")
        rec = R.make_record(p, g >= 0.5, threshold, min_area, geometry)
        rows.append({"image": str(name), "height": g.shape[0], "width": g.shape[1], **rec})
    return pd.DataFrame(rows)


def default_penalties(records: pd.DataFrame) -> dict:
    """{"Pdiag": image diagonal}; the images must share one size."""
    sizes = records[["height", "width"]].drop_duplicates()
    if len(sizes) != 1:
        raise ValueError("images differ in size; give the penalties explicitly (--penalty)")
    h, w = sizes.iloc[0]
    return {"Pdiag": float(np.hypot(h, w))}


def summarise_records(records: pd.DataFrame, penalties: Mapping[str, float] | None = None) -> dict:
    """Every count, rate and convention for one set of per-image records.

    ``hd95_pos`` (HD95 over non-empty references) charges a complete miss the
    largest of the penalties.
    """
    missing = [c for c in RECORD_COLUMNS if c not in records.columns]
    if missing:
        raise ValueError(f"records are missing columns {missing}")
    keys = [c for c in ("model", "seed", "threshold", "min_area") if c in records.columns]
    if keys and len(records[keys].drop_duplicates()) > 1:
        raise ValueError(f"records hold more than one run ({', '.join(keys)}); "
                         "score one run at a time")
    if penalties is None:
        penalties = default_penalties(records)
    penalties = dict(penalties)
    d_pos = max(penalties.values())
    m = C.run_metrics(records, d_pos=d_pos, penalties=penalties)
    m["sensitivity"] = 1 - m["q"]
    m["specificity"] = 1 - m["r0"]
    m["penalties"] = penalties
    return m


def format_summary(m: dict) -> str:
    def f(x, nd=4):
        return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{nd}f}"
    lines = [
        f"images                N = {m['N']}  (non-empty references N+ = {m['N_pos']}, "
        f"empty references N0 = {m['N_neg']})",
        f"complete misses       q  = {f(m['q'])}   ({m['n_miss']} of {m['N_pos']});  sensitivity 1 - q  = {f(m['sensitivity'])}",
        f"foreground on empty   r0 = {f(m['r0'])}   ({m['n_fa']} of {m['N_neg']});  specificity 1 - r0 = {f(m['specificity'])}",
        f"both masks non-empty  n_det = {m['n_det']}  (images kept by deletion, family X)",
        "",
        f"{'':22s}{'pos':>10s}{'c0=0':>10s}{'c0=1':>10s}",
        f"{'Dice':22s}{f(m['dice_pos']):>10s}{f(m['dice_c0']):>10s}{f(m['dice_c1']):>10s}",
        f"{'Boundary IoU':22s}{f(m['biou_pos']):>10s}{f(m['biou_c0']):>10s}{f(m['biou_c1']):>10s}",
        "",
        "HD95 (px)",
        f"  X  deletion          {f(m['hd95_X'], 2)}",
        f"  X' one-sided deleted {f(m['hd95_Xprime'], 2)}",
        f"  Z  d = 0             {f(m['hd95_Z'], 2)}",
    ]
    for name, d in m["penalties"].items():
        lines.append(f"  P  d = {d:<13.2f} {f(m['hd95_' + name], 2)}   ({name})")
    return "\n".join(lines)


def _jsonable(m: dict) -> dict:
    return {k: (None if isinstance(v, float) and np.isnan(v) else
                v.item() if isinstance(v, np.generic) else v) for k, v in m.items()}


def main(argv: Iterable[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="emc-score", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--pred", type=Path, help="folder of predicted masks or probability maps")
    src.add_argument("--records", type=Path, help="CSV of per-image records (columns "
                     + ", ".join(RECORD_COLUMNS) + ")")
    ap.add_argument("--ref", type=Path, help="folder of reference masks (with --pred)")
    ap.add_argument("--threshold", type=float, default=0.5, help="prediction threshold (default 0.5)")
    ap.add_argument("--min-area", type=int, default=0,
                    help="drop predicted 8-connected components smaller than this (default 0)")
    ap.add_argument("--geometry", choices=R.GEOMETRIES, default="mask",
                    help="HD95 and boundary-band definition (see emc.records; default mask)")
    ap.add_argument("--penalty", type=float, action="append", metavar="D",
                    help="one-sided-empty HD95 penalty in pixels; repeat for several "
                         "(default: the image diagonal)")
    ap.add_argument("--json", type=Path, help="also write the summary as JSON")
    ap.add_argument("--records-out", type=Path, help="also write the per-image records as CSV")
    args = ap.parse_args(None if argv is None else list(argv))

    try:
        if args.pred is not None:
            if args.ref is None:
                ap.error("--pred needs --ref")
            records = score_pairs(pair_files(args.pred, args.ref), args.threshold,
                                  args.min_area, args.geometry)
        else:
            records = pd.read_csv(args.records, float_precision="round_trip")
        penalties = ({f"P{d:g}": d for d in args.penalty} if args.penalty else None)
        if penalties is None and "height" not in records.columns:
            ap.error("records without image sizes need --penalty")
        summary = summarise_records(records, penalties)
    except ValueError as e:
        print(f"emc-score: error: {e}", file=sys.stderr)
        return 2

    print(format_summary(summary))
    if args.records_out:
        args.records_out.parent.mkdir(parents=True, exist_ok=True)
        records.to_csv(args.records_out, index=False)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(_jsonable(summary), indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
