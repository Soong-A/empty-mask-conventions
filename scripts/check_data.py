"""Empty annotations in the released data (Appendix C.2, Table C.1).

    python -m scripts.check_data polypgen --raw PATH/PolypGen2021_MultiCenterData_v3
    python -m scripts.check_data siim --rle PATH/train-rle.csv

polypgen
  1. empty masks at native resolution (no pixel above the threshold), for every
     threshold 1..254: the count stays at 125;
  2. empty masks by acquisition centre (Table C.1), checked against splits/polypgen_frames.csv;
  3. lesion-bearing masks that vanish when resized to 96-512 px and thresholded at 127;
  4. images whose mask cannot be paired by name (the stray-bracket frame) and the
     area of their released mask.
siim
  every annotation of train-rle.csv rendered at 1024 x 1024, resized to 256-512 px
  with nearest, bilinear and area filters and thresholded at 127; writes the number
  of erased annotations per setting and compares it with results/data_checks/.
"""
from __future__ import annotations

import argparse
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from emc.report import OUTPUTS, RESULTS, SPLITS

FILTERS = {"nearest": Image.NEAREST, "bilinear": Image.BILINEAR, "area": Image.BOX}


def cmd_polypgen(raw: Path):
    frames = pd.read_csv(SPLITS / "polypgen_frames.csv", dtype=str, keep_default_na=False)
    masks = sorted(raw.glob("data_C*/masks_C*/*"))
    peak = {}
    for p in masks:
        peak[p] = int(np.asarray(Image.open(p).convert("L")).max())
    counts = {t: sum(v <= t for v in peak.values()) for t in range(1, 255)}
    stable = [t for t, n in counts.items() if n == 125]
    print(f"1. {len(masks)} released masks; empty (max <= t) = 125 for thresholds {stable[0]}..{stable[-1]} "
          f"of 1..254 (range of counts {min(counts.values())}-{max(counts.values())})")

    centre = {p: re.search(r"data_(C\d)", str(p)).group(1) for p in masks}
    by_centre = pd.DataFrame({"centre": [centre[p] for p in masks], "empty": [peak[p] <= 127 for p in masks]})
    table = by_centre.groupby("centre").empty.agg(["size", "sum"])
    table["pct"] = (100 * table["sum"] / table["size"]).round(2)
    print("2. empty masks by centre:\n" + table.T.to_string())
    manifest = frames.groupby("centre").source_mask_empty.apply(lambda s: (s == "1").sum())
    print("   matches splits/polypgen_frames.csv:", bool((table["sum"] == manifest).all()))

    print("3. lesion-bearing masks that vanish after resizing (threshold 127):")
    for name, filt in FILTERS.items():
        gone = defaultdict(list)
        for p in masks:
            if peak[p] <= 127:
                continue
            m = Image.open(p).convert("L")
            for size in (512, 384, 256, 224, 128, 96):
                if np.asarray(m.resize((size, size), filt)).max() <= 127:
                    gone[size].append(p.name)
        print(f"   {name:<8} " + "  ".join(f"{s}px: {len(gone[s])}" for s in (512, 384, 256, 224, 128, 96)))
        everywhere = set.intersection(*(set(gone[s]) for s in (512, 384, 256, 224, 128, 96))) if gone else set()
        for n in sorted(everywhere):
            area = int((np.asarray(Image.open(next(p for p in masks if p.name == n)).convert("L")) > 127).sum())
            print(f"            vanishes at every resolution: {n} ({area} px at native resolution)")

    images = sorted(raw.glob("data_C*/images_C*/*"))
    names = {p.name for p in masks}
    unpaired = [p for p in images if f"{p.stem}_mask{p.suffix}" not in names]
    print(f"4. {len(images)} images; {len(unpaired)} without a mask of the same name:")
    for p in unpaired:
        clean = p.stem.rstrip("]")
        m = next((q for q in masks if q.stem == f"{clean}_mask"), None)
        area = int((np.asarray(Image.open(m).convert("L")) > 127).sum()) if m else None
        frame = frames[frames.source_image.str.endswith(p.name)]
        print(f"   {p.name}: released mask {m.name if m else None} has {area} foreground px; "
              f"frame {frame.frame.iloc[0]}, mixed split {frame.mixed.iloc[0]}, centre-held-out split {frame.c4.iloc[0]}")


def cmd_siim(rle: Path):
    from scripts.prepare_siim import rle_to_mask
    runs = defaultdict(list)
    for row in pd.read_csv(rle, dtype=str).itertuples(index=False):
        runs[row[0].strip()].append(row[1].strip())
    rows, erased = [], []
    positives = {}
    for image_id, rr in runs.items():
        rr = [r for r in rr if r != "-1"]
        if rr:
            m = np.zeros((1024, 1024), np.uint8)
            for r in rr:
                m = np.maximum(m, rle_to_mask(r))
            positives[image_id] = m
    n_all = len(runs)
    for size in (256, 320, 352, 384, 448, 512):
        for name, filt in FILTERS.items():
            gone = [i for i, m in positives.items()
                    if not (np.asarray(Image.fromarray(m).resize((size, size), filt)) > 127).any()]
            erased += [{"resolution": size, "filter": name, "ImageId": i,
                        "area_px_at_1024": int((positives[i] > 0).sum())} for i in gone]
            rows.append({"resolution": size, "filter": name, "threshold": 127, "positives": len(positives),
                         "erased": len(gone),
                         "lesion_free_pct_after": round(100 * (n_all - len(positives) + len(gone)) / n_all, 3)})
    out = pd.DataFrame(rows)
    path = OUTPUTS / "data_checks" / "siim_resize_erasure.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)
    pd.DataFrame(erased).to_csv(path.with_name("siim_resize_erased_images.csv"), index=False)
    print(out.to_string(index=False))
    ref = pd.read_csv(RESULTS / "data_checks" / "siim_resize_erasure.csv")
    same = (ref.set_index(["resolution", "filter"]).erased == out.set_index(["resolution", "filter"]).erased).all()
    print(f"-> {path}; identical to the released counts: {bool(same)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("polypgen")
    p.add_argument("--raw", required=True)
    s = sub.add_parser("siim")
    s.add_argument("--rle", required=True)
    a = ap.parse_args()
    cmd_polypgen(Path(a.raw)) if a.cmd == "polypgen" else cmd_siim(Path(a.rle))


if __name__ == "__main__":
    main()
