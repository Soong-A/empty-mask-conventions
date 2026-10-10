"""Build the SIIM-ACR partition from the Kaggle release (DICOM images + train-rle.csv).

    python -m scripts.prepare_siim --dicom-root PATH/siim-acr --rle PATH/train-rle.csv

Images are listed in splits/siim_split_manifest.csv (ImageId, split, is_empty,
area_px_at_1024). Within each split, images are sorted by ImageId and written as
00000.png, 00001.png, ... to datasets/siim/{train,val,test}/{images,masks}:

    image  DICOM -> VOI LUT -> MONOCHROME1 inverted -> min-max to 0..255 -> bilinear 256 x 256
    mask   union of the run-length annotations at 1024 x 1024 -> nearest-neighbour 256 x 256

Three checks stop the script on failure: the decoded foreground area of every
image equals the manifest, every split has the manifest's number of empty masks
after resizing (nearest-neighbour resizing erases no annotation; Appendix C.2),
and images and masks pair by name. The intensity statistics of the training
split are printed (129.1255 / 65.1556 in emc/training/transforms.py).
"""
from __future__ import annotations

import argparse
import os
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from emc.records import REPO

NATIVE = 1024


def rle_to_mask(rle: str, height: int = NATIVE, width: int = NATIVE) -> np.ndarray:
    """SIIM-ACR run-length code: column-major, start positions relative to the previous run."""
    tokens = np.asarray(rle.split(), dtype=np.int64)
    flat = np.zeros(height * width, dtype=np.uint8)
    pos = 0
    for start, length in zip(tokens[0::2], tokens[1::2]):
        pos += start
        flat[pos:pos + length] = 255
        pos += length
    return flat.reshape(width, height).T


def read_dicom(path) -> np.ndarray:
    import pydicom
    try:
        from pydicom.pixels import apply_voi_lut
    except ImportError:
        from pydicom.pixel_data_handlers.util import apply_voi_lut
    ds = pydicom.dcmread(path)
    arr = ds.pixel_array
    try:
        arr = apply_voi_lut(arr, ds)
    except Exception:
        pass
    if getattr(ds, "PhotometricInterpretation", "") == "MONOCHROME1":
        arr = arr.max() - arr
    arr = arr.astype(np.float32)
    lo, hi = float(arr.min()), float(arr.max())
    arr = np.zeros_like(arr) if hi <= lo else (arr - lo) / (hi - lo)
    return (arr * 255.0).round().astype(np.uint8)


def process(job):
    image_id, dicom, rles, area, split, stem, out = job
    img = read_dicom(dicom)
    if img.shape != (NATIVE, NATIVE):
        img = np.array(Image.fromarray(img).resize((NATIVE, NATIVE), Image.BILINEAR))
    mask = np.zeros((NATIVE, NATIVE), np.uint8)
    for rle in rles:
        if rle.strip() != "-1":
            mask = np.maximum(mask, rle_to_mask(rle))
    native = int((mask > 0).sum())
    if native != area:
        return image_id, f"decoded area {native} != manifest {area}"
    img = np.array(Image.fromarray(img).resize((256, 256), Image.BILINEAR))
    mask = ((np.array(Image.fromarray(mask).resize((256, 256), Image.NEAREST)) > 127) * 255).astype(np.uint8)
    Image.fromarray(img).save(Path(out, split, "images", stem + ".png"))
    Image.fromarray(mask).save(Path(out, split, "masks", stem + ".png"))
    return image_id, (split, int(mask.max() == 0), img.astype(np.float64).sum(),
                      (img.astype(np.float64) ** 2).sum(), img.size)


def assign_stems(manifest: pd.DataFrame) -> pd.DataFrame:
    m = manifest.sort_values(["split", "ImageId"]).copy()
    m["stem"] = m.groupby("split").cumcount().map("{:05d}".format)
    return m


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dicom-root", required=True)
    ap.add_argument("--rle", required=True)
    ap.add_argument("--out", default=str(REPO / "datasets" / "siim"))
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()

    manifest = assign_stems(pd.read_csv(REPO / "splits" / "siim_split_manifest.csv", dtype={"ImageId": str}))
    rles = defaultdict(list)
    for row in pd.read_csv(a.rle, dtype=str).itertuples(index=False):
        rles[row[0].strip()].append(row[1].strip())
    dicoms = {}
    for dirpath, _, files in os.walk(a.dicom_root):
        for f in files:
            if f.lower().endswith(".dcm"):
                dicoms.setdefault(Path(f).stem, os.path.join(dirpath, f))
    missing = [i for i in manifest.ImageId if i not in dicoms]
    if missing:
        raise SystemExit(f"{len(missing)} manifest images have no DICOM, e.g. {missing[0]}")
    for s in ("train", "val", "test"):
        for k in ("images", "masks"):
            Path(a.out, s, k).mkdir(parents=True, exist_ok=True)

    jobs = [(r.ImageId, dicoms[r.ImageId], rles[r.ImageId], 0 if r.is_empty else int(r.area_px_at_1024),
             r.split, r.stem, a.out) for r in manifest.itertuples()]
    results, errors = {}, []
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for image_id, res in ex.map(process, jobs, chunksize=16):
            (errors.append((image_id, res)) if isinstance(res, str) else results.__setitem__(image_id, res))
    if errors:
        raise SystemExit(f"{len(errors)} images failed, e.g. {errors[0]}")

    empty = Counter(v[0] for v in results.values() if v[1])
    want = manifest.groupby("split").is_empty.sum()
    for s in ("train", "val", "test"):
        print(f"{s:<6} {int((manifest.split == s).sum()):5d} images, {empty[s]} empty (manifest {int(want[s])})")
        if empty[s] != want[s]:
            raise SystemExit("resizing changed the number of empty masks")
        if sorted(os.listdir(Path(a.out, s, "images"))) != sorted(os.listdir(Path(a.out, s, "masks"))):
            raise SystemExit(f"{s}: image and mask names differ")
    tr = [v for v in results.values() if v[0] == "train"]
    n = sum(v[4] for v in tr)
    mean = sum(v[2] for v in tr) / n
    print(f"training intensity mean {mean:.4f}, std {(sum(v[3] for v in tr) / n - mean ** 2) ** 0.5:.4f}")


if __name__ == "__main__":
    main()
