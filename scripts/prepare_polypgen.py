"""Build the three PolypGen evaluation partitions from the public single-frame release.

    python -m scripts.prepare_polypgen --raw PATH/PolypGen2021_MultiCenterData_v3

Every frame of splits/polypgen_frames.csv is written once per partition as a
256 x 256 PNG (image: bilinear resize; mask: nearest-neighbour resize, then
foreground = grey level > 127):

    datasets/polypgen_mixed/{train,val}/{images,masks}       mixed partition (Panel A)
    datasets/polypgen_c4/{train,val,test}/{images,masks}     centre C4 held out (Panel B)
    datasets/polypgen_archive/all/{images,masks}             all 1537 frames (Table B.4)

The manifest reproduces the evaluation pipeline of the paper, including its two
pipeline-created empty references (Appendix C.2): frame 00265.png, whose 2-pixel
lesion disappears when resized, and frame 00950.png, whose image name ends in a
stray ']' so that its mask was not paired (written as an empty mask). Pass
--pair-by-name to pair that frame with its released mask instead. The script
checks that the empty masks it writes are those flagged in the manifest.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from emc.records import REPO

PARTITIONS = {"polypgen_mixed": "mixed", "polypgen_c4": "c4", "polypgen_archive": None}


def prepare_frame(raw: Path, row, pair_by_name: bool):
    image = Image.open(raw / row.source_image).convert("RGB").resize((256, 256), Image.BILINEAR)
    source_mask = row.source_mask
    if not source_mask and pair_by_name:
        stem = Path(row.source_image).stem.rstrip("]")
        source_mask = str(Path(row.source_image).parent).replace("images_", "masks_") + f"/{stem}_mask.jpg"
    if source_mask:
        mask = np.asarray(Image.open(raw / source_mask).convert("L").resize((256, 256), Image.NEAREST))
        mask = ((mask > 127) * 255).astype(np.uint8)
    else:
        mask = np.zeros((256, 256), np.uint8)
    return image, Image.fromarray(mask)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", required=True, help="folder PolypGen2021_MultiCenterData_v3 (data_C1 ... data_C6)")
    ap.add_argument("--out", default=str(REPO / "datasets"))
    ap.add_argument("--pair-by-name", action="store_true")
    a = ap.parse_args()
    raw, out = Path(a.raw), Path(a.out)
    frames = pd.read_csv(REPO / "splits" / "polypgen_frames.csv", dtype=str, keep_default_na=False)
    written_empty = []
    for i, row in enumerate(frames.itertuples(), 1):
        image, mask = prepare_frame(raw, row, a.pair_by_name)
        if mask.getextrema()[1] == 0:
            written_empty.append(row.frame)
        for partition, column in PARTITIONS.items():
            split = getattr(row, column) if column else "all"
            for kind, im in (("images", image), ("masks", mask)):
                d = out / partition / split / kind
                d.mkdir(parents=True, exist_ok=True)
                im.save(d / row.frame)
        if i % 200 == 0:
            print(f"{i}/{len(frames)} frames")
    expected = set(frames.frame[frames.mask_empty == "1"])
    if a.pair_by_name:
        expected.discard("00950.png")
    if set(written_empty) != expected:
        raise SystemExit(f"empty masks differ from the manifest: {sorted(set(written_empty) ^ expected)}")
    print(f"{len(frames)} frames, {len(written_empty)} empty references -> {out}")


if __name__ == "__main__":
    main()
