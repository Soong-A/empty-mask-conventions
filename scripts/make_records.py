"""Turn stored probability maps into convention-free per-image records.

    python -m scripts.make_records --partition polypgen_mixed --data datasets/polypgen_mixed \
        --split val --pred-root predictions/polypgen_mixed --thresholds 0.3 0.5 0.7 --min-areas 0 10 50 100

--pred-root holds one folder per run, named <Model>_seed<seed> (scripts/predict.py).
Each map is binarised at every threshold, connected components smaller than the
minimum area are removed, and the record of emc.records is written for every
(image, threshold, minimum area). The boundary geometry defaults to the one the
released records of the partition use (emc.records.PARTITIONS). The output
(default records/<partition>.csv.gz) has the released schema, so every table
script runs on regenerated records unchanged.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

from emc import records as R
from emc.training.data import FolderDataset
from emc.training.protocol import PROTOCOLS

PROTOCOL = {"polypgen_mixed": "polypgen", "polypgen_ablation": "polypgen", "polypgen_archive": "polypgen", "polypgen_c4": "polypgen_c4",
            "siim": "siim"}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--partition", required=True, choices=sorted(R.PARTITIONS))
    ap.add_argument("--data", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--pred-root", required=True)
    ap.add_argument("--thresholds", type=float, nargs="+", default=[0.5])
    ap.add_argument("--min-areas", type=int, nargs="+", default=[0])
    ap.add_argument("--geometry", choices=R.GEOMETRIES, default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    geometry = a.geometry or R.PARTITIONS[a.partition].geometry
    refs = FolderDataset(a.data, a.split, PROTOCOLS[PROTOCOL[a.partition]].test_transform())
    gts = [(refs[i][1].squeeze().numpy() >= 0.5) for i in range(len(refs))]
    rows = []
    for run in sorted(p for p in Path(a.pred_root).iterdir() if p.is_dir()):
        m = re.fullmatch(r"(.+)_seed(\d+)", run.name)
        if not m:
            continue
        model, seed = m.group(1), int(m.group(2))
        for idx, (name, gt) in enumerate(zip(refs.names, gts)):
            prob = np.load(run / f"{Path(name).stem}.npy")
            for t in a.thresholds:
                for area in a.min_areas:
                    rec = R.make_record(prob, gt, t, area, geometry)
                    rows.append({"partition": a.partition, "model": model, "seed": seed, "idx": idx,
                                 "image": name, **rec})
        print(f"{run.name}: {len(refs)} images")
    out = R.write(pd.DataFrame(rows), a.out or R.records_path(a.partition))
    print(f"{len(rows)} records -> {out}")


if __name__ == "__main__":
    main()
