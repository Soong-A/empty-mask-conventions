"""Write the probability maps of one trained model on one evaluation split.

    python -m scripts.predict --model unetpp --checkpoint runs/polypgen/unetpp_seed42/best.pth \
        --protocol polypgen --data datasets/polypgen_mixed --split val --seed 42

Writes <out>/<image name>.npy (float32, 256 x 256, values in [0, 1]) and
<out>/manifest.csv; <out> defaults to predictions/<partition>/<Model>_seed<seed>.
--protocol sets the preprocessing (intensity statistics and mask resize) the model
was trained with: polypgen for the mixed partition and the 1537-frame archive,
polypgen_c4 for the centre-held-out partition, siim for SIIM-ACR. The maps are the
input of scripts/make_records.py and of the library probe (scripts/library_probe.py).
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from emc import models
from emc.records import REPO
from emc.training.data import FolderDataset
from emc.training.protocol import PROTOCOLS

PARTITION = {("polypgen", "val"): "polypgen_mixed", ("polypgen", "all"): "polypgen_archive",
             ("polypgen_c4", "test"): "polypgen_c4", ("siim", "test"): "siim"}


def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="registry key or display name")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--variant", default=None, help="PRISM ablation arm")
    ap.add_argument("--protocol", required=True, choices=sorted(PROTOCOLS))
    ap.add_argument("--data", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()

    key = models.key_of(a.model)
    name = models.MODELS[key].name + (f"-{a.variant}" if a.variant else "")
    partition = PARTITION.get((a.protocol, a.split), f"{a.protocol}_{a.split}")
    out = Path(a.out or REPO / "predictions" / partition / f"{name}_seed{a.seed}")
    out.mkdir(parents=True, exist_ok=True)

    model = models.build(key, a.variant)
    models.load_weights(model, a.checkpoint)
    model.to(a.device).eval()
    data = FolderDataset(a.data, a.split, PROTOCOLS[a.protocol].test_transform())
    loader = DataLoader(data, batch_size=1, shuffle=False)
    rows = []
    with torch.no_grad():
        for name_i, (img, _) in zip(data.names, loader):
            prob = models.probability(key, model(img.to(a.device).float()))
            arr = prob.squeeze().float().cpu().numpy().astype(np.float32)
            np.save(out / f"{Path(name_i).stem}.npy", arr)
            rows.append({"image": name_i, "pred": f"{Path(name_i).stem}.npy",
                         "min": float(arr.min()), "max": float(arr.max())})
    pd.DataFrame(rows).to_csv(out / "manifest.csv", index=False)
    (out / "checkpoint.txt").write_text(f"{Path(a.checkpoint).resolve()}\nsha256 {sha256(a.checkpoint)}\n",
                                       encoding="utf-8")
    print(f"{len(rows)} probability maps -> {out}")


if __name__ == "__main__":
    main()
