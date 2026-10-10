"""Train one model for one seed under one of the paper's protocols.

    python -m scripts.train --model unetpp --protocol polypgen --seed 42
    python -m scripts.train --model prism --variant ng_no_div --protocol polypgen --seed 7
    python -m scripts.train --model lbunet --protocol siim --seed 1337
    python -m scripts.train --model egeunet --protocol siim --seed 42 --epochs 30 --per-image-dice

Models: unetpp, mobileunet, tinyunet, prism, unext_s, lbunet, egeunet, malunet.
Protocols (emc/training/protocol.py): polypgen (mixed partition, 300 epochs),
polypgen_c4 (centre C4 held out), siim (SIIM-ACR, 111 epochs). The paper trains
seeds 42, 1337 and 2024; the PRISM ablation uses 7, 42, 1337, 2024, 3407, 12345,
271828 and 314159. --epochs shortens a run without changing the cosine schedule
(the stress test of Appendix D uses 30 of 111 epochs with a per-image Dice term).

The data root defaults to datasets/<polypgen_mixed | polypgen_c4 | siim>
(scripts/prepare_polypgen.py, scripts/prepare_siim.py). Writes <out>/best.pth (EMA
weights of the selected epoch) and <out>/run.json (configuration and validation
curve). The selected checkpoint is the input of scripts/predict.py.

Reproducibility: with --deterministic (default) cuDNN runs deterministically. The
original PolypGen runs drew their rotation angle before seeding (it was not
recorded), and the SIIM-ACR runs used cuDNN autotuning, so a rerun reproduces
them statistically rather than bit for bit.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from emc import models
from emc.records import REPO
from emc.training import engine
from emc.training.data import BalancedEpochSampler, FolderDataset
from emc.training.losses import build_objective
from emc.training.protocol import OBJECTIVE_FAMILY, PROTOCOLS

DATA_DIR = {"polypgen": "polypgen_mixed", "polypgen_c4": "polypgen_c4", "siim": "siim"}


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, choices=sorted(models.MODELS))
    ap.add_argument("--protocol", required=True, choices=sorted(PROTOCOLS))
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--variant", default=None, help="PRISM ablation arm (prism_dn, ng_no_div, ng_sub, "
                                                    "ng_no_hires, ng_no_edge)")
    ap.add_argument("--data", default=None, help="prepared dataset root")
    ap.add_argument("--epochs", type=int, default=None, help="stop early; the schedule keeps T_max")
    ap.add_argument("--per-image-dice", action="store_true", help="SIIM-ACR: per-image Dice term (Appendix D)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--no-deterministic", dest="deterministic", action="store_false")
    return ap.parse_args()


def main():
    a = parse_args()
    p = PROTOCOLS[a.protocol]
    epochs = a.epochs or p.epochs
    if epochs > p.t_max:
        raise SystemExit(f"--epochs must not exceed {p.t_max}")
    batch_dice = p.batch_dice and not a.per_image_dice
    tag = a.model + (f"_{a.variant}" if a.variant else "") + f"_seed{a.seed}"
    out = Path(a.out or REPO / "runs" / a.protocol / tag)
    out.mkdir(parents=True, exist_ok=True)
    root = Path(a.data or REPO / "datasets" / DATA_DIR[a.protocol])
    spec = models.MODELS[a.model]

    engine.set_seed(a.seed, a.deterministic)
    train_tf = p.train_transform()
    train_set = FolderDataset(root, p.splits[0], train_tf, points=spec.needs_points)
    val_set = FolderDataset(root, p.splits[1], p.test_transform())
    if p.balanced:
        sampler = BalancedEpochSampler(train_set.empty_masks(), a.seed)
        loader = DataLoader(train_set, batch_size=p.batch_size, sampler=sampler, num_workers=a.workers,
                            pin_memory=True, persistent_workers=a.workers > 0)
    else:
        sampler = None
        # LB-UNet's original loop kept the last partial batch (154 steps per epoch); the others drop it.
        loader = DataLoader(train_set, batch_size=p.batch_size, shuffle=True, num_workers=a.workers,
                            pin_memory=True, drop_last=a.model != "lbunet", persistent_workers=a.workers > 0)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=a.workers)

    model = models.build(a.model, a.variant).to(a.device)
    objective = build_objective(a.model, OBJECTIVE_FAMILY[a.protocol], batch_dice).to(a.device)
    optimiser, schedule = engine.optimiser_and_schedule(model, p)
    ema = engine.EMA(model, p.ema_decay)
    rotation = next(t for t in train_tf.transforms if t.__class__.__name__ == "Rotation")
    print(f"{tag}: {sum(x.numel() for x in model.parameters()):,} parameters, {len(train_set)} training "
          f"images, {len(loader)} steps per epoch, {epochs} epochs (T_max {p.t_max})")

    best, best_epoch, curve = -float("inf"), 0, []
    for epoch in range(1, epochs + 1):
        if sampler is not None:
            sampler.set_epoch(epoch)
        t0 = time.time()
        loss = engine.train_one_epoch(loader, model, objective, optimiser, ema, a.device)
        schedule.step()
        metrics = engine.evaluate(val_loader, ema.module, a.model, a.device, p.threshold)
        score = engine.selection_score(metrics, p.selection)
        curve.append({"epoch": epoch, "train_loss": loss, **metrics})
        if score > best:
            best, best_epoch = score, epoch
            torch.save({"model_state_dict": ema.module.state_dict(), "epoch": epoch}, out / "best.pth")
        print(f"epoch {epoch:3d}  loss {loss:.4f}  {p.selection} {score:.4f}  "
              f"dice_pos {metrics['dice_pos']:.4f}  q {metrics['q']:.3f}  r0 {metrics['r0']:.3f}  "
              f"({time.time() - t0:.0f} s){'  *' if best_epoch == epoch else ''}")

    run = {"model": spec.name, "key": a.model, "variant": a.variant, "seed": a.seed, "protocol": a.protocol,
           "data": str(root), "epochs": epochs, "t_max": p.t_max, "batch_dice": batch_dice,
           "drop_last": (not p.balanced) and a.model != "lbunet",
           "rotation_angle": rotation.angle, "deterministic": a.deterministic,
           "parameters": sum(x.numel() for x in model.parameters()),
           "selection": p.selection, "best_epoch": best_epoch, "best_score": best, "val_curve": curve}
    (out / "run.json").write_text(json.dumps(run, indent=1), encoding="utf-8")
    print(f"best epoch {best_epoch} ({p.selection} {best:.4f}) -> {out / 'best.pth'}")


if __name__ == "__main__":
    main()
