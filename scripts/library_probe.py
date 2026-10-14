"""Library probe: what metric libraries return for empty masks (Table 3, Appendix A.1).

    python -m scripts.library_probe probe                     # every installed library
    python -m scripts.library_probe check [--dir DIR]         # Table 3 from the probe files
    python -m scripts.library_probe monai-predictions --pred-root predictions/polypgen_mixed \
        --data datasets/polypgen_mixed                         # MONAI on the stored predictions

`probe` calls the public API of each importable library (MONAI, MedPy,
scikit-image, seg-metrics) on four 32 x 32 configurations -- two empty masks, an
empty reference with a non-empty prediction, a non-empty reference with an empty
prediction, and two non-empty squares 3 px apart (control) -- and records the
returned value, the exception, or NaN, with any warning text, in
outputs/library_probe/<library>-<version>.json. MedPy 0.4.0 and 0.5.2 cannot be
installed together; probe each in its own environment:

    pip install medpy==0.4.0   # in a second environment, then run `probe` there

`check` reads the probe files (default: the released ones in results/library_probe)
and verifies every executed cell of Table 3; surface-distance 0.1 is read from its
source and is not executed. `monai-predictions` runs MONAI's HausdorffDistanceMetric
(percentile 95) and DiceMetric with their default reductions on the stored
probability maps (Section 3.3 and Appendix B.1).
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import sys
import warnings
from pathlib import Path

import numpy as np

from emc.report import OUTPUTS, RESULTS

CASES = ("both_empty", "ref_empty", "pred_empty", "control")


def make_cases():
    empty = np.zeros((32, 32), np.uint8)
    ref = empty.copy(); ref[9:21, 8:20] = 1
    shifted = empty.copy(); shifted[12:24, 8:20] = 1
    return {"both_empty": (empty, empty), "ref_empty": (ref, empty),        # (prediction, reference)
            "pred_empty": (empty, ref), "control": (shifted, ref)}


def execute(fn):
    """Return value (NaN / +Inf spelled out) or the exception type, plus warnings."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            v = fn()
            if isinstance(v, dict):
                out = {"value": {k: _jsonable(x) for k, x in v.items()}}
            else:
                out = {"value": _jsonable(v)}
        except Exception as exc:  # the exception is the outcome
            out = {"raised": type(exc).__name__, "message": str(exc)[:200]}
    out["warnings"] = sorted({str(w.message)[:200] for w in caught})
    return out


def _jsonable(v):
    v = float(v.item() if hasattr(v, "item") else v)
    return "NaN" if math.isnan(v) else ("+Inf" if v == math.inf else v)


def probe_monai(cases):
    import torch
    from monai.metrics import DiceMetric, HausdorffDistanceMetric, SurfaceDistanceMetric

    def t(a):
        return torch.from_numpy(a.astype(np.float32))[None, None]

    def run(metric):
        def call(pred, ref):
            metric.reset()
            metric(t(pred), t(ref))
            agg, n = metric.aggregate()
            return {"per_case": metric.get_buffer().flatten()[0], "aggregate": agg, "not_nans": n}
        return call

    metrics = {
        "HD95": lambda: HausdorffDistanceMetric(percentile=95, reduction="mean", get_not_nans=True),
        "HD": lambda: HausdorffDistanceMetric(reduction="mean", get_not_nans=True),
        "ASD": lambda: SurfaceDistanceMetric(reduction="mean", get_not_nans=True),
        "Dice (default)": lambda: DiceMetric(reduction="mean", get_not_nans=True),
        "Dice (ignore_empty=False)": lambda: DiceMetric(reduction="mean", get_not_nans=True, ignore_empty=False),
    }
    return {m: {c: execute(lambda f=run(make()), p=p, r=r: f(p, r)) for c, (p, r) in cases.items()}
            for m, make in metrics.items()}


def probe_medpy(cases):
    from medpy.metric import binary
    return {m: {c: execute(lambda p=p, r=r: getattr(binary, m)(p, r)) for c, (p, r) in cases.items()}
            for m in ("dc", "jc", "hd", "hd95", "asd")}


def probe_skimage(cases):
    from skimage.metrics import hausdorff_distance
    return {f"hausdorff_distance ({k})": {c: execute(lambda p=p, r=r, k=k: hausdorff_distance(p, r, method=k))
                                          for c, (p, r) in cases.items()} for k in ("standard", "modified")}


def probe_segmetrics(cases):
    import seg_metrics.seg_metrics as sg

    def call(p, r, name):
        out = sg.write_metrics(labels=[1], gdth_img=r, pred_img=p, csv_file=None, metrics=[name],
                               verbose=False, spacing=[1.0, 1.0], fully_connected=True)
        return out[0][name][0]
    return {m: {c: execute(lambda p=p, r=r, m=m: call(p, r, m)) for c, (p, r) in cases.items()}
            for m in ("dice", "jaccard", "hd", "hd95", "msd")}


LIBRARIES = {"monai": ("monai", probe_monai), "medpy": ("MedPy", probe_medpy),
             "scikit-image": ("scikit-image", probe_skimage), "seg-metrics": ("seg-metrics", probe_segmetrics)}


def cmd_probe(out: Path):
    out.mkdir(parents=True, exist_ok=True)
    cases = make_cases()
    for lib, (dist, fn) in LIBRARIES.items():
        try:
            version = importlib.metadata.version(dist)
        except importlib.metadata.PackageNotFoundError:
            print(f"{lib}: not installed, skipped")
            continue
        record = {"library": lib, "version": version, "python": sys.version.split()[0],
                  "numpy": np.__version__, "results": fn(cases)}
        path = out / f"{lib}-{version}.json"
        path.write_text(json.dumps(record, indent=1), encoding="utf-8")
        print(f"{lib} {version} -> {path}")


def outcome(cell) -> str:
    """Short label of one probe outcome: a number, NaN, +Inf or 'refuses (<Exception>)'."""
    if "raised" in cell:
        return f"refuses ({cell['raised']})"
    v = cell["value"]
    if isinstance(v, dict):
        return f"per case {_short(v['per_case'])}, mean {_short(v['aggregate'])} over {_short(v['not_nans'])}"
    return _short(v)


def _short(v):
    return v if isinstance(v, str) else f"{v:g}"


# Table 3: (file prefix, metric, case) -> expected outcome label
EXPECTED = {
    ("monai-1.5.2", "HD95", "both_empty"): "per case NaN, mean 0 over 0",
    ("monai-1.5.2", "HD95", "ref_empty"): "per case NaN, mean 0 over 0",
    ("monai-1.5.2", "HD95", "pred_empty"): "per case NaN, mean 0 over 0",
    ("monai-1.5.2", "HD", "pred_empty"): "per case +Inf, mean +Inf over 1",
    ("monai-1.5.2", "ASD", "pred_empty"): "per case +Inf, mean +Inf over 1",
    ("monai-1.5.2", "Dice (default)", "both_empty"): "per case NaN, mean 0 over 0",
    ("monai-1.5.2", "Dice (default)", "ref_empty"): "per case NaN, mean 0 over 0",
    ("monai-1.5.2", "Dice (default)", "pred_empty"): "per case 0, mean 0 over 1",
    ("monai-1.5.2", "Dice (ignore_empty=False)", "both_empty"): "per case 1, mean 1 over 1",
    ("monai-1.5.2", "Dice (ignore_empty=False)", "ref_empty"): "per case 0, mean 0 over 1",
    ("medpy-0.4.0", "dc", "both_empty"): "0",
    ("medpy-0.4.0", "jc", "both_empty"): "refuses (ZeroDivisionError)",
    ("medpy-0.4.0", "dc", "pred_empty"): "0",
    ("medpy-0.4.0", "hd95", "both_empty"): "refuses (RuntimeError)",
    ("medpy-0.4.0", "hd95", "pred_empty"): "refuses (RuntimeError)",
    ("medpy-0.5.2", "dc", "both_empty"): "1",
    ("medpy-0.5.2", "jc", "both_empty"): "1",
    ("medpy-0.5.2", "dc", "pred_empty"): "0",
    ("medpy-0.5.2", "hd95", "both_empty"): "refuses (RuntimeError)",
    ("medpy-0.5.2", "hd95", "ref_empty"): "refuses (RuntimeError)",
    ("scikit-image-0.26.0", "hausdorff_distance (standard)", "both_empty"): "0",
    ("scikit-image-0.26.0", "hausdorff_distance (standard)", "pred_empty"): "+Inf",
    ("scikit-image-0.26.0", "hausdorff_distance (modified)", "ref_empty"): "+Inf",
    ("seg-metrics-1.2.8", "dice", "both_empty"): "1",
    ("seg-metrics-1.2.8", "dice", "pred_empty"): "0",
    ("seg-metrics-1.2.8", "hd95", "both_empty"): "0",
    ("seg-metrics-1.2.8", "hd95", "pred_empty"): "0",
}


def cmd_check(directory: Path) -> int:
    files = {p.stem: json.loads(p.read_text(encoding="utf-8")) for p in sorted(directory.glob("*.json"))}
    if not files:
        raise SystemExit(f"no probe files in {directory}")
    rows, bad = [], 0
    for name, rec in files.items():
        for metric, per_case in rec["results"].items():
            row = {"library": f"{rec['library']} {rec['version']}", "metric": metric}
            row.update({c: outcome(per_case[c]) for c in CASES})
            row["warning on an empty case"] = any(per_case[c]["warnings"] for c in CASES if c != "control")
            rows.append(row)
    for (stem, metric, case), want in EXPECTED.items():
        if stem not in files:
            print(f"  (not probed here: {stem})")
            continue
        got = outcome(files[stem]["results"][metric][case])
        ok = got == want
        bad += not ok
        print(f"  {'ok ' if ok else 'MISMATCH'} {stem:<20} {metric:<28} {case:<11} {got}")
    monai = sorted(k for k in files if k.startswith("monai-"))
    for other in monai[1:]:   # Section 3.1: the empty-case behaviour is unchanged across MONAI versions
        same = all(outcome(files[monai[0]]["results"][m][c]) == outcome(files[other]["results"][m][c])
                   for m in files[monai[0]]["results"] for c in CASES)
        bad += not same
        print(f"  {'ok ' if same else 'MISMATCH'} {monai[0]} and {other} give identical outcomes: {same}")
    import pandas as pd
    out = OUTPUTS / "tables" / "table3_library_probe.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"{len(rows)} probed metrics -> {out}; {bad} mismatches with Table 3")
    return 1 if bad else 0


def cmd_monai_predictions(pred_root: Path, data: str, out: Path, threshold: float = 0.5):
    import pandas as pd
    import torch
    from monai.metrics import DiceMetric, HausdorffDistanceMetric
    from PIL import Image
    masks = sorted((Path(data) / "val" / "masks").iterdir())
    gts = [np.asarray(Image.open(m).convert("L")) / 255.0 >= 0.5 for m in masks]
    rows = []
    for run in sorted(p for p in pred_root.iterdir() if p.is_dir() and "_seed" in p.name):
        model, seed = run.name.rsplit("_seed", 1)
        hd = HausdorffDistanceMetric(include_background=True, percentile=95, reduction="mean", get_not_nans=True)
        dc = DiceMetric(include_background=True, reduction="mean", get_not_nans=True)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for m, gt in zip(masks, gts):
                pr = np.load(run / f"{m.stem}.npy") >= threshold
                tp = torch.from_numpy(pr.astype(np.float32))[None, None]
                tg = torch.from_numpy(gt.astype(np.float32))[None, None]
                hd(tp, tg)
                dc(tp, tg)
        (h, hn), (d, dn) = hd.aggregate(), dc.aggregate()
        rows.append({"model": model, "seed": int(seed), "monai_hd95_default": float(h), "hd95_not_nans": float(hn),
                     "n_frames": len(masks), "monai_dice_default": float(d), "dice_not_nans": float(dn)})
        print(f"{run.name}: HD95 {float(h):.2f} over {float(hn):.0f} frames, Dice {float(d):.4f} over {float(dn):.0f}")
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"-> {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("probe")
    p.add_argument("--out", default=str(OUTPUTS / "library_probe"))
    c = sub.add_parser("check")
    c.add_argument("--dir", default=str(RESULTS / "library_probe"))
    m = sub.add_parser("monai-predictions")
    m.add_argument("--pred-root", required=True)
    m.add_argument("--data", required=True, help="prepared mixed partition (uses its val split)")
    m.add_argument("--out", default=str(OUTPUTS / "library_probe" / "monai_on_predictions.csv"))
    a = ap.parse_args()
    if a.cmd == "probe":
        cmd_probe(Path(a.out))
    elif a.cmd == "check":
        sys.exit(cmd_check(Path(a.dir)))
    else:
        cmd_monai_predictions(Path(a.pred_root), a.data, Path(a.out))


if __name__ == "__main__":
    main()
