"""Regenerate every figure of the paper into outputs/figures/.

    python -m scripts.make_figures                                   # figures from released files
    python -m scripts.make_figures --data datasets --pred-root predictions   # also the frame figures

Without --data, the figures that show PolypGen frames (4a, 5) or need image
statistics (B.1) are skipped; they need the prepared images (scripts/prepare_polypgen.py)
and, for 4a and 5, the stored probability maps of PRISM and LB-UNet, seed 42
(scripts/predict.py). Figure 1 needs MONAI 1.5.2.
"""
from __future__ import annotations

import argparse
import runpy
import sys
from pathlib import Path

# (paper figure, module, needs: None | "data" | "data+pred")
FIGURES = [
    ("1", "teaser", None),
    ("2", "related_work", None),
    ("3", "composition", None),
    ("4(a)", "frames", "data+pred"),            # also Figure 5
    ("4(b)", "rank_flip", None),
    ("6", "ablation", None),
    ("7", "resolution", None),
    ("8", "published", None),
    ("B.1", "failures", "data"),
    ("C.1", "penalty_sweep", None),
    ("D.1", "collapse", None),
]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=None, help="root holding the prepared partitions (e.g. datasets)")
    ap.add_argument("--pred-root", default=None, help="root holding predictions/<partition>/<Model>_seed<seed>")
    a = ap.parse_args()
    failed = []
    for number, module, needs in FIGURES:
        argv = []
        if needs and a.data is None or needs == "data+pred" and a.pred_root is None:
            print(f"Figure {number}: skipped (needs --data{' and --pred-root' if needs == 'data+pred' else ''})")
            continue
        if needs:
            argv += ["--data", str(Path(a.data) / "polypgen_mixed")]
        if needs == "data+pred":
            argv += ["--pred-root", str(Path(a.pred_root) / "polypgen_mixed")]
        print(f"Figure {number}: scripts.figures.{module}")
        sys.argv = [module, *argv]
        try:
            runpy.run_module(f"scripts.figures.{module}", run_name="__main__")
        except Exception as e:                      # keep going; report at the end
            failed.append((number, f"{type(e).__name__}: {e}"))
            print(f"  failed: {failed[-1][1]}")
    if failed:
        raise SystemExit("failed: " + "; ".join(f"Figure {n}" for n, _ in failed))


if __name__ == "__main__":
    main()
