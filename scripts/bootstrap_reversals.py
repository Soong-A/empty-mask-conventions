"""Image-level bootstrap of the rank reversals (Section 2.6, Section 3.2-3.4, Appendix B.6, Table B.6).

Images are resampled with replacement; the same resampled images are used for every
model and seed (paired), and each replicate averages the seed values of every model,
as the point estimates of Table 4 do.

    python -m scripts.bootstrap_reversals            # 10,000 replicates (~5 s)

Writes outputs/bootstrap_reversals.json (with the in-text numbers as printed under
"paper") and outputs/tables/tableB6_bootstrap.tex.
"""
from __future__ import annotations

import argparse
import itertools
import json

import numpy as np

from emc import conventions as C
from emc import records as R
from emc.report import OUTPUTS, fmt, write_tex

PANELS = {"A": "polypgen_mixed", "B": "polypgen_c4", "C": "siim"}
SEED = 20261014


def run_arrays(g):
    g = g.sort_values("idx")
    gt, pr = g.gt_area.to_numpy() > 0, g.pred_area.to_numpy() > 0
    den = g.gt_area.to_numpy(float) + g.pred_area.to_numpy(float)
    dice = np.where(den > 0, 2 * g.inter.to_numpy(float) / np.where(den > 0, den, 1), 0.0)
    det = gt & pr
    return {"dice": dice, "both": (~gt & ~pr).astype(float), "det": det.astype(float),
            "one": (gt ^ pr).astype(float), "hd": np.where(det, g.hd95_det.to_numpy(float), 0.0),
            "empty_ref": (~gt).astype(float)}


def panel(partition: str, B: int, d: float = C.DIAGONAL) -> dict:
    df = R.load(partition)
    models = list(dict.fromkeys(df.model))
    seeds = sorted(df.seed.unique())
    n = R.PARTITIONS[partition].n_frames
    idx = np.random.default_rng(SEED).integers(0, n, size=(B, n))
    full = np.arange(n)[None, :]
    v = {k: np.zeros((len(models), B + 1)) for k in ("d0", "d1", "X", "P", "kdet")}
    empty_ref = None
    for i, m in enumerate(models):
        for s in seeds:
            a = run_arrays(df[(df.model == m) & (df.seed == s)])
            empty_ref = a["empty_ref"]
            for I, sl in ((full, slice(0, 1)), (idx, slice(1, None))):
                sd, sb = a["dice"][I].sum(1), a["both"][I].sum(1)
                sh, kd, k1 = a["hd"][I].sum(1), a["det"][I].sum(1), a["one"][I].sum(1)
                with np.errstate(invalid="ignore", divide="ignore"):
                    v["d0"][i, sl] += sd / n / len(seeds)
                    v["d1"][i, sl] += (sd + sb) / n / len(seeds)
                    v["X"][i, sl] += sh / kd / len(seeds)
                    v["P"][i, sl] += (sh + k1 * d) / n / len(seeds)
                    v["kdet"][i, sl] += kd / len(seeds)
    out = {"partition": partition, "models": models, "empty_ref_fraction": float(empty_ref.mean())}
    pairs = list(itertools.combinations(range(len(models)), 2))
    for name, (a, b) in {"dice": ("d0", "d1"), "hd95": ("X", "P")}.items():
        rev = np.stack([np.sign(v[a][i] - v[a][j]) * np.sign(v[b][i] - v[b][j]) < 0 for i, j in pairs])
        counts = rev.sum(0)
        out[name] = {
            "point": int(counts[0]),
            "ci95": [float(np.percentile(counts[1:], 2.5)), float(np.percentile(counts[1:], 97.5))],
            "p_zero": float((counts[1:] == 0).mean()),
            "recurrence": {f"{models[i]} / {models[j]}": float(rev[k, 1:].mean())
                           for k, (i, j) in enumerate(pairs) if rev[k, 0]},
            # point-estimate gap of each observed reversal under the first convention
            "point_gap": {f"{models[i]} / {models[j]}": float(abs(v[a][i, 0] - v[a][j, 0]))
                          for k, (i, j) in enumerate(pairs) if rev[k, 0]},
            # near-ties: pairs not reversed in the point estimate that reverse in replicates
            "max_recurrence_not_reversed": float(rev[~rev[:, 0], 1:].mean(1).max()),
            "counts": counts[1:].tolist(),
        }
    if partition == "siim":
        e = np.concatenate([[empty_ref.mean()], empty_ref[idx].mean(1)])
        margin = e[1:] - v["d1"][:, 1:].max(0)
        best2 = np.argsort(v["X"][:, 1:], axis=0)[:2]
        target = {models.index("UNet++"), models.index("UNet-Tiny")}
        r = [np.corrcoef(v["X"][:, b], v["kdet"][:, b])[0, 1] for b in range(1, B + 1)]
        out["siim_extra"] = {
            "always_empty_beats_all": float((e[None, 1:] > v["d1"][:, 1:]).all(0).mean()),
            "always_empty_margin_ci95": [float(np.percentile(margin, 2.5)), float(np.percentile(margin, 97.5))],
            "unetpp_unettiny_two_best_deletion_hd95": float(np.mean([set(best2[:, k]) == target for k in range(B)])),
            "corr_deletion_hd95_vs_scored_median_ci95": [float(np.median(r)), float(np.percentile(r, 2.5)),
                                                          float(np.percentile(r, 97.5))],
        }
    return out


def table_b6(res: dict) -> None:
    """Table B.6 body in the paper's cell format (outputs/tables/tableB6_bootstrap.tex)."""
    labels = {"A": "A, PolypGen mixed", "B": "B, PolypGen C4 held out", "C": "C, SIIM-ACR"}
    rows = []
    for p in "ABC":
        row = [labels[p], f"${fmt(res[p]['empty_ref_fraction'], 3)}$"]
        for name in ("dice", "hd95"):
            r = res[p][name]
            lo, hi = (fmt(x, 0) for x in r["ci95"])
            row += [f"${r['point']}$ $[{lo},{hi}]$", rf"${fmt(100 * r['p_zero'], 1)}\%$"]
        rows.append(row)
    write_tex("tableB6_bootstrap", rows)


def paper_numbers(res: dict) -> dict:
    """The in-text numbers, formatted as printed, keyed by section."""
    def pct(x):
        return fmt(100 * x, 0)
    A, B, C = res["A"], res["B"], res["C"]
    dice_c, extra = C["dice"]["recurrence"], C["siim_extra"]
    stable = {k: r for k, r in dice_c.items() if r >= 0.9}
    unstable = {k: r for k, r in dice_c.items() if r < 0.9}
    corr = extra["corr_deletion_hd95_vs_scored_median_ci95"]
    return {
        "section3_2": {
            "siim_dice_reversals_ci95": [fmt(x, 0) for x in C["dice"]["ci95"]],
            "siim_dice_exceeds_A_and_B_above_0.999": res["p_C_exceeds_A_and_B_dice"] > 0.999,
            "siim_hd95_reversals_ci95": [fmt(x, 0) for x in C["hd95"]["ci95"]],
        },
        "section3_3": {
            "corr_deletion_hd95_vs_scored_ci95": [fmt(corr[1], 2), fmt(corr[2], 2)],
            "unetpp_unettiny_two_best_deletion_hd95_pct": pct(extra["unetpp_unettiny_two_best_deletion_hd95"]),
        },
        "section3_4": {
            "always_empty_beats_all_in_every_replicate": extra["always_empty_beats_all"] == 1.0,
            "always_empty_margin_ci95": [fmt(x, 3) for x in extra["always_empty_margin_ci95"]],
        },
        "appendixB_6": {
            "siim_dice_never_zero": C["dice"]["p_zero"] == 0.0,
            "p_C_exceeds_A_and_B_hd95": fmt(res["p_C_exceeds_A_and_B_hd95"], 2),
            "siim_dice_recurring_90pct": len(stable),
            "siim_dice_recurring_above_99.6pct": sum(r > 0.996 for r in dice_c.values()),
            "siim_dice_other_recurrence_pct": [pct(min(unstable.values())), pct(max(unstable.values()))],
            "siim_dice_other_c0_gap_below_0.001": all(C["dice"]["point_gap"][k] < 0.001 for k in unstable),
            "panelA_dice_recurrence_pct": {k: pct(r) for k, r in A["dice"]["recurrence"].items()},
            "panelB_dice_recurrence_pct": {k: pct(r) for k, r in B["dice"]["recurrence"].items()},
            "siim_hd95_recurrence_pct": [pct(min(C["hd95"]["recurrence"].values())),
                                         pct(max(C["hd95"]["recurrence"].values()))],
            "panelAB_hd95_recurrence_pct": [pct(r) for p in (A, B) for r in p["hd95"]["recurrence"].values()],
            "max_recurrence_not_reversed_pct": pct(max(res[p][n]["max_recurrence_not_reversed"]
                                                       for p in "ABC" for n in ("dice", "hd95"))),
            "unetpp_unettiny_two_best_deletion_hd95_pct": fmt(100 * extra["unetpp_unettiny_two_best_deletion_hd95"], 1),
            "corr_deletion_hd95_vs_scored_median": fmt(corr[0], 2),
        },
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--replicates", type=int, default=10000)
    a = ap.parse_args(argv)
    res = {p: panel(part, a.replicates) for p, part in PANELS.items()}
    for name in ("dice", "hd95"):
        pa, pb, pc = (np.bincount(res[p][name]["counts"], minlength=29) / a.replicates for p in "ABC")
        res[f"p_C_exceeds_A_and_B_{name}"] = float(sum(pa[i] * pb[j] * pc[k] for i in range(29) for j in range(29)
                                                     for k in range(29) if k > i and k > j))
    res["paper"] = paper_numbers(res)
    table_b6(res)
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    path = OUTPUTS / "bootstrap_reversals.json"
    path.write_text(json.dumps(res, indent=1), encoding="utf-8")
    for p in "ABC":
        print(p, {n: (res[p][n]["point"], res[p][n]["ci95"], round(res[p][n]["p_zero"], 4)) for n in ("dice", "hd95")})
    print({k: round(v, 4) for k, v in res.items() if k.startswith("p_C")})
    print(res["C"]["siim_extra"])
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
