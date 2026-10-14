"""Tables 1, 4, B.1-B.5 and C.1, and every number the text quotes from them.

Everything here is computed from released files only (records/, splits/,
results/); no image data or GPU is needed.

    python -m scripts.make_tables

Writes outputs/tables/<table>.tex (table bodies in the paper's cell format) and
outputs/numbers.json (the in-text statistics, keyed by section). The test suite
(tests/test_paper.py) compares both with the submitted paper.
"""
from __future__ import annotations

import argparse
import itertools

import numpy as np
import pandas as pd

from emc import conventions as C
from emc import records as R
from emc.report import RESULTS, SPLITS, cell, fmt, write_json, write_tex
from emc.stats import clopper_pearson, frame_bootstrap_reorders, paired_seed_bootstrap

# Model order of the evaluation scripts (rows of Table B.4).
MODELS = ["UNet++", "MobileUNet", "UNet-Tiny", "PRISM", "UNeXt-S", "LB-UNet", "EGE-UNet", "MALUNet"]
PANELS = {"A": "polypgen_mixed", "B": "polypgen_c4", "C": "siim"}
HD_COLS = ["hd95_X", "hd95_Xprime", "hd95_Z", "hd95_P128", "hd95_P256", "hd95_P362"]
NUM: dict = {}


def load_panel(partition: str, threshold=0.5, min_area=0):
    runs = C.per_run(R.load(partition, threshold, min_area, models=MODELS))
    return runs, C.summarise(runs, MODELS)


def col(s: pd.DataFrame, name: str) -> dict:
    return s[name].to_dict()


def neg(d: dict) -> dict:
    """Distances: lower is better; negate so every column is 'higher is better'."""
    return {k: -v for k, v in d.items()}


def n_reordered(a: dict, b: dict) -> int:
    return len(C.reordered_pairs(a, b))


# ---------------------------------------------------------------------------
# Table 4, Sections 3.2-3.4 and 3.6, 4.3, Appendices A.2 and B.1
# ---------------------------------------------------------------------------

def table4(panels: dict) -> None:
    rows = []
    for key, (runs, s) in panels.items():
        s = s.sort_values("dice_pos", ascending=False)
        rk_c1 = C.ranks(col(s, "dice_c1"), True)
        rk_x = C.ranks(col(s, "hd95_X"), False)
        rk_p = C.ranks(col(s, "hd95_P362"), False)
        best = {c: (s[c].idxmax() if c.startswith("dice") else s[c].idxmin())
                for c in ("dice_pos", "dice_c1", "hd95_X", "hd95_P362")}
        for m, r in s.iterrows():
            rows.append([m, cell(fmt(100 * r.q, 2)), cell(fmt(100 * r.r0, 1)),
                         cell(fmt(100 * (1 - r.q), 2)), cell(fmt(100 * (1 - r.r0), 1)), cell(fmt(r.n_det, 1)),
                         cell(fmt(r.dice_pos, 3), m == best["dice_pos"]),
                         cell(fmt(r.dice_c0, 3)),
                         cell(fmt(r.dice_c1, 3), m == best["dice_c1"], rk_c1[m]),
                         cell(fmt(r.hd95_X, 2), m == best["hd95_X"], rk_x[m]),
                         cell(fmt(r.hd95_Z, 2)),
                         cell(fmt(r.hd95_P362, 2), m == best["hd95_P362"], rk_p[m])])
    reorder_c = {k: n_reordered(col(s, "dice_c0"), col(s, "dice_c1")) for k, (_, s) in panels.items()}
    reorder_x = {k: n_reordered(neg(col(s, "hd95_X")), neg(col(s, "hd95_P362")))
                 for k, (_, s) in panels.items()}
    footer = ("% Pairs of 28 that reorder, A / B / C. Changing only c: "
              + " / ".join(str(v) for v in reorder_c.values())
              + ". Deletion rule against d = diagonal: "
              + " / ".join(str(v) for v in reorder_x.values()) + ".")
    write_tex("table4_main", rows, footer=footer)
    NUM["table4"] = {"pairs_reordered_c": reorder_c, "pairs_reordered_deletion_vs_diag": reorder_x}

    # Appendix B.1: reversals within individual training seeds
    within = {}
    for key, (runs, s) in panels.items():
        out = {}
        for name, (a, b, sign) in {"c": ("dice_c0", "dice_c1", 1),
                                   "deletion": ("hd95_X", "hd95_P362", -1)}.items():
            pairs = C.reordered_pairs({m: sign * v for m, v in col(s, a).items()},
                                      {m: sign * v for m, v in col(s, b).items()})
            k = C.seed_reversals(runs, pairs, a, b)
            out[name] = {"at_least_two_seeds": sum(x >= 2 for x in k),
                         "all_three_seeds": sum(x == 3 for x in k)}
        within[key] = out
    # SIIM-ACR: every reversed pair keeps one order across seeds under c=1 and under P(diag)
    runs_c, s_c = panels["C"]
    stable = {}
    for name, (a, b, ref, sign) in {"c": ("dice_c0", "dice_c1", "dice_c1", 1),
                                    "deletion": ("hd95_X", "hd95_P362", "hd95_P362", -1)}.items():
        pairs = C.reordered_pairs({m: sign * v for m, v in col(s_c, a).items()},
                                  {m: sign * v for m, v in col(s_c, b).items()})
        v = runs_c.set_index(["model", "seed"])[ref]
        stable[name] = all(len({np.sign(v[(m, sd)] - v[(n, sd)]) for sd in runs_c.seed.unique()}) == 1
                           for m, n in pairs)
    _, s_a = panels["A"]
    _, s_b = panels["B"]
    share = (s_c.N_neg / s_c.N * (1 - s_c.r0) / s_c.dice_c1)
    NUM["appendixB_1_seeds"] = {
        "reversals_within_seeds": within,
        "siim_reversed_pairs_seed_stable_under_c1_and_diag": stable,
    }
    NUM["section3_2"] = {
        "siim_empty_reference_share_of_dice_c1_pct": [round(100 * share.min()), round(100 * share.max())],
        "panelB_models_r0_above_95pct": int((s_b.r0 > 0.95).sum()),
        "panelB_unetpp_gain_c1_minus_c0": fmt(s_b.at["UNet++", "dice_c1"] - s_b.at["UNet++", "dice_c0"], 3),
        "panelA_lbunet_prism_dice_pos": [fmt(s_a.at["LB-UNet", "dice_pos"], 3),
                                         fmt(s_a.at["PRISM", "dice_pos"], 3)],
    }

    # Section 3.3
    gap ={k: (s.hd95_P362 - s.hd95_X) for k, (_, s) in panels.items()}
    spread_pos = {k: s.hd95_pos.max() - s.hd95_pos.min() for k, (_, s) in panels.items()}
    frames = {k: [fmt(s.n_det.min(), 0), fmt(s.n_det.max(), 0), int(s.N_pos.iloc[0])]
              for k, (_, s) in panels.items()}
    worst_q = s_c.q.nlargest(2).index.tolist()
    best_x = s_c.hd95_X.nsmallest(2).index.tolist()
    NUM["section3_3"] = {
        "deletion_to_diag_shift_px": {k: [fmt(gap[k].min(), 0), fmt(gap[k].max(), 0)] for k in ("A", "C")},
        "spread_on_nonempty_refs_px": {k: fmt(spread_pos[k], 0) for k in ("A", "C")},
        "siim_P128_P256_reproduce_diag_ranking": all(
            n_reordered(neg(col(s_c, c)), neg(col(s_c, "hd95_P362"))) == 0 for c in ("hd95_P128", "hd95_P256")),
        "siim_deletion_reversed_pairs": n_reordered(neg(col(s_c, "hd95_X")), neg(col(s_c, "hd95_P362"))),
        "scored_lesion_frames": {k: frames[k] for k in ("A", "C")},
        "siim_corr_deletion_hd95_vs_frames": fmt(np.corrcoef(s_c.hd95_X, s_c.n_det)[0, 1], 2),
        "siim_two_highest_q": worst_q, "siim_two_best_deletion_hd95": best_x,
        "siim_q_of_those_pct": [fmt(100 * s_c.at[m, "q"], 0) for m in worst_q],
    }

    # Section 3.4: the always-empty predictor scores N0/N under c = 1
    empty_score, beats = {}, {}
    for k, (_, s) in panels.items():
        f = float((s.N_neg / s.N).iloc[0])
        empty_score[k] = fmt(f, 3)
        beats[k] = int((s.dice_c1 < f).sum())
    ratio = float((s_c.N_neg / s_c.N_pos).iloc[0])
    NUM["section3_4"] = {
        "always_empty_dice": empty_score,
        "models_below_always_empty": beats,
        "siim_best_model_dice_c1": fmt(s_c.dice_c1.max(), 3),
        "siim_N0_over_Npos": fmt(ratio, 2),
        "siim_models_that_cannot_exceed_even_with_dice_pos_1": int((1.0 / s_c.r0 < ratio).sum()),
    }

    # Section 3.6: recovery window for non-empty-reference overlap
    NUM["section3_6"] = {"overlap_window": {k: fmt(min(1.0, float((s.N_neg / s.N_pos).iloc[0])), 3)
                                            for k, (_, s) in panels.items()}}

    # Appendix A.2: share of the checkpoint criterion from empty references
    frames_df = pd.read_csv(SPLITS / "polypgen_frames.csv", dtype=str)
    c4_val = frames_df[frames_df.c4 == "val"]
    f_val = (c4_val.mask_empty == "1").mean()
    NUM["appendixA_2"] = {
        "criterion_share_mixed_pct": fmt(100 * float((s_a.N_neg / s_a.N).iloc[0]), 1),
        "criterion_share_c4_val_pct": fmt(100 * f_val, 1),
        "always_empty_criterion_mixed": fmt(2 * float((s_a.N_neg / s_a.N).iloc[0]), 3),
    }

    # Section 4.3 (and the abstract): triage illustration
    per_1000 = 1000 * float((s_c.N_pos / s_c.N).iloc[0])
    NUM["section4_3"] = {
        "siim_lesion_per_1000": fmt(per_1000, 0),
        "triage": {m: {"sensitivity_pct": fmt(100 * (1 - s_c.at[m, "q"]), 1),
                       "specificity_pct": fmt(100 * (1 - s_c.at[m, "r0"]), 1),
                       "unflagged": fmt(per_1000 * s_c.at[m, "q"], 0),
                       "false_flags": fmt((1000 - per_1000) * s_c.at[m, "r0"], 0)}
                   for m in ("UNet++", "EGE-UNet")},
    }


# ---------------------------------------------------------------------------
# Appendix B.1: all six conventions
# ---------------------------------------------------------------------------

def table_b1(panels: dict) -> None:
    rows = []
    for key in ("A", "C"):
        runs, s = panels[key]
        s = s.sort_values("hd95_X")
        rk = {c: C.ranks(col(s, c), False) for c in HD_COLS}
        best = {c: s[c].idxmin() for c in HD_COLS}
        for m, r in s.iterrows():
            cells = []
            for c in HD_COLS:
                v = fmt(r[c], 2)
                if key == "C":
                    v_cell = (rf"$\mathbf{{{v}}}$" if m == best[c] else f"${v}$") \
                        + rf"{{\scriptsize$\pm{fmt(r[c + '_sd'], 2)}$}}\rk{{{rk[c][m]}}}"
                else:
                    v_cell = cell(v, m == best[c], rk[c][m])
                cells.append(v_cell)
            rows.append([m] + cells)
    write_tex("tableB1_six_conventions", rows)

    # text of Appendix B.1
    out = {}
    runs_a, s_a = panels["A"]
    runs_c, s_c = panels["C"]
    resid = max(max(C.identity_residuals(r).values()) for _, r in runs_c.iterrows())
    out["siim_eq1_max_residual"] = float(f"{resid:.0e}")
    out["polypgen_c0_over_pos"] = fmt(float((s_a.dice_c0 / s_a.dice_pos).iloc[0]), 3)
    out["polypgen_biou_pairs_reordered_c0_vs_c1"] = n_reordered(col(s_a, "biou_c0"), col(s_a, "biou_c1"))
    out["siim_dice_span_c0_c1"] = [fmt(s_c.dice_c0.max() - s_c.dice_c0.min(), 3),
                                   fmt(s_c.dice_c1.max() - s_c.dice_c1.min(), 3)]
    out["siim_dice_pairs_reordered"] = n_reordered(col(s_c, "dice_c0"), col(s_c, "dice_c1"))
    out["siim_biou_pairs_reordered"] = n_reordered(col(s_c, "biou_c0"), col(s_c, "biou_c1"))
    taus = {}
    for key in ("A", "C"):
        _, s = panels[key]
        ref = neg(col(s, "hd95_P362"))
        taus[key] = {c: {"tau_b": fmt(C.kendall_tau_b(neg(col(s, c)), ref), 2),
                         "reordered": n_reordered(neg(col(s, c)), ref)} for c in HD_COLS[:-1]}
    out["tau_b_vs_diag"] = taus
    out["polypgen_min_tau_b"] = min(float(v["tau_b"]) for v in taus["A"].values())
    out["polypgen_max_reordered"] = max(v["reordered"] for v in taus["A"].values())
    # the 24 SIIM-ACR operating points (model x seed), 276 pairs
    pts = runs_c.set_index(["model", "seed"])
    ref = (-pts["hd95_P362"]).to_dict()
    out["siim_24_points_tau_b"] = {c: fmt(C.kendall_tau_b((-pts[c]).to_dict(), ref), 2) for c in HD_COLS[:-1]}

    # MONAI's own HD95 on the stored PolypGen predictions (deletion-rule frames)
    monai = pd.read_csv(RESULTS / "library_probe" / "monai_on_predictions.csv")
    mm = monai.groupby("model").mean(numeric_only=True)
    ours = s_a["hd95_X"]
    rel = mm["monai_hd95_default"] / ours[mm.index] - 1
    both = monai.merge(runs_a[["model", "seed", "n_det"]], on=["model", "seed"], validate="one_to_one")
    same_frames = bool((both.hd95_not_nans == both.n_det).all())
    dice_same = np.allclose(mm["monai_dice_default"][s_a.index], s_a["dice_pos"], atol=5e-7)
    out["monai_hd95_on_retained_frames"] = {m: fmt(mm.at[m, "monai_hd95_default"], 2) for m in MODELS}
    out["monai_vs_ours_pct"] = [fmt(100 * rel.min(), 0), fmt(100 * rel.max(), 0)]
    out["monai_same_order"] = (list(mm["monai_hd95_default"].sort_values().index)
                               == list(ours.sort_values().index))
    out["monai_same_frames_as_deletion_rule"] = bool(same_frames)
    out["monai_dice_equals_dice_pos"] = bool(dice_same)
    out["monai_dice_frames"] = int(monai.dice_not_nans.unique()[0])
    NUM["appendixB_1"] = out


# ---------------------------------------------------------------------------
# Appendix B.2: centre-held-out partition
# ---------------------------------------------------------------------------

def tables_b2_b3(panels: dict) -> None:
    runs, s = panels["B"]
    s = s.sort_values("dice_pos", ascending=False)
    ov = ["dice_pos", "dice_c0", "dice_c1", "biou_pos", "biou_c0", "biou_c1"]
    rows_o, rows_d = [], []
    for m, r in s.iterrows():
        rows_o.append([m] + [
            (rf"$\mathbf{{{fmt(r[c], 3)}}} \pm {fmt(r[c + '_sd'], 3)}$" if m == s[c].idxmax()
             else rf"${fmt(r[c], 3)} \pm {fmt(r[c + '_sd'], 3)}$") for c in ov])
        rows_d.append([m] + [
            (rf"$\mathbf{{{fmt(r[c], 2)}}} \pm {fmt(r[c + '_sd'], 2)}$" if m == s[c].idxmin()
             else rf"${fmt(r[c], 2)} \pm {fmt(r[c + '_sd'], 2)}$") for c in HD_COLS])
    write_tex("tableB2_c4_overlap", rows_o)
    write_tex("tableB3_c4_distance", rows_d)

    res_overlap = max(abs(r.dice_c1 - r.dice_c0 - r.N_neg / r.N * (1 - r.r0)) for _, r in runs.iterrows())

    def gap_identity(r, d=C.DIAGONAL):
        return abs((r.hd95_P362 - r.hd95_X)
                   - (r.hd95_det * (r.N_pos / r.N * (1 - r.q) - 1) + d * (r.N_pos / r.N * r.q + r.N_neg / r.N * r.r0)))
    res_dist = max(gap_identity(r) for _, r in runs.iterrows())
    res_dist_means = max(gap_identity(r) for _, r in s.iterrows())
    orders_o = {tuple(sorted(MODELS, key=lambda m: -s.at[m, c])) for c in ["dice_pos", "dice_c0", "dice_c1"]}
    orders_d = {tuple(sorted(MODELS, key=lambda m: s.at[m, c])) for c in HD_COLS}
    dist_pairs = set()
    for a, b in itertools.combinations(HD_COLS, 2):
        dist_pairs |= set(C.reordered_pairs(neg(col(s, a)), neg(col(s, b))))
    lo, hi = clopper_pearson(81, 81)
    NUM["appendixB_2"] = {
        "overlap_identity_max_residual_below_1e-6": res_overlap < 1e-6,
        "distance_identity_max_residual_px": float(f"{res_dist:.1e}"),
        "distance_identity_with_seed_means_px": fmt(res_dist_means, 2),
        "biou_tau_b_c0_vs_c1": fmt(C.kendall_tau_b(col(s, "biou_c0"), col(s, "biou_c1")), 2),
        "biou_pairs_reordered": n_reordered(col(s, "biou_c0"), col(s, "biou_c1")),
        "dice_pairs_reordered": n_reordered(col(s, "dice_c0"), col(s, "dice_c1")),
        "distinct_overlap_orders": len(orders_o),
        "distinct_distance_orders": len(orders_d),
        "Npos_over_N": fmt(float((s.N_pos / s.N).iloc[0]), 3),
        "distance_reordered_pairs": sorted(dist_pairs),
        "malunet_r0_pct": [fmt(100 * s.at["MALUNet", "r0"], 2), fmt(100 * s.at["MALUNet", "r0_sd"], 2)],
        "clopper_pearson_81_of_81_pct": [fmt(100 * lo, 1), fmt(100 * hi, 0)],
        "corr_q_r0": fmt(np.corrcoef(s.q, s.r0)[0, 1], 2),
    }


# ---------------------------------------------------------------------------
# Appendix B.3: 1537-frame archive
# ---------------------------------------------------------------------------

def table_b4(bootstrap: int) -> None:
    df = R.load("polypgen_archive", models=MODELS)
    s = C.summarise(C.per_run(df), MODELS)
    rk = {c: C.ranks(col(s, c), c.startswith("dice")) for c in ("hd95_X", "hd95_P362", "dice_c1", "dice_c0")}
    best = {c: (s[c].idxmax() if c.startswith("dice") else s[c].idxmin()) for c in rk}
    rows = []
    for m, r in s.iterrows():
        dist = [(rf"$\mathbf{{{fmt(r[c], 2)}}} \pm {fmt(r[c + '_sd'], 2)}$" if m == best[c]
                 else rf"${fmt(r[c], 2)} \pm {fmt(r[c + '_sd'], 2)}$") + rf"\rk{{{rk[c][m]}}}"
                for c in ("hd95_X", "hd95_P362")]
        over = [cell(fmt(r[c], 4), m == best[c], rk[c][m]) for c in ("dice_c1", "dice_c0")]
        rows.append([m, cell(fmt(100 * r.q, 2)), cell(fmt(100 * r.r0, 2))] + dist + over)
    ratio = s.hd95_P362 / s.hd95_X
    counts = frame_bootstrap_reorders(df, MODELS, B=bootstrap)
    lo, hi = np.percentile(counts, [2.5, 97.5])
    out = {
        "ratio_P_over_X": [fmt(ratio.min(), 2), fmt(ratio.max(), 2)],
        "hd95_pairs_reordered": n_reordered(neg(col(s, "hd95_X")), neg(col(s, "hd95_P362"))),
        "dice_pairs_reordered": n_reordered(col(s, "dice_c0"), col(s, "dice_c1")),
        "hd95_reordered_frame_bootstrap_95pct": [int(lo), int(hi)],
        "tau_b_hd95": fmt(C.kendall_tau_b(neg(col(s, "hd95_X")), neg(col(s, "hd95_P362"))), 2),
        "tau_b_dice": fmt(C.kendall_tau_b(col(s, "dice_c0"), col(s, "dice_c1")), 2),
        "N_pos": int(s.N_pos.iloc[0]), "N_empty": int(s.N_neg.iloc[0]),
        "prism_ahead_of_lbunet_and_egeunet_in_all_columns": all(
            rk[c]["PRISM"] < min(rk[c]["LB-UNet"], rk[c]["EGE-UNet"]) for c in rk),
    }
    write_tex("tableB4_archive", rows, footer=(
        f"% Second distance column relative to first: x{out['ratio_P_over_X'][0]} to x{out['ratio_P_over_X'][1]}. "
        f"Pairs reordered: HD95 {out['hd95_pairs_reordered']}/28 (95% interval {out['hd95_reordered_frame_bootstrap_95pct']}), "
        f"Dice {out['dice_pairs_reordered']}/28. Kendall tau_b: HD95 {out['tau_b_hd95']}, Dice {out['tau_b_dice']}."))
    # Section 3.6: resolution limits on the archive (Equation 4, d_max = 362 px)
    n, n_pos, n_neg = float(s.N.iloc[0]), float(s.N_pos.iloc[0]), float(s.N_neg.iloc[0])
    exact = 362 * (n_neg / n * s.r0 + n_pos / n * s.q)
    worst = 362 * (n_neg / n + n_pos / n * s.q)
    out["resolution_limit_exact_px"] = [fmt(exact.min(), 0), fmt(exact.max(), 0)]
    out["resolution_limit_worst_px"] = [fmt(worst.min(), 0), fmt(worst.max(), 0)]
    out["highest_q_pct"] = fmt(100 * s.q.max(), 2)
    NUM["appendixB_3"] = out


# ---------------------------------------------------------------------------
# Appendix B.4: PRISM ablation
# ---------------------------------------------------------------------------

ARMS = ["A1", "A2", "A3", "A4"]


def ablation_contrasts(B: int = 100000, seed: int = 20260727) -> tuple[dict, list]:
    """Paired ablation contrasts, complete model (M) minus each ablation (A1-A4).

    Returns {(arm, column): (mean, ci_low, ci_high)} and the seeds. Non-empty-reference
    values are reconstructed per seed from the full-set value and r0 with Equation (1).
    Intervals are 95% percentile intervals over the eight paired seeds (sorted
    ascending; one generator per metric family, as for the published table):
      Dice                one resample of seed indices per contrast, shared by its
                          three columns (stratum contributions and full set);
      Boundary IoU, HD95  one resample per contrast and metric, in the order
                          B-IoU full, B-IoU non-empty, HD95 full, HD95 non-empty.
    """
    d = pd.read_csv(RESULTS / "ablation" / "per_seed.csv")
    n, n_pos, n_neg, r = d.n_frames, d.n_pos, d.n_empty, d.r0
    d["dice_pos_part"] = 100 * d.dice_pos * n_pos / n
    d["dice_empty_part"] = 100 * (1 - r) * n_neg / n
    d["dice_full_pp"] = 100 * d.dice_full
    d["biou_full_pp"] = d.biou_full_pct
    d["biou_pos_pp"] = 100 * (n / n_pos) * (d.biou_full_pct / 100 - (n_neg / n) * (1 - r))
    d["hd95_full"] = d.hd95_full_px
    d["hd95_pos"] = (n / n_pos) * (d.hd95_full_px - (n_neg / n) * r * C.DIAGONAL)
    m = d[d.arm == "M"].set_index("seed").sort_index()

    def diffs(arm, cols):
        a = d[d.arm == arm].set_index("seed").sort_index()
        assert list(a.index) == list(m.index), f"{arm} and M must share the eight seeds"
        return {c: (m[c] - a[c]).to_numpy() for c in cols}

    res = {}
    rng_dice = np.random.default_rng(seed)
    rng_bd = np.random.default_rng(seed)
    for arm in ARMS:
        idx = rng_dice.integers(0, len(m), size=(B, len(m)))
        for c, v in diffs(arm, ["dice_pos_part", "dice_empty_part", "dice_full_pp"]).items():
            lo, hi = np.quantile(v[idx].mean(axis=1), [0.025, 0.975])
            res[(arm, c)] = (float(v.mean()), float(lo), float(hi))
        for c, v in diffs(arm, ["biou_full_pp", "biou_pos_pp", "hd95_full", "hd95_pos"]).items():
            idx = rng_bd.integers(0, len(m), size=(B, len(m)))
            lo, hi = np.quantile(v[idx].mean(axis=1), [0.025, 0.975])
            res[(arm, c)] = (float(v.mean()), float(lo), float(hi))
    return res, [int(x) for x in m.index]


def table_b5(B: int = 100000, seed: int = 20260727) -> None:
    res, seeds = ablation_contrasts(B, seed)

    def ci(arm, c):
        mean, lo, hi = res[(arm, c)]
        s = rf"{fmt(mean, 2, True)}\;[{fmt(lo, 2, True)},{fmt(hi, 2, True)}]"
        excl = float(fmt(lo, 2)) > 0 or float(fmt(hi, 2)) < 0
        return rf"$\mathbf{{{s}}}$" if excl else f"${s}$"

    label = {"A1": r"M vs.\ A1", "A2": r"M vs.\ A2", "A3": r"M vs.\ A3$^{\dagger}$", "A4": r"M vs.\ A4"}
    rows = [[label[a], ci(a, "dice_pos_part"), ci(a, "dice_empty_part"), ci(a, "dice_full_pp")] for a in ARMS]
    rows += [[label[a], ci(a, "biou_pos_pp"), "--", ci(a, "biou_full_pp")] for a in ARMS]
    rows += [[label[a], ci(a, "hd95_pos"), "--", ci(a, "hd95_full")] for a in ARMS]
    write_tex("tableB5_ablation", rows)
    pd.DataFrame([{"contrast": f"M - {a}", "column": c, "mean": v[0], "ci95_lo": v[1], "ci95_hi": v[2]}
                  for (a, c), v in res.items()]).to_csv(
        R.REPO / "outputs" / "tables" / "tableB5_ablation.csv", index=False)

    def excludes_zero(arm, c):
        return res[(arm, c)][1] > 0 or res[(arm, c)][2] < 0
    NUM["section3_4_ablation"] = {
        "dice_full_improved": int(sum(excludes_zero(a, "dice_full_pp") and res[(a, "dice_full_pp")][0] > 0 for a in ARMS)),
        "boundary_full_improved": int(sum(
            (excludes_zero(a, "biou_full_pp") and res[(a, "biou_full_pp")][0] > 0)
            or (excludes_zero(a, "hd95_full") and res[(a, "hd95_full")][0] < 0) for a in ARMS)),
        "nonempty_intervals_include_zero": not any(excludes_zero(a, c) for a in ARMS
                                                   for c in ("dice_pos_part", "biou_pos_pp", "hd95_pos")),
        "hd95_sign_changes": int(sum(np.sign(res[(a, "hd95_pos")][0]) != np.sign(res[(a, "hd95_full")][0]) for a in ARMS)),
        "largest_dice_gain_from_empty_refs": [fmt(res[("A1", "dice_empty_part")][0], 2),
                                              fmt(res[("A1", "dice_full_pp")][0], 2)],
        "bootstrap": {"B": B, "seed": seed, "n_seeds": len(seeds), "seeds": seeds},
    }


# ---------------------------------------------------------------------------
# Tables 1 and C.1 (from the frame manifest) and Appendix C.1 (robustness grid)
# ---------------------------------------------------------------------------

def table1_and_c1() -> None:
    f = pd.read_csv(SPLITS / "polypgen_frames.csv", dtype=str)
    f["is_empty"] = f.mask_empty == "1"
    rows = []
    for centre in ["C1", "C2", "C3", "C5", "C6", "C4"]:
        g = f[f.centre == centre]
        cells = [centre]
        for split in ("train", "val", "test"):
            h = g[g.c4 == split]
            cells += ([f"${(~h.is_empty).sum()}$", f"${h.is_empty.sum()}$"] if len(h) else ["---", "---"])
        rows.append(cells)
    tot = ["Total"]
    for split in ("train", "val", "test"):
        h = f[f.c4 == split]
        tot += [f"${(~h.is_empty).sum()}$", f"${h.is_empty.sum()}$"]
    rows.append(tot)
    nrow = [r"$N$ and $N^{\varnothing}\!/N$"]
    for split in ("train", "val", "test"):
        h = f[f.c4 == split]
        nrow.append(rf"\multicolumn{{2}}{{c}}{{${len(h)}$, ${fmt(100 * h.is_empty.mean(), 2)}\%$}}")
    rows.append(nrow)
    write_tex("table1_c4_split", rows)

    centres = ["C1", "C2", "C3", "C4", "C5", "C6"]
    native = f.source_mask_empty == "1"
    pct = {c: 100 * native[f.centre == c].mean() for c in centres}
    top = max(pct, key=pct.get)
    rows = [["Frames"] + [f"${(f.centre == c).sum()}$" for c in centres] + [f"${len(f)}$"],
            ["Empty"] + [f"${native[f.centre == c].sum()}$" for c in centres] + [f"${native.sum()}$"],
            [r"Empty (\%)"] + [cell(fmt(pct[c], 2), c == top) for c in centres]
            + [f"${fmt(100 * native.mean(), 2)}$"]]
    write_tex("tableC1_empty_by_centre", rows)
    NUM["appendixC_2"] = {"native_empty": int(native.sum()), "pipeline_empty": int(f.is_empty.sum()),
                          "archive_empty_pct": fmt(100 * native.mean(), 0)}


def robustness_grid() -> None:
    df = R.load("polypgen_mixed", threshold=None, min_area=None, models=MODELS)
    cells, leaders, rev_c, rev_x = [], set(), [], []
    r_at = {}
    for (t, a), g in df.groupby(["threshold", "min_area"]):
        s = C.summarise(C.per_run(g), MODELS)
        for c in ("dice_pos", "dice_c0", "dice_c1"):
            leaders.add(s[c].idxmax())
        for c in HD_COLS:
            leaders.add(s[c].idxmin())
        rev_c.append(n_reordered(col(s, "dice_c0"), col(s, "dice_c1")))
        rev_x.append(n_reordered(neg(col(s, "hd95_X")), neg(col(s, "hd95_P362"))))
        r_at[(t, a)] = s.r0
        cells.append({"threshold": t, "min_area": a, **{f"r0_{m}": s.at[m, "r0"] for m in MODELS},
                      **{f"q_{m}": s.at[m, "q"] for m in MODELS}})
    pd.DataFrame(cells).to_csv(R.REPO / "outputs" / "tables" / "robustness_grid.csv", index=False)
    strict = r_at[(0.7, 100)]
    NUM["appendixC_1_grid"] = {
        "n_cells": len(cells),
        "leading_models": sorted(leaders),
        "dice_c0_vs_c1_reordered_range": [min(rev_c), max(rev_c)],
        "deletion_vs_diag_reordered_range": [min(rev_x), max(rev_x)],
        "strictest_cell_r0_pct": {m: fmt(100 * strict[m], 0) for m in MODELS},
        "unet_derivatives_strictest_r0_pct": [fmt(100 * strict[["UNet++", "MobileUNet", "UNet-Tiny"]].min(), 0),
                                              fmt(100 * strict[["UNet++", "MobileUNet", "UNet-Tiny"]].max(), 0)],
    }


def penalty_sweep(panels: dict) -> None:
    out = {}
    for key, part in (("A", None), ("C", None)):
        _, s = panels[key]
        rise = s.hd95_P362 - s.hd95_Z
        out[key] = {"rise_d0_to_diag_px": [fmt(rise.min(), 0), fmt(rise.max(), 0)],
                    "pairs_reordered_Z_vs_diag": n_reordered(neg(col(s, "hd95_Z")), neg(col(s, "hd95_P362"))),
                    "order_unchanged_128_to_diag": all(
                        n_reordered(neg(col(s, c)), neg(col(s, "hd95_P362"))) == 0 for c in ("hd95_P128", "hd95_P256"))}
    NUM["appendixC_1_sweep"] = out


# ---------------------------------------------------------------------------
# Appendix A.4 (summarised in Section 3.6): two published results
# ---------------------------------------------------------------------------

def published_scores() -> None:
    """CANet per-centre distances (family-Z rescaling, leave-one-centre-out) and the
    EndoKED Hausdorff gains against the worst-case resolution limit."""
    df = pd.read_csv(RESULTS / "published" / "canet_table3.csv")
    e = df.groupby("centre")[["n_empty", "n_frames"]].first()
    e = e.n_empty / e.n_frames
    H = df.pivot(index="method", columns="centre", values="hd")
    resc = H / (1 - e)
    cv = lambda X: X.std(axis=1, ddof=0) / X.mean(axis=1)  # noqa: E731
    err = {"naive": [], "rescaled": []}
    err_c4 = {"naive": [], "rescaled": []}
    for m in H.index:
        for c in H.columns:
            others = [x for x in H.columns if x != c]
            naive = abs(H.at[m, c] - H.loc[m, others].mean())
            corr = abs(H.at[m, c] - (1 - e[c]) * resc.loc[m, others].mean())
            err["naive"].append(naive); err["rescaled"].append(corr)
            if c == "C4":
                err_c4["naive"].append(naive); err_c4["rescaled"].append(corr)
    gains = pd.read_csv(RESULTS / "published" / "endoked_hd_gains.csv")
    f = 125 / 1537                     # empty masks of the released annotation, q = 0
    limits = {256: f * float(np.hypot(256, 256)), 448: f * float(np.hypot(448, 448))}
    NUM["appendixA_4"] = {
        "canet": {
            "multiplier_range": [fmt((1 / (1 - e)).min(), 3), fmt((1 / (1 - e)).max(), 3)],
            "cv_reported": fmt(cv(H).mean(), 3), "cv_rescaled": fmt(cv(resc).mean(), 3),
            "cv_decreases_for_every_method": bool((cv(resc) < cv(H)).all()),
            "loco_mae": [fmt(np.mean(err["naive"]), 3), fmt(np.mean(err["rescaled"]), 3)],
            "loco_mae_c4": [fmt(np.mean(err_c4["naive"]), 3), fmt(np.mean(err_c4["rescaled"]), 3)],
            "n_predictions": len(err["naive"]),
        },
        "endoked": {
            "limits_px": {k: fmt(v, 1) for k, v in limits.items()},
            "gains_below_limit": {k: int((gains.hd_gain < v).sum()) for k, v in limits.items()},
            "n_gains": len(gains),
        },
    }


# ---------------------------------------------------------------------------
# Appendix D: training-side stress test (released validation curves)
# ---------------------------------------------------------------------------

# Total BCE and Dice weight of each objective, summed over heads (arch.py):
# LB-UNet  main BceDice(1,2) + 5 deep (0.1..0.5) and 3 boundary (0.1..0.3) heads BceDice(0.5,1)
# EGE-UNet main BceDice(1,2) + 5 deep heads BceDice(0.5,1) weighted 0.1..0.5
# UNet++   mean of four BceDice(1,1) heads;  MobileUNet BceDice(1,1)
OBJECTIVE_WEIGHTS = {"lbunet": (2.05, 4.1), "egeunet": (1.75, 3.5), "unetpp": (1.0, 1.0), "mobileunet": (1.0, 1.0)}


def stress_test() -> None:
    """Collapse of four architectures under a per-image Dice term (SIIM-ACR, 30 epochs).

    The plateau value removes the overlap contribution an empty prediction receives in
    the rebalanced stream (half of the objective's Dice weight: lesion-bearing images
    score Dice loss 1, lesion-free images 0) and divides by the BCE weight; it uses
    the mean training loss of the last ten epochs.
    """
    import json
    v = pd.read_csv(RESULTS / "stress_test" / "val_curves.csv")
    runs = json.loads((RESULTS / "stress_test" / "runs.json").read_text(encoding="utf-8"))
    out = {"plateau": {}, "first_silent_epoch": {}, "peak_dice_pos": {}}
    for net, g in v.groupby("network"):
        g = g.sort_values("epoch")
        wb, wd = OBJECTIVE_WEIGHTS[net]
        out["plateau"][net] = fmt((g.train_loss.tail(10).mean() - 0.5 * wd) / wb, 3)
        silent = g[g.q >= 1.0].epoch
        out["first_silent_epoch"][net] = int(silent.min()) if len(silent) else None
        out["peak_dice_pos"][net] = float(g.dice_pos.max())
    lb = v[v.network == "lbunet"].set_index("epoch")
    best = runs["lbunet"]["best_epoch"]
    comp = runs["lbunet"]["split_composition"]
    lr_end = 1e-5 + (1e-3 - 1e-5) * (1 + np.cos(np.pi * 30 / 111)) / 2
    out.update({
        "train_n_lesion": [comp["train"]["n"], comp["train"]["n_lesion"]],
        "val_n_lesion": [comp["val"]["n"], comp["val"]["n_lesion"]],
        "three_collapse_within_two_epochs": sorted(
            n for n in out["first_silent_epoch"] if n != "lbunet" and out["first_silent_epoch"][n] <= 2),
        "three_never_exceed_0.0005": all(out["peak_dice_pos"][n] < 5e-4 for n in ("egeunet", "unetpp", "mobileunet")),
        "lbunet_peak": [fmt(lb.dice_pos.max(), 2), int(lb.dice_pos.idxmax())],
        "lbunet_collapse_epoch": out["first_silent_epoch"]["lbunet"],
        "lbunet_reported_at_best_and_two_later": [fmt(lb.at[best, "dice_full_c1"], 3),
                                                  fmt(lb.at[best + 2, "dice_full_c1"], 3)],
        "lr_after_30_of_111_epochs": float(f"{lr_end:.1e}"),
    })
    NUM["appendixD"] = out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bootstrap", type=int, default=10000,
                    help="frame-bootstrap replicates for Table B.4 (paper: 10000)")
    args = ap.parse_args()

    panels = {k: load_panel(p) for k, p in PANELS.items()}
    table1_and_c1()
    table4(panels)
    table_b1(panels)
    tables_b2_b3(panels)
    table_b4(args.bootstrap)
    table_b5()
    robustness_grid()
    penalty_sweep(panels)
    published_scores()
    stress_test()
    path = write_json("numbers", NUM)
    print(f"tables written to outputs/tables/, in-text numbers to {path.relative_to(R.REPO)}")


if __name__ == "__main__":
    main()
