"""Compare every regenerated table and in-text statistic with the submitted paper.

Run ``python -m scripts.make_tables`` (and ``python -m scripts.audit`` for Table 5,
``python -m scripts.bootstrap_reversals`` for Table B.6) first, then
``python -m pytest tests``. ``paper_tables.json`` holds the cells of
the paper's tables exactly as typeset; the expected in-text numbers below are
copied from the text with their section.
"""
import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
TABLES = REPO / "outputs" / "tables"
PAPER = json.loads((Path(__file__).with_name("paper_tables.json")).read_text(encoding="utf-8"))["tables"]


def generated_rows(name):
    path = TABLES / f"{name}.tex"
    if not path.exists():
        pytest.skip(f"{path.relative_to(REPO)} not generated")
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip() and not line.startswith("%"):
            rows.append([c.strip() for c in line.rstrip().removesuffix(r"\\").split(" & ")])
    return rows


def norm(cell):
    return "".join(cell.split())


@pytest.mark.parametrize("name", sorted(PAPER))
def test_table_cells_match_paper(name):
    got, want = generated_rows(name), PAPER[name]
    assert len(got) == len(want), f"{name}: {len(got)} rows, paper has {len(want)}"
    for g, w in zip(got, want):
        assert [norm(c) for c in g] == [norm(c) for c in w], f"{name}\n got  {g}\n want {w}"


# In-text statistics: (path in outputs/numbers.json, value printed in the paper)
EXPECTED = {
    # Section 3.2
    "table4/pairs_reordered_c": {"A": 2, "B": 3, "C": 12},
    "table4/pairs_reordered_deletion_vs_diag": {"A": 1, "B": 1, "C": 5},
    "section3_2/siim_empty_reference_share_of_dice_c1_pct": [83, 89],
    "section3_2/panelB_models_r0_above_95pct": 4,
    "section3_2/panelB_unetpp_gain_c1_minus_c0": "0.226",
    # Section 3.3
    "section3_3/deletion_to_diag_shift_px": {"A": ["21", "34"], "C": ["25", "103"]},
    "section3_3/spread_on_nonempty_refs_px": {"A": "20", "C": "29"},
    "section3_3/siim_P128_P256_reproduce_diag_ranking": True,
    "section3_3/siim_deletion_reversed_pairs": 5,
    "section3_3/scored_lesion_frames": {"A": ["256", "276", 276], "C": ["374", "433", 476]},
    "section3_3/siim_corr_deletion_hd95_vs_frames": "0.96",
    "section3_3/siim_q_of_those_pct": ["21", "21"],
    # Section 3.4
    "section3_4/always_empty_dice/C": "0.777",
    "section3_4/models_below_always_empty": {"A": 0, "B": 4, "C": 8},
    "section3_4/siim_best_model_dice_c1": "0.713",
    "section3_4/siim_N0_over_Npos": "3.49",
    "section3_4/siim_models_that_cannot_exceed_even_with_dice_pos_1": 5,
    "section3_4_ablation/dice_full_improved": 2,
    "section3_4_ablation/boundary_full_improved": 3,
    "section3_4_ablation/nonempty_intervals_include_zero": True,
    "section3_4_ablation/hd95_sign_changes": 3,
    "section3_4_ablation/largest_dice_gain_from_empty_refs": ["1.55", "1.93"],
    # Section 3.6
    "appendixB_3/resolution_limit_exact_px": ["8", "30"],
    "appendixB_3/resolution_limit_worst_px": ["30", "40"],
    "appendixB_3/highest_q_pct": "2.91",
    "section3_6/overlap_window/A": "0.112",
    "section3_6/overlap_window/C": "1.000",
    # Section 4.3 (sensitivity and specificity also in the abstract)
    "section4_3/siim_lesion_per_1000": "223",
    "section4_3/triage": {"UNet++": {"sensitivity_pct": "78.7", "specificity_pct": "81.3",
                                     "unflagged": "47", "false_flags": "145"},
                          "EGE-UNet": {"sensitivity_pct": "91.0", "specificity_pct": "47.6",
                                       "unflagged": "20", "false_flags": "407"}},
    # Appendix A.2
    "appendixA_2/criterion_share_mixed_pct": "10.1",
    "appendixA_2/criterion_share_c4_val_pct": "1.5",
    "appendixA_2/always_empty_criterion_mixed": "0.202",
    # Appendix A.4
    "appendixA_4/canet/multiplier_range": ["1.002", "1.555"],
    "appendixA_4/canet/cv_reported": "0.141",
    "appendixA_4/canet/cv_rescaled": "0.056",
    "appendixA_4/canet/cv_decreases_for_every_method": True,
    "appendixA_4/canet/loco_mae": ["0.514", "0.233"],
    "appendixA_4/canet/loco_mae_c4": ["1.303", "0.111"],
    "appendixA_4/canet/n_predictions": 60,
    "appendixA_4/endoked/limits_px": {"256": "29.4", "448": "51.5"},
    "appendixA_4/endoked/gains_below_limit": {"256": 4, "448": 7},
    # Appendix B.1, reversals within individual seeds
    "appendixB_1_seeds/reversals_within_seeds/A/c": {"at_least_two_seeds": 2, "all_three_seeds": 1},
    "appendixB_1_seeds/reversals_within_seeds/B/c": {"at_least_two_seeds": 2, "all_three_seeds": 1},
    "appendixB_1_seeds/reversals_within_seeds/C/c": {"at_least_two_seeds": 11, "all_three_seeds": 5},
    "appendixB_1_seeds/reversals_within_seeds/A/deletion": {"at_least_two_seeds": 1, "all_three_seeds": 1},
    "appendixB_1_seeds/reversals_within_seeds/B/deletion": {"at_least_two_seeds": 1, "all_three_seeds": 0},
    "appendixB_1_seeds/reversals_within_seeds/C/deletion": {"at_least_two_seeds": 4, "all_three_seeds": 2},
    "appendixB_1_seeds/siim_reversed_pairs_seed_stable_under_c1_and_diag": {"c": True, "deletion": True},
    # Appendix B.1
    "appendixB_1/siim_eq1_max_residual": 6e-14,
    "appendixB_1/polypgen_c0_over_pos": "0.899",
    "appendixB_1/polypgen_biou_pairs_reordered_c0_vs_c1": 2,
    "appendixB_1/siim_dice_span_c0_c1": ["0.006", "0.267"],
    "appendixB_1/siim_dice_pairs_reordered": 12,
    "appendixB_1/polypgen_min_tau_b": 0.93,
    "appendixB_1/polypgen_max_reordered": 1,
    "appendixB_1/siim_24_points_tau_b": {"hd95_X": "0.68", "hd95_Xprime": "0.81", "hd95_Z": "0.75",
                                         "hd95_P128": "0.99", "hd95_P256": "1.00"},
    "appendixB_1/monai_hd95_on_retained_frames": {
        "UNet++": "23.23", "MobileUNet": "23.98", "UNet-Tiny": "27.00", "PRISM": "33.37",
        "LB-UNet": "33.86", "UNeXt-S": "38.59", "EGE-UNet": "39.25", "MALUNet": "55.48"},
    "appendixB_1/monai_vs_ours_pct": ["7", "17"],
    "appendixB_1/monai_same_order": True,
    "appendixB_1/monai_dice_equals_dice_pos": True,
    "appendixB_1/monai_dice_frames": 276,
    # Appendix B.2
    "appendixB_2/overlap_identity_max_residual_below_1e-6": True,
    "appendixB_2/distance_identity_with_seed_means_px": "0.04",
    "appendixB_2/biou_tau_b_c0_vs_c1": "0.86",
    "appendixB_2/biou_pairs_reordered": 2,
    "appendixB_2/dice_pairs_reordered": 3,
    "appendixB_2/distinct_overlap_orders": 2,
    "appendixB_2/distinct_distance_orders": 2,
    "appendixB_2/Npos_over_N": "0.643",
    "appendixB_2/distance_reordered_pairs": [["PRISM", "UNeXt-S"]],
    "appendixB_2/malunet_r0_pct": ["100.00", "0.00"],
    "appendixB_2/clopper_pearson_81_of_81_pct": ["95.5", "100"],
    "appendixB_2/corr_q_r0": "-0.99",
    # Appendix B.3
    "appendixB_3/ratio_P_over_X": ["1.50", "2.04"],
    "appendixB_3/hd95_pairs_reordered": 1,
    "appendixB_3/dice_pairs_reordered": 0,
    "appendixB_3/hd95_reordered_frame_bootstrap_95pct": [0, 2],
    "appendixB_3/tau_b_hd95": "0.93",
    "appendixB_3/tau_b_dice": "1.00",
    "appendixB_3/N_empty": 127,
    "appendixB_3/prism_ahead_of_lbunet_and_egeunet_in_all_columns": True,
    # Appendix C.1
    "appendixC_1_grid/leading_models": ["UNet++"],
    "appendixC_1_grid/dice_c0_vs_c1_reordered_range": [1, 2],
    "appendixC_1_grid/unet_derivatives_strictest_r0_pct": ["17", "28"],
    "appendixC_1_sweep/A/rise_d0_to_diag_px": ["24", "39"],
    "appendixC_1_sweep/A/pairs_reordered_Z_vs_diag": 1,
    "appendixC_1_sweep/C/rise_d0_to_diag_px": ["70", "155"],
    "appendixC_1_sweep/C/pairs_reordered_Z_vs_diag": 3,
    "appendixC_1_sweep/C/order_unchanged_128_to_diag": True,
    # Appendix D
    "appendixD/train_n_lesion": [7471, 1664],
    "appendixD/val_n_lesion": [1069, 239],
    "appendixD/three_collapse_within_two_epochs": ["egeunet", "mobileunet", "unetpp"],
    "appendixD/three_never_exceed_0.0005": True,
    "appendixD/lbunet_peak": ["0.33", 7],
    "appendixD/lbunet_collapse_epoch": 9,
    "appendixD/lbunet_reported_at_best_and_two_later": ["0.120", "0.776"],
    "appendixD/plateau": {"unetpp": "0.121", "mobileunet": "0.121", "egeunet": "0.126", "lbunet": "0.111"},
    "appendixD/lr_after_30_of_111_epochs": 8.3e-4,
    # Appendix C.2
    "appendixC_2/native_empty": 125,
    "appendixC_2/pipeline_empty": 127,
}


def lookup(numbers, path):
    node = numbers
    for key in path.split("/"):
        node = node[key]
    return node


@pytest.fixture(scope="module")
def numbers():
    path = REPO / "outputs" / "numbers.json"
    if not path.exists():
        pytest.skip("outputs/numbers.json not generated; run python -m scripts.make_tables")
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.mark.parametrize("path", sorted(EXPECTED))
def test_in_text_number(numbers, path):
    assert lookup(numbers, path) == EXPECTED[path]


# Image-level bootstrap of the rank reversals: (path under "paper" in
# outputs/bootstrap_reversals.json, value printed in the paper)
EXPECTED_BOOTSTRAP = {
    # Section 3.2
    "section3_2/siim_dice_reversals_ci95": ["7", "19"],
    "section3_2/siim_dice_exceeds_A_and_B_above_0.999": True,
    "section3_2/siim_hd95_reversals_ci95": ["1", "7"],
    # Section 3.3
    "section3_3/corr_deletion_hd95_vs_scored_ci95": ["0.84", "0.98"],
    "section3_3/unetpp_unettiny_two_best_deletion_hd95_pct": "98",
    # Section 3.4
    "section3_4/always_empty_beats_all_in_every_replicate": True,
    "section3_4/always_empty_margin_ci95": ["0.047", "0.079"],
    # Appendix B.6
    "appendixB_6/siim_dice_never_zero": True,
    "appendixB_6/p_C_exceeds_A_and_B_hd95": "0.75",
    "appendixB_6/siim_dice_recurring_90pct": 6,
    "appendixB_6/siim_dice_recurring_above_99.6pct": 4,
    "appendixB_6/siim_dice_other_recurrence_pct": ["52", "71"],
    "appendixB_6/siim_dice_other_c0_gap_below_0.001": True,
    "appendixB_6/panelA_dice_recurrence_pct": {"LB-UNet / PRISM": "55", "LB-UNet / UNeXt-S": "50"},
    "appendixB_6/panelB_dice_recurrence_pct": {"MobileUNet / UNet-Tiny": "99", "PRISM / UNeXt-S": "76",
                                               "EGE-UNet / LB-UNet": "21"},
    "appendixB_6/siim_hd95_recurrence_pct": ["54", "83"],
    "appendixB_6/panelAB_hd95_recurrence_pct": ["57", "79"],
    "appendixB_6/max_recurrence_not_reversed_pct": "49",       # "up to about half"
    "appendixB_6/unetpp_unettiny_two_best_deletion_hd95_pct": "98.2",
    "appendixB_6/corr_deletion_hd95_vs_scored_median": "0.93",
}


@pytest.fixture(scope="module")
def bootstrap():
    path = REPO / "outputs" / "bootstrap_reversals.json"
    if not path.exists():
        pytest.skip("outputs/bootstrap_reversals.json not generated; run python -m scripts.bootstrap_reversals")
    return json.loads(path.read_text(encoding="utf-8"))["paper"]


@pytest.mark.parametrize("path", sorted(EXPECTED_BOOTSTRAP))
def test_bootstrap_number(bootstrap, path):
    assert lookup(bootstrap, path) == EXPECTED_BOOTSTRAP[path]
