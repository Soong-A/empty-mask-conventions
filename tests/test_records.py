"""Consistency of the released records with each other and with the released result files.

These tests need only the repository (records/, results/); the last one also needs the
prepared PolypGen frames in datasets/polypgen_mixed and is skipped without them.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from emc import conventions as C
from emc import records as R
from emc.report import RESULTS

REPO = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("partition", sorted(R.PARTITIONS))
def test_records_load(partition):
    df = R.load(partition)            # validates one row per frame for every run and setting
    assert set(df.columns) >= set(R.COLUMNS)
    assert (df.inter <= np.minimum(df.gt_area, df.pred_area)).all()
    defined = (df.gt_area > 0) & (df.pred_area > 0)
    assert df.hd95_det.notna().eq(defined).all(), "hd95_det must be defined exactly when both masks are non-empty"


def test_ablation_records_reproduce_per_seed_results():
    """Dice and r0 of results/ablation/per_seed.csv equal the records (up to the 1e-5
    smoothing of the training-time evaluation). Its B-IoU and HD95 were logged by that
    evaluation to two decimals; regenerated from the checkpoints they agree to
    <= 0.01 percentage points and <= 0.5 px (33 of 40 runs within 0.01 px)."""
    runs = C.per_run(R.load("polypgen_ablation")).rename(columns={"model": "variant"})
    m = pd.read_csv(RESULTS / "ablation" / "per_seed.csv").merge(runs, on=["variant", "seed"],
                                                                 suffixes=("", "_rec"))
    assert len(m) == 40
    assert np.allclose(m.dice_full, m.dice_c1, atol=1e-7)
    assert np.allclose(m.dice_pos, m.dice_pos_rec, atol=1e-7)
    assert np.allclose(m.r0, m.r0_rec, atol=1e-12)
    assert (m.biou_full_pct - 100 * m.biou_c1).abs().max() <= 0.01
    assert (m.hd95_full_px - m.hd95_P362).abs().max() <= 0.5


@pytest.mark.skipif(not (REPO / "datasets" / "polypgen_mixed" / "val").exists(),
                    reason="needs datasets/polypgen_mixed (scripts/prepare_polypgen.py)")
def test_failure_analysis_counts():
    """Appendix B.4 / Figure B.1: 52 failure frames, 32 of them with low boundary contrast."""
    from scripts.figures import failures as F
    _, attrs = F.frame_table(REPO / "datasets" / "polypgen_mixed")
    f = F.failures(attrs)
    assert len(attrs) == 276 and len(f) == 52
    assert [int(f[k].sum()) for k in ("miss", "tiny", "multiple", "low_contrast")] == [1, 12, 7, 32]
