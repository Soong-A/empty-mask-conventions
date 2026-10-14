"""The emc-score tool: hand-computed conventions on the four empty-mask cases, the
command line on image folders, and agreement with the code behind the paper."""
import json

import numpy as np
import pytest
from PIL import Image

from emc import conventions as C
from emc import records as R
from emc import score as S

DIAG = float(np.hypot(32, 32))


def square(on: bool) -> np.ndarray:
    m = np.zeros((32, 32))
    if on:
        m[8:16, 8:16] = 1
    return m


# (prediction, reference): detected, complete miss, false alarm, both empty
CASES = {"hit": (square(1), square(1)), "miss": (square(0), square(1)),
         "false_alarm": (square(1), square(0)), "both_empty": (square(0), square(0))}
EXPECTED = {
    "N": 4, "N_pos": 2, "N_neg": 2, "n_det": 1, "n_miss": 1, "n_fa": 1, "n_both_empty": 1,
    "q": 0.5, "r0": 0.5, "sensitivity": 0.5, "specificity": 0.5,
    "dice_pos": 0.5, "dice_c0": 0.25, "dice_c1": 0.5,
    "biou_pos": 0.5, "biou_c0": 0.25, "biou_c1": 0.5,
    "hd95_X": 0.0, "hd95_Xprime": 0.0, "hd95_Z": 0.0, "hd95_Pdiag": 2 * DIAG / 4,
}


def test_four_cases_by_hand():
    m = S.summarise_records(S.score_pairs(CASES))
    for k, v in EXPECTED.items():
        assert m[k] == pytest.approx(v), k


def test_command_line_on_image_folders(tmp_path):
    for sub in ("pred", "ref"):
        (tmp_path / sub).mkdir()
    for name, (p, g) in CASES.items():
        Image.fromarray((255 * p).astype(np.uint8)).save(tmp_path / "pred" / f"{name}.png")
        Image.fromarray((255 * g).astype(np.uint8)).save(tmp_path / "ref" / f"{name}.png")
    out = tmp_path / "summary.json"
    assert S.main(["--pred", str(tmp_path / "pred"), "--ref", str(tmp_path / "ref"),
                   "--json", str(out), "--records-out", str(tmp_path / "records.csv")]) == 0
    m = json.loads(out.read_text(encoding="utf-8"))
    for k, v in EXPECTED.items():
        assert m[k] == pytest.approx(v), k
    # the written records score the same again, with an explicit penalty
    assert S.main(["--records", str(tmp_path / "records.csv"), "--penalty", "100",
                   "--json", str(out)]) == 0
    assert json.loads(out.read_text(encoding="utf-8"))["hd95_P100"] == pytest.approx(50.0)


def test_probability_maps_and_threshold(tmp_path):
    for sub in ("pred", "ref"):
        (tmp_path / sub).mkdir()
    prob = 0.4 * square(1)                         # foreground at 0.3, empty at 0.5
    np.save(tmp_path / "pred" / "a.npy", prob)
    Image.fromarray((255 * square(1)).astype(np.uint8)).save(tmp_path / "ref" / "a.png")
    pairs = S.pair_files(tmp_path / "pred", tmp_path / "ref")
    assert S.summarise_records(S.score_pairs(pairs, threshold=0.3))["q"] == 0.0
    assert S.summarise_records(S.score_pairs(pairs, threshold=0.5))["q"] == 1.0


def test_unpaired_files_are_reported(tmp_path):
    for sub in ("pred", "ref"):
        (tmp_path / sub).mkdir()
    Image.fromarray(np.zeros((8, 8), np.uint8)).save(tmp_path / "ref" / "a.png")
    assert S.main(["--pred", str(tmp_path / "pred"), "--ref", str(tmp_path / "ref")]) == 2


def test_released_records_match_the_paper_code():
    df = R.load("polypgen_mixed", models=["UNet++"], seeds=[42])
    m = S.summarise_records(df, penalties={"P362": C.DIAGONAL})
    ref = C.run_metrics(df)
    for k in ("q", "r0", "dice_pos", "dice_c0", "dice_c1", "biou_c1", "hd95_X", "hd95_Xprime",
              "hd95_Z", "hd95_P362", "hd95_pos"):
        assert m[k] == ref[k], k


def test_boundary_band_matches_opencv():
    cv2 = pytest.importorskip("cv2")
    from scipy import ndimage
    rng = np.random.default_rng(0)
    for i in range(500):
        h, w = rng.integers(1, 48, 2)
        m = rng.random((h, w)) < rng.uniform(0.01, 0.99)
        if i % 2:
            m = ndimage.binary_dilation(rng.random((h, w)) < 0.02, iterations=int(rng.integers(1, 6)))
        k = np.ones((7, 7), np.uint8)
        want = (cv2.dilate(m.astype(np.uint8), k) - cv2.erode(m.astype(np.uint8), k)) > 0
        assert np.array_equal(R.boundary_band(m, "mask"), want)
