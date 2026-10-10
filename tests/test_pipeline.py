"""Smoke test of the training -> prediction -> records chain on a synthetic dataset.

Every model trains for one epoch on 16 random 256 x 256 frames (CPU), predicts four
validation frames and is turned into records; this checks that the code paths run and
that their outputs have the released format, not the numbers of the paper.
"""
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from emc import models
from emc import records as R

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def dataset(tmp_path_factory):
    root = tmp_path_factory.mktemp("synthetic")
    rng = np.random.default_rng(0)
    for split, n in (("train", 16), ("val", 4), ("test", 4)):
        for sub in ("images", "masks", "points_boundary2"):
            (root / split / sub).mkdir(parents=True)
        for i in range(n):
            img = rng.integers(0, 256, (256, 256, 3), dtype=np.uint8)
            mask = np.zeros((256, 256), np.uint8)
            if i % 4:                                  # one empty reference in four
                r, c = rng.integers(40, 216, 2)
                mask[r - 30:r + 30, c - 30:c + 30] = 255
            name = f"{i:05d}.png"
            Image.fromarray(img).save(root / split / "images" / name)
            Image.fromarray(mask).save(root / split / "masks" / name)
            Image.fromarray(mask).save(root / split / "points_boundary2" / name)
    return root


def run(*args):
    out = subprocess.run([sys.executable, "-m", *args], cwd=REPO, capture_output=True, text=True)
    assert out.returncode == 0, out.stdout[-2000:] + out.stderr[-4000:]
    return out.stdout


@pytest.mark.parametrize("key", sorted(models.MODELS))
def test_train_predict_records(key, dataset, tmp_path):
    protocol = "siim" if key == "egeunet" else "polypgen"     # exercise both objective families
    run("scripts.train", "--model", key, "--protocol", protocol, "--seed", "42", "--data", str(dataset),
        "--epochs", "1", "--workers", "0", "--device", "cpu", "--out", str(tmp_path / "run"))
    assert (tmp_path / "run" / "best.pth").exists() and (tmp_path / "run" / "run.json").exists()
    split = "val" if protocol == "polypgen" else "test"
    name = models.MODELS[key].name
    run("scripts.predict", "--model", key, "--checkpoint", str(tmp_path / "run" / "best.pth"),
        "--protocol", protocol, "--data", str(dataset), "--split", split, "--seed", "42",
        "--device", "cpu", "--out", str(tmp_path / "pred" / f"{name}_seed42"))
    maps = sorted((tmp_path / "pred" / f"{name}_seed42").glob("*.npy"))
    assert len(maps) == 4 and all(np.load(m).shape == (256, 256) for m in maps)
    out = tmp_path / "records.csv.gz"
    run("scripts.make_records", "--partition", "polypgen_mixed", "--data", str(dataset), "--split", split,
        "--pred-root", str(tmp_path / "pred"), "--thresholds", "0.3", "0.5", "--min-areas", "0", "10",
        "--out", str(out))
    rec = pd.read_csv(out)
    assert list(rec.columns) == R.COLUMNS and len(rec) == 4 * 2 * 2
    assert set(rec.model) == {name}


@pytest.mark.parametrize("script", [
    "scripts.make_tables", "scripts.audit", "scripts.train", "scripts.predict", "scripts.make_records",
    "scripts.prepare_polypgen", "scripts.prepare_siim", "scripts.lbunet_points", "scripts.library_probe",
    "scripts.check_data", "scripts.count_parameters",
    "scripts.figures.teaser", "scripts.figures.related_work", "scripts.figures.composition",
    "scripts.figures.rank_flip", "scripts.figures.frames", "scripts.figures.ablation",
    "scripts.figures.resolution", "scripts.figures.published", "scripts.figures.penalty_sweep",
    "scripts.figures.collapse", "scripts.figures.failures"])
def test_help(script):
    run(script, "--help")
