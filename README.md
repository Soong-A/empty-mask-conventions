# When empty predictions win: empty-mask conventions in medical image segmentation

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23373345.svg)](https://doi.org/10.5281/zenodo.23373345)

Code and data for

> Yurong Song, Zhewei Chen, Jianbin Wu. **When empty predictions win: Empty-mask conventions alter
> model rankings and evaluated subsets in medical image segmentation.** Submitted to *Computer
> Methods and Programs in Biomedicine*.

Per-image metrics such as Dice, Boundary IoU and HD95 are undefined when the reference mask, the
prediction, or both are empty, and every evaluation code resolves this with some convention. The
paper shows that the choice changes scores and model rankings when the predictions are held fixed. It
also shows that exclusion rules (MONAI's default) make the evaluated subset depend on the model, and
that the convention usually cannot be recovered from a published number.

This repository releases everything behind the numbers in the paper:

* the **per-image records** of every evaluated run (convention-free counts from which every
  convention is computed);
* the scripts that turn them into **every table, in-text number and figure**;
* the full chain behind the records: **data preparation, model code, training, inference and
  record extraction**;
* the **metric-library probe**, the **literature-audit** sheets, the **split manifests** and the
  **provenance** of every checkpoint;
* **`emc-score`**, a small pip-installable tool that scores your own masks under every convention
  (Section 5.4).

Every table and figure can be regenerated on a CPU in about two minutes from the released files.
The test suite then checks each one against the submitted paper. Re-running the experiments
themselves needs the public datasets and a GPU.

---

## Contents

1. [Repository layout](#1-repository-layout)
2. [Installation](#2-installation)
3. [Quick start: all tables and figures from the released records](#3-quick-start)
4. [Where each result of the paper comes from](#4-where-each-result-comes-from)
5. [Per-image records and empty-mask conventions](#5-per-image-records-and-empty-mask-conventions)
6. [Full pipeline: from the public datasets to the records](#6-full-pipeline)
7. [Released result files](#7-released-result-files)
8. [Tests](#8-tests)
9. [Reproducibility notes](#9-reproducibility-notes)
10. [Third-party code](#10-third-party-code)
11. [Citation](#11-citation)

---

## 1. Repository layout

```
emc/                          the library ("empty-mask conventions")
  score.py                    emc-score: every convention for your own masks (Section 5.4)
  records.py                  per-image record schema, partitions, HD95 / boundary band, I/O
  conventions.py              every convention (families P, Z, X, X'), ranks, Kendall tau_b, crossings
  stats.py                    Clopper-Pearson intervals, frame and paired-seed bootstrap
  report.py                   rounding (half away from zero) and LaTeX/JSON writers
  plotting.py                 shared figure style
  models/                     the eight architectures (registry in models/__init__.py)
  training/                   protocols, transforms, losses, training engine, datasets
scripts/
  make_tables.py              Tables 1, 4, B.1-B.5, C.1 + every in-text number (outputs/numbers.json)
  bootstrap_reversals.py      image-level bootstrap of the rank reversals, Table B.6
  make_figures.py             every figure (calls scripts/figures/*)
  audit.py                    literature audit, Table 5
  library_probe.py            metric-library probe, Table 3; MONAI on the stored predictions
  count_parameters.py         parameter counts (Section 2.2, Appendix A.2)
  check_data.py               empty annotations in the released data (Appendix C.2, Table C.1)
  prepare_polypgen.py         PolypGen partitions from the public release
  prepare_siim.py             SIIM-ACR partition from the Kaggle release
  lbunet_points.py            boundary-point targets for training LB-UNet
  train.py                    train one model / seed / protocol
  predict.py                  probability maps of one checkpoint
  make_records.py             probability maps -> per-image records
  figures/                    one module per figure
records/                      per-image records of every evaluated run (gzip CSV, 13 MB)
splits/                       frame-level split manifests (PolypGen, SIIM-ACR)
results/                      released inputs that are not records (see Section 7)
tests/                        comparison with the paper, record consistency, pipeline smoke test
outputs/                      everything generated (not tracked)
```

Every script is run as a module from the repository root (`python -m scripts.<name>`). Every
script takes `--help`.

## 2. Installation

Python 3.10 or newer. Install PyTorch and torchvision for your platform first
(<https://pytorch.org>), then:

```bash
pip install -r requirements.txt
```

`requirements.txt` pins the versions with which every command in this README was last run (Python 3.12,
CPU PyTorch). The scorer of Section 5.4 also installs on its own, without PyTorch:

```bash
pip install "git+https://github.com/Soong-A/empty-mask-conventions@v1.3"
```

Notes:

* `matplotlib==3.10.8`: 3.11 changes text metrics and breaks the layout search of Figure 2.
* `monai==1.5.2` is the version probed in the paper (Figure 1, Table 3). MONAI 1.6.1 gives identical
  results (`results/library_probe/monai-1.6.1.json`).
* MedPy 0.4.0 and 0.5.2 cannot be installed in the same environment. Table 3 needs both, so probe
  0.4.0 in a second environment (Section 6.6).
* The figures use Times New Roman and fall back to Liberation Serif or DejaVu Serif, so outputs differ
  slightly from the paper's PDFs at the pixel level only.

## 3. Quick start

No dataset, checkpoint or GPU is needed for this step.

```bash
python -m scripts.make_tables      # Tables 1, 4, B.1-B.5, C.1 and outputs/numbers.json   (~10 s)
python -m scripts.audit            # Table 5 (literature audit)                            (~1 s)
python -m scripts.bootstrap_reversals  # Table B.6, bootstrap of the rank reversals      (~15 s)
python -m scripts.library_probe check   # Table 3 from the released probe files
python -m scripts.count_parameters      # parameter counts
python -m scripts.make_figures     # Figures 1, 2, 3, 4(b), 6, 7, B.2, C.1, D.1            (~75 s)
python -m pytest tests             # compare everything with the paper
```

Outputs go to `outputs/tables/` (LaTeX table bodies in the paper's cell format, plus CSV with
unrounded values), `outputs/numbers.json` (in-text statistics, keyed by section) and
`outputs/figures/`.

Three figures show PolypGen frames or need image statistics. They need the prepared images and,
for Figures 4(a) and 5, the probability maps of PRISM and LB-UNet for seed 42 (Section 6):

```bash
python -m scripts.make_figures --data datasets --pred-root predictions
```

## 4. Where each result comes from

All commands are `python -m scripts.<...>`. "records" means only the released files are needed.

| Paper item | Command | Needs | Output |
|---|---|---|---|
| Table 1 (centre-held-out split) | `make_tables` | splits | `tables/table1_c4_split.tex` |
| Table 2 (notation), Table 6 (reporting checklist), Table A.1 (PRISM stages) | (text; Table A.1 is `emc/models/prism.py`) | | |
| Table 3 (library behaviour on empty masks) | `library_probe probe`, then `library_probe check` | libraries | `library_probe/*.json`, `tables/table3_library_probe.csv` |
| Table 4 (main, Panels A-C) | `make_tables` | records | `tables/table4_main.tex` |
| Table 5 (literature audit) | `audit` | results/audit | `tables/table5_audit.tex` |
| Table B.1 (six conventions, PolypGen and SIIM-ACR) | `make_tables` | records | `tables/tableB1_six_conventions.tex` |
| Tables B.2, B.3 (centre-held-out partition) | `make_tables` | records | `tables/tableB2_c4_overlap.tex`, `tableB3_c4_distance.tex` |
| Table B.4 (1537-frame archive) | `make_tables` | records | `tables/tableB4_archive.tex` |
| Table B.5 (PRISM ablation) | `make_tables` | results/ablation | `tables/tableB5_ablation.tex` |
| Table B.6 and the bootstrap numbers of Sections 3.2-3.4 and Appendix B.6 (10,000 image resamples, seed 20261014) | `bootstrap_reversals` | records | `tables/tableB6_bootstrap.tex`, `bootstrap_reversals.json` |
| Table C.1 (empty masks by centre) | `make_tables`; from the raw release: `check_data polypgen` | splits / raw data | `tables/tableC1_empty_by_centre.tex` |
| In-text numbers of Sections 3.2-3.6, 4.3 and Appendices A.2, A.4, B-D | `make_tables` | records, results | `numbers.json` |
| MONAI on the stored predictions (Sections 3.2-3.3, Appendix B.1) | `library_probe monai-predictions` | predictions | `library_probe/monai_on_predictions.csv` |
| Parameter counts (Section 2.2, Appendix A.2) | `count_parameters` | | `parameters.csv` |
| Figure 1 | `figures.teaser` | MONAI 1.5.2 | `figures/fig_teaser.png` |
| Figure 2 | `figures.related_work` | | `figures/fig_related.pdf` |
| Figure 3 | `figures.composition` | records | `figures/fig_composition.pdf` |
| Figure 4(a), Figure 5 | `figures.frames` | images, predictions | `figures/fig_compare.png`, `fig_misses_rules.png` |
| Figure 4(b) | `figures.rank_flip` | records | `figures/fig_compare_rank.pdf` |
| Figure B.2 | `figures.ablation` | results/ablation | `figures/fig_ablation.pdf` |
| Figure 6 | `figures.resolution` | records | `figures/fig_bands.pdf` |
| Figure 7 | `figures.published` | results/published | `figures/fig_published.pdf` |
| Figure B.1 | `figures.failures` | records, images | `figures/fig_analysis_8seed.pdf` |
| Figure C.1 | `figures.penalty_sweep` | records | `figures/fig_convention_sensitivity.pdf` |
| Figure D.1 | `figures.collapse` | results/stress_test | `figures/fig_objective_collapse.pdf` |
| SIIM-ACR resize erasure (Appendix C.2) | `check_data siim --rle train-rle.csv` | Kaggle CSV | `data_checks/siim_resize_erasure.csv` |

Output paths are relative to `outputs/`.

## 5. Per-image records and empty-mask conventions

### 5.1 Schema

`records/<partition>.csv.gz` holds one row per (run, image, threshold, minimum component area):

| Column | Meaning |
|---|---|
| `partition`, `model`, `seed`, `idx`, `image` | run and frame (`image` is the file name in the prepared partition) |
| `threshold`, `min_area` | binarisation threshold and minimum 8-connected component area (0 = no filtering) |
| `gt_area`, `pred_area`, `inter` | foreground pixels of the reference, of the prediction, and of their intersection |
| `hd95_det` | 95th-percentile symmetric boundary distance in pixels, defined only when both masks are non-empty (empty otherwise) |
| `biou_inter`, `biou_union` | intersection and union of the two boundary bands (Boundary IoU = inter / union) |

No convention is applied when the records are written. An empty mask leaves `hd95_det` undefined,
and each convention decides what to substitute (`emc/conventions.py`). Load the records with
`emc.records.load(partition)`, which also checks that every run covers every frame.

| Partition | Frames | Runs | Paper |
|---|---|---|---|
| `polypgen_mixed` | 307 (276 lesion-bearing, 31 empty) | 8 models x 3 seeds | Table 4 Panel A, Figures 3-6, C.1 |
| `polypgen_c4` | 227 (centre C4 held out; 81 empty) | 8 x 3 | Table 4 Panel B, Tables B.2-B.3 |
| `polypgen_archive` | 1537 (all frames; the mixed-partition checkpoints) | 8 x 3 | Table B.4, Figure 6 |
| `siim` | 2135 (SIIM-ACR held-out test) | 8 x 3 | Table 4 Panel C, Table B.1 |
| `polypgen_ablation` | 307 (= `polypgen_mixed`) | 5 PRISM arms x 8 seeds | Appendix B.4, Figure B.1 |

Seeds are 42, 1337 and 2024. The ablation uses 7, 42, 1337, 2024, 3407, 12345, 271828 and 314159.
PRISM in Table 4 is ablation arm M (`prism_dn`, the same checkpoints for seeds 42, 1337 and 2024).

### 5.2 Conventions

With N frames, N+ lesion-bearing and N0 lesion-free, complete-miss rate q and lesion-free
false-alarm rate r0 (`emc.conventions.run_metrics`):

* **Family P(d)**: a one-sided empty frame scores the penalty d (distance) or 0 (overlap). Two empty
  masks score 0 (distance) or c (overlap, c = 1 or 0). The paper uses d = 362 = the image diagonal,
  128 and 256.
* **Family Z**: P with d = 0.
* **Family X** (deletion, MONAI's default reduction): the mean over frames where both masks are non-empty.
* **X′**: one-sided empty frames are removed and two empty masks score 0.

Equation (1) of the paper, `M_full = (N+/N) M_pos + (N0/N) M_empty`, holds exactly for every
convention and every run. `emc.conventions.identity_residuals` checks it.

### 5.3 Boundary geometry

Two boundary definitions occur among the released records, as in the paper's evaluation code
(`emc.records.GEOMETRIES`):

* `mask` (`polypgen_mixed`, `polypgen_ablation`, `siim`): HD95 takes distances from each surface to
  the other *mask*. Boundary IoU uses a 7 x 7 dilation-minus-erosion band.
* `surface` (`polypgen_c4`, `polypgen_archive`): HD95 takes surface-to-surface distances. Boundary IoU
  uses an inner band of width round(0.02 x diagonal) = 7 px.

Within a partition all models share one geometry, so rankings are unaffected. Recomputing Panel B
with the `mask` geometry changes its HD95 values by about 0.5 px and no rank.
`make_records --geometry` selects either one.

### 5.4 Scoring your own masks

`emc-score` writes the same convention-free records for your own predictions and applies every
convention to them. It needs only NumPy, SciPy, pandas and Pillow (installation: Section 2).

```bash
emc-score --pred my_predictions/ --ref my_references/ --json summary.json --records-out records.csv
```

Files are paired by name without the extension (`.png`, `.jpg`, `.bmp`, `.tif` or `.npy`). An 8-bit
image is divided by 255, so 0/255 masks and probability maps saved as images both work; predictions
are thresholded at `--threshold` (0.5), references at 0.5. The summary reports N, N+ and N0, q and r0
with the image-level sensitivity 1 - q and specificity 1 - r0, Dice and Boundary IoU on non-empty
references and with c = 0 and 1, and HD95 under X (deletion), X′, Z and P(d). The default d is the
image diagonal; `--penalty` sets others. `--geometry` and `--min-area` are those of Section 5.3 and
Appendix C.1, and `emc-score --records records.csv --penalty 362` scores one run of existing records.
From Python:

```python
from emc.score import score_pairs, summarise_records
records = score_pairs({"case1": (prediction, reference)})      # arrays or file paths
summary = summarise_records(records)                            # dict of every count, rate and convention
```

Inside this repository, `pip install -e .` makes the command available alongside the scripts.

## 6. Full pipeline

### 6.1 Data

* **PolypGen** (Ali et al., *Scientific Data* 10:75, 2023, doi:10.1038/s41597-023-01981-y): the
  single-frame release `PolypGen2021_MultiCenterData_v3` (1537 frames, centres C1-C6).
* **SIIM-ACR Pneumothorax Segmentation** (Kaggle, 2019): the DICOM images and `train-rle.csv`.

Neither dataset is redistributed here: the repository contains no image, mask or run-length
annotation, only frame identifiers and per-image counts derived from the annotations (split
manifests and records).

* PolypGen is released under CC BY 4.0 for research and educational use; users must cite Ali et al.
  (2023). The records and manifests here are derived from it (resized to 256 x 256; Section 6.1).
* SIIM-ACR is governed by the Kaggle competition rules, which permit use for academic research and
  prohibit redistributing the competition data. Obtain the images and `train-rle.csv` from Kaggle
  after accepting the rules.

`splits/` fixes every split at the frame level:

* `splits/polypgen_frames.csv`: frame, centre, source image and mask, mixed-partition split,
  centre-held-out split, and whether the released and the evaluated mask are empty.
* `splits/siim_split_manifest.csv`: ImageId, split, empty flag, lesion area at 1024 px.

```bash
python -m scripts.prepare_polypgen --raw PATH/PolypGen2021_MultiCenterData_v3
python -m scripts.prepare_siim --dicom-root PATH/siim-acr --rle PATH/train-rle.csv
```

These scripts write `datasets/polypgen_mixed`, `datasets/polypgen_c4`, `datasets/polypgen_archive`
and `datasets/siim` (256 x 256 PNG). Each checks its output against the manifests.
`prepare_polypgen` reproduces the evaluation pipeline of the paper exactly (pixel-identical),
including its two pipeline-created empty references (Appendix C.2):

* frame 00265.png, whose 2-pixel lesion vanishes on resizing;
* frame 00950.png, whose image name ends in a stray `]`, so that its mask was not paired.

Pass `--pair-by-name` to pair frame 00950.png with its released mask instead.

```bash
python -m scripts.check_data polypgen --raw PATH/PolypGen2021_MultiCenterData_v3   # Appendix C.2, Table C.1
python -m scripts.check_data siim --rle PATH/train-rle.csv                          # resize-erasure probe
```

### 6.2 Models

`emc/models/__init__.py` registers the eight architectures with the configuration used in every
benchmark:

| Key | Model | Parameters | Source |
|---|---|---|---|
| `prism` | PRISM (this study) | 40,401 active (44,465 registered) | `emc/models/prism.py` |
| `egeunet` | EGE-UNet | 53,374 | official release |
| `lbunet` | LB-UNet | 55,865 | official release |
| `malunet` | MALUNet | 177,943 | official release |
| `unext_s` | UNeXt-S | 253,561 | official release |
| `tinyunet` | UNet-Tiny (TinyU-Net) | 481,105 | official release |
| `unetpp` | UNet++ | 9,163,428 | pytorch-nested-unet |
| `mobileunet` | MobileUNet | 13,341,349 | lafith/Mobile-UNet, commit d32de4d |

`python -m scripts.count_parameters` prints these counts. PRISM's class keeps three dormant modules
of an earlier configuration so that the released checkpoints load strictly. The forward pass does
not use them (`emc.models.prism.dormant_parameters`). The ablation arms are variants of the same
class:

| Arm | Variant | Change |
|---|---|---|
| M | `prism_dn` | complete model |
| A1 | `ng_no_div` | no divisive modulation |
| A2 | `ng_sub` | subtractive control |
| A3 | `ng_no_hires` | no full-resolution stage |
| A4 | `ng_no_edge` | no image-gradient cue |

The variant `pgc_no_gain` is an earlier name of `prism_dn`.

### 6.3 Training

```bash
python -m scripts.lbunet_points --data datasets/polypgen_mixed --first-contour   # LB-UNet only
python -m scripts.lbunet_points --data datasets/siim
python -m scripts.train --model unetpp --protocol polypgen --seed 42
python -m scripts.train --model prism --variant ng_no_div --protocol polypgen --seed 7
python -m scripts.train --model lbunet --protocol siim --seed 1337
python -m scripts.train --model egeunet --protocol siim --seed 42 --epochs 30 --per-image-dice   # stress test, Appendix D
```

The protocols are defined in `emc/training/protocol.py`.

| | `polypgen`, `polypgen_c4` | `siim` |
|---|---|---|
| epochs (cosine T_max) | 300 | 111 |
| training stream | all images, shuffled | every lesion-bearing image + an equal draw of lesion-free images per epoch |
| augmentation | colour jitter, noise, h/v flip, one rotation angle per run, elastic, scale-crop | same, no vertical flip, rotation +-15 deg per image |
| mask resize | bilinear | nearest neighbour |
| Dice term | per image | over the batch |
| checkpoint selection | max(mIoU + Dice) per image on val | max Dice on lesion-bearing val images |

All protocols share AdamW (lr 1e-3, weight decay 0.03), batch size 8, cosine annealing to 1e-5 and
EMA weights (decay 0.999). The loss of each model is the one its authors released
(`emc/training/losses.py`). `train.py` writes `runs/<protocol>/<run>/best.pth` and `run.json`, which
holds the configuration and the validation curve.

### 6.4 Inference and records

```bash
python -m scripts.predict --model prism --checkpoint runs/polypgen/prism_seed42/best.pth \
    --protocol polypgen --data datasets/polypgen_mixed --split val --seed 42
python -m scripts.make_records --partition polypgen_mixed --data datasets/polypgen_mixed \
    --split val --pred-root predictions/polypgen_mixed
```

`predict` writes `predictions/<partition>/<Model>_seed<seed>/*.npy` (float32 probability maps),
`manifest.csv` and the checkpoint's SHA-256. `make_records` reads every run folder under
`--pred-root` and writes records in the released format. Every table script runs on regenerated
records unchanged. Use `--thresholds` and `--min-areas` for the robustness grid of Appendix C.1.

For the ablation partition, name the run folders by variant (`--out predictions/polypgen_ablation/prism_dn_seed7`, ...)
and pass `--partition polypgen_ablation`.

### 6.5 Provenance

`results/provenance/checkpoints.csv` lists, for each of the 136 evaluated runs:

* partition, model, seed, registry key, variant, protocol and model configuration;
* the selected epoch;
* the run folder and the SHA-256 of the checkpoint.

The LB-UNet checkpoints of the mixed partition did not store their epoch.

### 6.6 Library probe

```bash
python -m scripts.library_probe probe                    # every installed library -> outputs/library_probe/
pip install medpy==0.4.0                                 # in a second environment
python -m scripts.library_probe probe                    # there
python -m scripts.library_probe check --dir outputs/library_probe   # verify Table 3
python -m scripts.library_probe monai-predictions --pred-root predictions/polypgen_mixed --data datasets/polypgen_mixed
```

## 7. Released result files

| Path | Content |
|---|---|
| `results/ablation/per_seed.csv` | PRISM ablation, 5 arms x 8 seeds: full-set Dice, Dice on lesion-bearing frames, r0 (from the per-image evaluation), full-set B-IoU and HD95 as logged by the training-time evaluation (2 decimals); input of Table B.5 and Figure B.2 |
| `results/audit/` | sampling frame (Semantic Scholar, retrieved 2026-08-02), assisted first-pass coding, checked coding sheet with quotations |
| `results/library_probe/` | probe output per library version; MONAI on the stored predictions |
| `results/published/` | numbers transcribed from CANet (Table 3) and EndoKED (Hausdorff-distance gains) for Figure 7 |
| `results/stress_test/` | validation curves and configuration of the four stress-test runs (Figure D.1) |
| `results/data_checks/` | SIIM-ACR resize-erasure counts (Appendix C.2) |
| `results/provenance/checkpoints.csv` | checkpoint provenance (Section 6.5) |

## 8. Tests

```bash
python -m scripts.make_tables     # generate first
python -m scripts.audit
python -m scripts.bootstrap_reversals
python -m pytest tests
```

* `tests/test_paper.py`: every cell of Tables 1, 4, 5, B.1-B.6 and C.1 equals the typeset cell
  (`tests/paper_tables.json`), and every in-text number checked equals the printed one.
* `tests/test_records.py`: the records load and are internally consistent. The ablation records
  reproduce `results/ablation/per_seed.csv`. The Figure B.1 counts (52 failures, 32 with low boundary
  contrast) hold; this test is skipped without the prepared images.
* `tests/test_models.py`: PRISM parameter counts.
* `tests/test_score.py`: `emc-score` on the four empty-mask cases (hand-computed), on image folders
  and on the released records, where it matches the code behind the paper.
* `tests/test_pipeline.py`: every model trains for one epoch on synthetic data, predicts and yields
  records (CPU, about 4 minutes). Every script answers `--help`.

## 9. Reproducibility notes

* **What is exact.** Every table and number is computed from the released records, so it is exact.
  Records regenerated from the checkpoints are bit-identical to the released ones when inference runs
  on a CUDA GPU with PyTorch's default TF32 setting for convolutions, as for the paper. A CPU forward pass differs by up
  to 3e-3 in probability, which can flip isolated pixels at the threshold.
* **Ablation B-IoU and HD95.** Table B.5 uses the values logged during training (2 decimals).
  Recomputed from the checkpoints they agree to <= 0.01 percentage points (B-IoU) and <= 0.5 px
  (HD95). Using the recomputed values changes 10 of the 28 table cells in the second decimal. No
  interval changes whether it excludes zero.
* **Retraining** reproduces the runs statistically, not bit for bit. The PolypGen runs drew their
  single rotation angle before the seed was set, so the angle was not recorded. The SIIM-ACR runs used
  cuDNN autotuning. `train.py` seeds everything and runs cuDNN deterministically by default.
* **LB-UNet boundary points.** `lbunet_points` seeds its genetic search from the file name, so reruns
  are identical. The PolypGen points used in the paper were generated without a seed. The SIIM-ACR
  points of the paper used the same seeding; a sample of 12 maps regenerates identically.
* **MedPy under NumPy 2.** The released MedPy 0.5.2 probe ran under NumPy 1.26, where `dc` returns 1
  for two empty masks (Table 3). Under NumPy 2 the 0/0 in `dc` no longer raises the
  `ZeroDivisionError` that MedPy catches, and `dc` returns NaN (`jc` still returns 1). Probe MedPy in
  an environment with `numpy<2` to reproduce Table 3.
* **Fonts.** Some of the paper's PDFs were rendered with Liberation Serif (metric-compatible with
  Times New Roman). Regenerated figures match them in content; pixels differ slightly.

## 10. Third-party code

The baseline architectures in `emc/models/` are copied from their authors' public repositories. Each
file header names its source and lists any change:

* EGE-UNet and MALUNet: JCruan519
* LB-UNet: the authors' release
* UNeXt: jeya-maria-jose/UNeXt-pytorch
* TinyU-Net: ChenJunren-Lab/TinyU-Net
* UNet++: 4uiiurz1/pytorch-nested-unet
* Mobile-UNet: lafith/Mobile-UNet

They remain under their original licences. The LB-UNet boundary-point generator (`scripts/lbunet_points.py`)
follows the LB-UNet release.

## 11. Citation

```bibtex
@article{song2026emptymask,
  author  = {Song, Yurong and Chen, Zhewei and Wu, Jianbin},
  title   = {When empty predictions win: Empty-mask conventions alter model rankings and
             evaluated subsets in medical image segmentation},
  journal = {Computer Methods and Programs in Biomedicine},
  note    = {Submitted},
  year    = {2026}
}
```

The version used in the paper is archived on Zenodo as release v1.3,
[doi:10.5281/zenodo.23373345](https://doi.org/10.5281/zenodo.23373345). The concept DOI
[10.5281/zenodo.23296309](https://doi.org/10.5281/zenodo.23296309) always resolves to the latest version;
releases v1.0, v1.1 and v1.2 are [doi:10.5281/zenodo.23296310](https://doi.org/10.5281/zenodo.23296310),
[doi:10.5281/zenodo.23365844](https://doi.org/10.5281/zenodo.23365844) and
[doi:10.5281/zenodo.23366704](https://doi.org/10.5281/zenodo.23366704).

```bibtex
@software{song2026emptymaskcode,
  author    = {Song, Yurong and Chen, Zhewei and Wu, Jianbin},
  title     = {empty-mask-conventions: code and per-image records for
               ``When empty predictions win''},
  publisher = {Zenodo},
  version   = {v1.3},
  year      = {2026},
  doi       = {10.5281/zenodo.23373345},
  url       = {https://doi.org/10.5281/zenodo.23373345}
}
```

## License

MIT (see `LICENSE`), except for the third-party model files listed in Section 10, which keep the
licences of their original repositories.
