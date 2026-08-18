# Elevation Map Completion

Robot-centric elevation map completion with sensor-geometry-aware augmentation and
heteroscedastic uncertainty estimation.

Elevation maps built from onboard depth sensing are full of holes: occlusions, grazing
incidence and limited field of view leave 55–75 % of a 50 × 50 m robot-centric window
unobserved. This repository trains a U-Net to fill those holes, predicts a per-cell
uncertainty in the same forward pass, and turns both into a slope-based traversability
decision that a planner can consume.

Reference implementation for:

> J. Goga, M. Kovac, M. Dekan, J. Pavlovicova, F. Duchon.
> **Robot-Centric Elevation Map Completion with Sensor-Geometry-Aware Augmentation and
> Uncertainty Estimation.** *Applied Sciences*, MDPI, 2026.

---

## Repository status

Code, configuration and licensing are in place and the training/evaluation entry points
run. Still landing (see `docs/` once present):

- `scripts/build_dataset.py` and `scripts/download_tartanground.py` — dataset generation
  is currently documented below but not yet scripted
- `examples/end_to_end.py`, `examples/sample/*.npz`
- `docs/DATASET.md`, `docs/REPRODUCE.md`, `docs/EXPERIMENTS.md`
- released model weights and `results/`

---

## What is in the pipeline

| # | Stage | Entry point |
|---|---|---|
| 1 | Build our dataset from TartanGround | `scripts/build_dataset.py` *(pending)* |
| 2 | Preprocessing, normalisation, folds | `elevcomp/dataset.py`, `elevcomp/folds.py` |
| 3 | Training and inference (ResNet-34 U-Net) | `scripts/train_cv.py`, `elevcomp/inference.py` |
| 4 | Ray-cone augmentation | `elevcomp/dataset.py` |
| 5 | Uncertainty estimation (β-NLL, D4 TTA) | `scripts/eval/eval_uncertainty.py` |
| 6 | Traversability prediction | `scripts/eval/traversability_eval.py` |
| 7 | End-to-end example | `examples/end_to_end.py` *(pending)* |

---

## Method

**Input** is a two-channel 256 × 256 tensor: the partial elevation map (251 × 251, centred
and padded) and its observation mask. **Output** is the completed elevation and, from the
same forward pass, a per-cell log-variance.

**Ray-cone augmentation** is the main methodological contribution. During training, 1–3
angular sectors of width 5–25° are removed from the input around the sensor origin, which
imitates a camera dropping out or an occluder blocking a direction. Unlike random
rectangular masking, the holes it creates share the geometry of real sensor failures. It
is applied on the raw arrays before normalisation and padding so the physical sensor
origin stays at the image centre.

**Uncertainty** uses a β-NLL head (Seitzer et al., ICLR 2022) with β = 0.5, which avoids
the degenerate high-variance solution of plain Gaussian NLL. Training runs 100 epochs of
masked L1 before the NLL term is switched on. A D4 test-time-augmentation ensemble is
implemented as an eight-pass alternative for comparison.

**Loss** is masked and two-term: `w_valid · L_valid + w_hole · L_hole` with
`(w_valid, w_hole) = (0.1, 1.0)`, so the holes — the cells the model actually has to
invent — dominate the gradient.

---

## Results

Five-fold leave-one-environment-out cross-validation. Mean ± population std over folds,
elevation in metres.

**Completion.** Hole RMSE **2.855 ± 0.819 m**, overall RMSE 2.994 ± 0.939 m — a **45 %
improvement** over the best classical baseline.

| Method | Hole RMSE [m] |
|---|---|
| nearest | 5.549 ± 1.986 |
| linear | 5.191 ± 1.888 |
| IDW | 5.322 ± 1.956 |
| Telea | 5.258 ± 1.915 |
| Navier–Stokes | 5.251 ± 1.932 |
| **U-Net (ResNet-34), ours** | **2.855 ± 0.819** |

**Architecture does not decide.** Four models land within 2.85–2.96 m hole RMSE, and a
trajectory-level paired Wilcoxon test (n = 25) finds no significant difference after Holm
correction (min. adjusted p = 0.165).

| Model | Family | RMSE | Hole RMSE | SSIM |
|---|---|---|---|---|
| U-Net (custom, 11.0 M) | CNN | 3.074 ± 1.049 | 2.934 ± 0.967 | 0.357 ± 0.087 |
| U-Net (ResNet-34, 24.4 M) | CNN | **2.994 ± 0.939** | 2.855 ± 0.819 | **0.393 ± 0.059** |
| Attention U-Net (24.5 M) | CNN | 3.004 ± 0.956 | **2.851 ± 0.830** | **0.393 ± 0.047** |
| SegFormer MiT-B2 (24.7 M) | Transformer | 3.136 ± 0.942 | 2.959 ± 0.831 | 0.364 ± 0.093 |

**Ray-cone augmentation buys robustness, not accuracy.** On clean input the three
augmentation settings are indistinguishable (2.838–2.855 m). When a sector is removed at
test time, the ray-cone model reduces RMSE on the removed cells by **8.3 / 9.1 / 9.7 /
9.1 %** at 30 / 45 / 60 / 90° and wins in **5 of 5 folds** at every width.

**Uncertainty.** The β-NLL head correlates with the actual error at r = 0.455, reaches
coverage@1σ of 0.549 and AUSE 0.417 — better calibrated than the D4 TTA ensemble (0.419)
at **one eighth of the cost**. A post-hoc scale γ = 1.18 ± 0.06 fitted on validation
raises coverage from 0.55 to 0.61.

**Traversability** (25° slope threshold, hole cells only). Without a gate, accuracy is
0.637 ± 0.030 at a false-safe rate of 0.431 ± 0.067. Gating on predicted σ trades coverage
for safety: σ̂ ≤ 1 m gives a false-safe rate of 0.292 over 57.6 % of the area, σ̂ ≤ 0.3 m
gives 0.134 over 32.1 %. Note that 71.6 ± 8.6 % of evaluated hole cells are traversable in
the ground truth, so accuracy sits below the majority-class rate — the false-safe rate is
the metric that matters, and a trivial always-safe classifier would score 1.0 on it.

**Cost** (RTX 5090, FP32, batch 1, input 2 × 256 × 256):

| Mode | Params | Passes | Latency | Peak memory |
|---|---|---|---|---|
| Single-pass β-NLL | 24.4 M | 1 | 3.6 ms | 410 MB |
| D4 TTA ensemble | 24.4 M | 8 | 28.5 ms | 413 MB |

---

## Installation

```bash
git clone https://github.com/michKovac/elevation-map-completion.git
cd elevation-map-completion

python -m venv .venv && source .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -e .

# only needed to regenerate the dataset from TartanGround
pip install -e ".[data]"
```

Exact versions behind the reported numbers are in `requirements-lock.txt`
(PyTorch 2.11 + CUDA 12.8, Linux). A conda alternative is in `environment.yml`.

---

## Data

The source data is the **TartanGround** dataset (CC BY 4.0), which we do not
redistribute. Our derived dataset is generated from it with the parameters below.

Download five environments, `anymal` version, trajectories P2000–P2004
(**ForestEnv uses P2001–P2005**), modalities `meta, depth, imu, sem_pcd`, cameras
`lcam_{front,left,right,back}`:

```python
import tartanair as ta
ta.init('<data-root>')
ta.download_ground(
    env=['ForestEnv', 'Gascola', 'ModularNeighborhood',
         'OldTownSummer', 'SeasonalForestWinter'],
    version=['anymal'],
    traj=['P2000', 'P2001', 'P2002', 'P2003', 'P2004'],
    modality=['meta', 'depth', 'imu', 'sem_pcd'],
    camera_name=['lcam_front', 'lcam_left', 'lcam_right', 'lcam_back'],
    unzip=True, delete_zip=True, data_source='huggingface')
```

Each sample is then built as follows.

| Parameter | Value |
|---|---|
| Partial input | depth of 4 cameras → point cloud → per-cell 20th percentile |
| Ground truth | global `{env}_sem.pcd` cloud via KD-tree → per-cell 20th percentile |
| Bounds | ±25 m (50 × 50 m window), 0.2 m/cell → 251 × 251 |
| Axis mapping | `yxz_neg` |
| Depth filter | 0.1–100 m; intrinsics fx = fy = 320, cx = cy = 320 |

This yields **32 329 samples** (5 environments × 5 trajectories, ≈ 10 GB) stored as `.npz`
with keys `partial_elevation`, `partial_mask`, `gt_elevation`, `gt_mask`, `frame_idx`,
`reference_idx`, `bounds`, `resolution`.

Point the code at the result with `ELEVCOMP_DATA_ROOT`, or pass `--data_root`.

---

## Reproducing the experiments

```bash
export ELEVCOMP_DATA_ROOT=/path/to/elevation_dataset

# main experiment — 5-fold leave-one-environment-out (Tables 4-13, 16-18)
python scripts/train_cv.py --name resnet34

# classical baselines on the same folds (CPU, any time after the experiment exists)
python scripts/run_baselines.py --exp_dir runs/cv_resnet34_<timestamp>

# augmentation and loss ablations
python scripts/run_ablations.py
python scripts/eval/compare_experiments.py runs/cv_abl_* --out reports/ablations

# architecture comparison
python scripts/train_cv.py --name attunet   --model attunet
python scripts/train_cv.py --name segformer --model segformer
python scripts/eval/wilcoxon_architectures.py

# uncertainty, robustness, traversability, cost
python scripts/eval/eval_uncertainty.py    --checkpoint <fold>/best.pth
python scripts/eval/temperature_scaling.py
python scripts/eval/robustness_eval.py
python scripts/eval/traversability_eval.py --tau_sweep
python scripts/eval/benchmark_inference.py --checkpoint <fold>/best.pth
python scripts/eval/latency_e2e.py
```

Evaluation tools default to the newest `runs/cv_*` directory; override with `--exp_dir`
or `ELEVCOMP_EXPERIMENT`. A two-epoch smoke test of the whole pipeline:

```bash
python scripts/train_cv.py --name smoke --debug 12 --epochs 2
```

Training takes roughly one day per fold on a single modern GPU at the paper settings
(200 epochs, batch 64). Folds are independent, so they can be split across GPUs with
`--folds`.

**Determinism note.** Tools that re-run trained folds and compare quantities near a
threshold (for example the σ > |e| coverage counter) must set
`torch.backends.cudnn.benchmark = True`, matching `elevcomp/cv.py`. A different kernel
choice shifts coverage by up to 2e-4. RMSE-style metrics are unaffected.

---

## Layout

```
elevcomp/           library — imported, not executed
  dataset.py          ElevationDataset, augmentations (incl. ray-cone)
  model.py            PConv-UNet / SimpleUNet / AttentionUNet / ResNet-34 U-Net / SegFormer
  losses.py           masked L1 and beta-NLL
  inference.py        single-pass beta-NLL and 8x D4 TTA
  metrics.py          MAE / RMSE / AbsRel / masked SSIM, in metres
  calibration.py      sparsification, AUSE, coverage@1-sigma
  folds.py            environment discovery and fold construction
  baselines.py        classical hole filling
  cv.py               cross-validation driver
  paths.py            ELEVCOMP_ROOT / _DATA_ROOT / _EXPERIMENT resolution
configs/            cv_resnet34.toml (paper settings), single_run.toml, ablations.toml
scripts/            training entry points; eval/ holds the analysis tools
examples/           end-to-end demo and sample data
```

---

## Citation

If you use this code or the released weights, please cite the paper (see `CITATION.cff`)
and the source dataset:

```bibtex
@InProceedings{patel2025tartanground,
  author    = {Patel, Manthan and Yang, Fan and Qiu, Yuheng and Cadena, Cesar
               and Scherer, Sebastian and Hutter, Marco and Wang, Wenshan},
  title     = {{TartanGround}: A Large-Scale Dataset for Ground Robot Perception
               and Navigation},
  booktitle = {2025 IEEE/RSJ International Conference on Intelligent Robots and
               Systems (IROS)},
  year      = {2025},
  pages     = {20524--20531},
  doi       = {10.1109/IROS60139.2025.11246002}
}
```

## Acknowledgements

The implementation builds on published work and open-source code, attributed in the module
headers: partial convolutions (Liu et al., ECCV 2018; implementation by naoto0804), U-Net
(Ronneberger et al., MICCAI 2015), ResNet (He et al., CVPR 2016), scSE attention (Roy et
al., MICCAI 2018), heteroscedastic uncertainty (Kendall & Gal, NeurIPS 2017), β-NLL
(Seitzer et al., ICLR 2022), D4 self-ensembling (Lim et al., CVPRW 2017), TTA as
uncertainty (Shanmugam et al., ICCV 2021), and
[segmentation_models_pytorch](https://github.com/qubvel/segmentation-models-pytorch).

## License

Code is MIT (`LICENSE`). Sample data, released weights and documentation are CC BY 4.0
(`LICENSE-DATA`), as derivatives of the CC BY 4.0 TartanGround dataset.

Funded by the EU NextGenerationEU through the Recovery and Resilience Plan for Slovakia
under project No. 09I05-03-V02-00039.
