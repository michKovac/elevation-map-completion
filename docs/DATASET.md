# Dataset

The elevation completion dataset is derived from **TartanGround**, which we do not
redistribute. This document describes how to obtain the source data and regenerate our
dataset exactly.

## Source

> Patel, M.; Yang, F.; Qiu, Y.; Cadena, C.; Scherer, S.; Hutter, M.; Wang, W.
> *TartanGround: A Large-Scale Dataset for Ground Robot Perception and Navigation.*
> IROS 2025, pp. 20524–20531. doi:[10.1109/IROS60139.2025.11246002](https://doi.org/10.1109/IROS60139.2025.11246002)
> <https://tartanair.org/tartanground/>

TartanGround is licensed **CC BY 4.0** and its toolkit is MIT. Derivatives — including
the dataset produced here and the sample files in `examples/sample/` — may be
redistributed provided the citation above is carried along. See `LICENSE-DATA`.

## 1. Download

```bash
pip install -e ".[data]"
python scripts/build_dataset.py --tartanground_root /data/tartanground --download
```

`--download` fetches the source data and then builds our dataset in one go. To do the
two steps separately, drop the flag once the data is already on disk.

This fetches five environments in the `anymal` version, modalities `meta`, `depth`,
`imu`, `sem_pcd`, and the four horizontal cameras `lcam_{front,left,right,back}`.
Around 130 GB unpacked.

| Environment | Terrain | Trajectories |
|---|---|---|
| ForestEnv | dense forest | P2001–P2005 |
| Gascola | rural terrain with vegetation | P2000–P2004 |
| ModularNeighborhood | suburban | P2000–P2004 |
| OldTownSummer | dense historic urban | P2000–P2004 |
| SeasonalForestWinter | forest, snow cover | P2000–P2004 |

ForestEnv is numbered from P2001 — `scripts/build_dataset.py --download` handles this.

The expected layout afterwards:

```
/data/tartanground/<Env>/<Env>_sem.pcd
/data/tartanground/<Env>/Data_anymal/<Traj>/depth_lcam_{front,left,right,back}/
/data/tartanground/<Env>/Data_anymal/<Traj>/pose_lcam_{front,left,right,back}.txt
```

## 2. Generate

```bash
python scripts/build_dataset.py \
    --tartanground_root /data/tartanground \
    --out datasets/elevation_dataset
```

(Already done if you passed `--download` above.)

Roughly 6.5 s per sample, dominated by the KD-tree query against the global cloud, so
the full build takes about two days single-threaded. Trajectories are independent —
run several processes with `--env` / `--trajs` to parallelise. The script skips samples
that already exist, so an interrupted run resumes; `--overwrite` forces a rebuild.

Result: **32 329 samples**, about 10 GB.

| Environment | Samples | Mean coverage |
|---|---|---|
| ForestEnv | 6 466 | 0.458 |
| Gascola | 6 467 | 0.417 |
| ModularNeighborhood | 6 463 | — |
| OldTownSummer | 6 467 | 0.28 |
| SeasonalForestWinter | 6 466 | — |

Point the training code at the result:

```bash
export ELEVCOMP_DATA_ROOT=$PWD/datasets/elevation_dataset
```

## How a sample is built

Each frame yields one sample covering a 50 × 50 m window centred on the robot.

**Partial input.** The four depth images of the frame are unprojected with the dataset
intrinsics (fx = fy = 320, cx = cy = 320, 640 × 640), keeping depths in 0.1–100 m. Each
cloud is transformed into the frame of the front camera using the per-camera poses, and
the four are merged. This is a single-frame observation: no temporal accumulation, so
the holes are exactly what the robot sees at that instant.

**Ground truth.** The environment's global semantic point cloud (`{env}_sem.pcd`, tens
of millions of points) is queried with a KD-tree around the robot pose and expressed in
the same local frame. It is dense and unoccluded, which is what makes it usable as a
completion target.

**Rasterisation.** Both clouds are mapped from camera axes to grid axes (`yxz_neg`,
i.e. `y, x, -z`) and binned onto a 251 × 251 grid at 0.2 m/cell. A cell's elevation is
the **20th percentile** of the heights inside it — not the minimum, which latches onto
stray points below the surface, and not the mean, which floats up into vegetation.

| Parameter | Value | Flag |
|---|---|---|
| Window | ±25 m (50 × 50 m) | `--half_extent` |
| Resolution | 0.2 m/cell → 251 × 251 | `--resolution` |
| Percentile | 20 | `--percentile` |
| Axis mapping | `yxz_neg` | `--axis_mapping` |
| Depth range | 0.1–100 m | `elevcomp/data/depth.py` |

### A note on the composite

Observed cells carry a systematic offset against the dense ground truth, because a
single-frame percentile and a percentile over the accumulated cloud are not the same
statistic — most visibly under vegetation. This is why a composite of measured and
predicted cells scores a *higher* RMSE (3.717 m) than the pure prediction (2.994 m).
It is a property of the data, not a bug in the composition, and is discussed in the
paper.

## Sample format

One compressed `.npz` per frame, named ``<Traj>_sample_<frame:06d>.npz``:

| Key | Shape | Type | Meaning |
|---|---|---|---|
| `partial_elevation` | 251 × 251 | float32 | observed elevation, NaN where unobserved |
| `partial_mask` | 251 × 251 | uint8 | 1 = observed |
| `gt_elevation` | 251 × 251 | float32 | dense reference elevation |
| `gt_mask` | 251 × 251 | uint8 | 1 = valid ground truth |
| `frame_idx`, `reference_idx` | scalar | int32 | source frame indices |
| `bounds` | 6 | float32 | (x_min, x_max, y_min, y_max, z_min, z_max) |
| `resolution` | scalar | float32 | metres per cell |

Read one with `elevcomp.data.io.load_sample`.

## Reproducibility

`scripts/build_dataset.py` reproduces the published dataset **bit-exactly** — verified
by regenerating samples from OldTownSummer, Gascola and ForestEnv and comparing every
array against the files used for the paper.

One detail matters for that. The depth path normalises the pose quaternion in float32
before handing it to SciPy, the ground-truth path does not, and SciPy normalises in
float64 either way. The two differ by about 1e-7 in the rotation matrix, which reaches
at most a few millimetres in the elevation grid. `pose_to_transform` therefore takes an
explicit `normalize_quaternion` flag and each path pins its own value. The difference is
four orders of magnitude below the error the model is measured at; the flag exists for
exact reproduction, not because either choice is meaningfully better.
