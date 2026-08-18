#!/usr/bin/env python3
"""
Build the elevation completion dataset from downloaded TartanGround data.

For every frame of every trajectory this writes one .npz holding the partial
elevation map accumulated from the four depth cameras and the dense ground-truth
elevation cropped from the environment's global point cloud.

The defaults reproduce the dataset used in the paper: a 50 x 50 m robot-centric
window at 0.2 m/cell (251 x 251), per-cell 20th percentile of point heights.

    # download the source data (~130 GB) and build everything
    python scripts/build_dataset.py --tartanground_root /data/tartanground --download

    # build one environment from data already on disk
    python scripts/build_dataset.py --tartanground_root /data/tartanground \
        --env OldTownSummer --out datasets/elevation_dataset

Expected input layout:

    <tartanground_root>/<Env>/<Env>_sem.pcd
    <tartanground_root>/<Env>/Data_anymal/<Traj>/depth_lcam_{front,left,right,back}/
    <tartanground_root>/<Env>/Data_anymal/<Traj>/pose_lcam_{front,left,right,back}.txt
"""
import argparse
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

from elevcomp.data.depth import CAMERAS, accumulate_frame, depth_files, load_camera_poses
from elevcomp.data.groundtruth import extract_gt_elevation, load_global_cloud
from elevcomp.data.io import save_sample
from elevcomp.data.raster import DEFAULT_PERCENTILE, apply_axis_mapping, rasterize_elevation
from elevcomp.paths import data_root

# Trajectories differ per environment: ForestEnv is numbered from P2001.
ENVIRONMENT_TRAJECTORIES = {
    "ForestEnv":           ["P2001", "P2002", "P2003", "P2004", "P2005"],
    "Gascola":             ["P2000", "P2001", "P2002", "P2003", "P2004"],
    "ModularNeighborhood": ["P2000", "P2001", "P2002", "P2003", "P2004"],
    "OldTownSummer":       ["P2000", "P2001", "P2002", "P2003", "P2004"],
    "SeasonalForestWinter":["P2000", "P2001", "P2002", "P2003", "P2004"],
}

# sem_pcd carries the global cloud used as ground truth; meta and imu come with
# the poses the accumulation needs.
DOWNLOAD_MODALITIES = ["meta", "depth", "imu", "sem_pcd"]
DOWNLOAD_CAMERAS = ["lcam_front", "lcam_left", "lcam_right", "lcam_back"]

PAPER_ENVIRONMENTS = list(ENVIRONMENT_TRAJECTORIES)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--tartanground_root', required=True,
                   help='directory holding (or to receive) the TartanGround environments')
    p.add_argument('--download', action='store_true',
                   help='fetch the source data first (~130 GB); needs the tartanair package')
    p.add_argument('--env', nargs='+', default=['all'],
                   help=f"environments to build, or 'all' for {', '.join(PAPER_ENVIRONMENTS)}")
    p.add_argument('--trajs', nargs='+', default=None,
                   help='trajectories (default: every subdirectory of Data_anymal)')
    p.add_argument('--out', default=None,
                   help='output root (default: $ELEVCOMP_DATA_ROOT or datasets/elevation_dataset)')

    g = p.add_argument_group('sample geometry (defaults reproduce the paper)')
    g.add_argument('--half_extent', type=float, default=25.0,
                   help='half size of the window in metres (default: 25 -> 50x50 m)')
    g.add_argument('--resolution', type=float, default=0.2, help='metres per cell')
    g.add_argument('--percentile', type=float, default=DEFAULT_PERCENTILE,
                   help='per-cell percentile of point heights')
    g.add_argument('--axis_mapping', default='yxz_neg',
                   help='camera-to-grid axis mapping (alias or spec like "y,x,-z")')

    p.add_argument('--limit', type=int, default=0,
                   help='build at most N frames per trajectory (0 = all)')
    p.add_argument('--overwrite', action='store_true',
                   help='rebuild samples that already exist')
    return p.parse_args()


def download(root: str, envs: list, num_workers: int = 4) -> None:
    """Fetch the source data with the official toolkit."""
    try:
        import tartanair as ta
    except ImportError:
        raise SystemExit('--download needs the tartanair package: pip install -e ".[data]"')

    ta.init(root)
    for env in envs:
        trajs = ENVIRONMENT_TRAJECTORIES[env]
        print(f'[{env}] downloading {len(trajs)} trajectories: {", ".join(trajs)}', flush=True)
        ta.download_ground(env=[env], version=['anymal'], traj=trajs,
                           modality=DOWNLOAD_MODALITIES, camera_name=DOWNLOAD_CAMERAS,
                           unzip=True, delete_zip=True, num_workers=num_workers,
                           data_source='huggingface')


def trajectory_dirs(env_dir: Path, requested) -> list:
    data_dir = env_dir / 'Data_anymal'
    if not data_dir.exists():
        raise SystemExit(f'{data_dir} not found — download the anymal version of {env_dir.name}')
    if requested:
        dirs = [data_dir / t for t in requested]
        missing = [d.name for d in dirs if not d.exists()]
        if missing:
            raise SystemExit(f'Trajectories not found in {data_dir}: {", ".join(missing)}')
        return dirs
    return [p for p in sorted(data_dir.iterdir()) if p.is_dir()]


def build_trajectory(traj_path: Path, out_dir: Path, global_points, global_tree,
                     bounds, args) -> int:
    """Build every frame of one trajectory. Returns the number of samples written."""
    poses_by_cam = load_camera_poses(traj_path)
    files_by_cam = {cam: depth_files(traj_path, cam) for cam in CAMERAS}

    n_frames = min(len(f) for f in files_by_cam.values())
    n_frames = min(n_frames, len(poses_by_cam['front']))
    if args.limit:
        n_frames = min(n_frames, args.limit)

    written = 0
    for frame_idx in tqdm(range(n_frames), desc=traj_path.name, unit='frame', leave=False):
        sample_name = f'{traj_path.name}_sample_{frame_idx:06d}'
        if not args.overwrite and (out_dir / f'{sample_name}.npz').exists():
            continue

        points, _ = accumulate_frame(traj_path, frame_idx, poses_by_cam,
                                     files_by_cam, bounds)
        points = apply_axis_mapping(points, args.axis_mapping)
        partial_elev, partial_mask = rasterize_elevation(
            points, bounds, args.resolution, percentile=args.percentile)

        gt_elev, gt_mask = extract_gt_elevation(
            global_points, global_tree, poses_by_cam['front'][frame_idx],
            bounds, args.resolution, percentile=args.percentile,
            axis_mapping=args.axis_mapping)

        save_sample(out_dir, sample_name, partial_elev, partial_mask, gt_elev, gt_mask,
                    frame_idx=frame_idx, reference_idx=frame_idx,
                    bounds=bounds, resolution=args.resolution)
        written += 1

    return written


def main():
    args = parse_args()
    root = Path(args.tartanground_root).expanduser().resolve()
    out_root = Path(args.out).expanduser() if args.out else data_root()

    envs = PAPER_ENVIRONMENTS if args.env == ['all'] else args.env
    unknown = [e for e in envs if e not in ENVIRONMENT_TRAJECTORIES]
    if unknown:
        raise SystemExit(f'Unknown environment(s): {", ".join(unknown)}')

    if args.download:
        download(str(root), envs)

    h = args.half_extent
    bounds = np.array([-h, h, -h, h, -h, h], dtype=np.float32)

    total = 0
    for env in envs:
        env_dir = root / env
        pcd_path = env_dir / f'{env}_sem.pcd'
        if not pcd_path.exists():
            raise SystemExit(f'{pcd_path} not found — the sem_pcd modality is required')

        print(f'[{env}] loading global point cloud and building KD-tree...', flush=True)
        global_points, global_tree = load_global_cloud(pcd_path)
        print(f'[{env}] {len(global_points):,} points', flush=True)

        for traj_path in trajectory_dirs(env_dir, args.trajs):
            written = build_trajectory(traj_path, out_root / env / traj_path.name,
                                       global_points, global_tree, bounds, args)
            total += written
            print(f'[{env}/{traj_path.name}] {written} samples', flush=True)

    print(f'\nWrote {total} samples to {out_root}')


if __name__ == '__main__':
    sys.exit(main())
