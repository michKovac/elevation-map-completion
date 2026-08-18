#!/usr/bin/env python3
"""
Download the TartanGround data the paper is built on.

Fetches the five environments in the `anymal` version with the four horizontal
depth cameras and the global semantic point cloud that provides the dense ground
truth. Roughly 130 GB unpacked.

    python scripts/download_tartanground.py --root /data/tartanground
    python scripts/download_tartanground.py --root /data/tartanground --env Gascola

The dataset is licensed CC BY 4.0; see docs/DATASET.md for the attribution that
must accompany any redistribution.

    Patel et al., "TartanGround: A Large-Scale Dataset for Ground Robot
    Perception and Navigation", IROS 2025.
    https://tartanair.org/tartanground/
"""
import argparse

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
MODALITIES = ["meta", "depth", "imu", "sem_pcd"]
CAMERAS = ["lcam_front", "lcam_left", "lcam_right", "lcam_back"]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--root', required=True, help='download destination')
    p.add_argument('--env', nargs='+', default=None,
                   help=f'environments (default: all five — {", ".join(ENVIRONMENT_TRAJECTORIES)})')
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--keep_zip', action='store_true', help='keep archives after unpacking')
    return p.parse_args()


def main():
    args = parse_args()
    try:
        import tartanair as ta
    except ImportError:
        raise SystemExit('The tartanair package is required: pip install -e ".[data]"')

    envs = args.env or list(ENVIRONMENT_TRAJECTORIES)
    unknown = [e for e in envs if e not in ENVIRONMENT_TRAJECTORIES]
    if unknown:
        raise SystemExit(f'Unknown environment(s): {", ".join(unknown)}')

    ta.init(args.root)
    for env in envs:
        trajs = ENVIRONMENT_TRAJECTORIES[env]
        print(f'[{env}] downloading {len(trajs)} trajectories: {", ".join(trajs)}', flush=True)
        ta.download_ground(
            env=[env],
            version=['anymal'],
            traj=trajs,
            modality=MODALITIES,
            camera_name=CAMERAS,
            unzip=True,
            delete_zip=not args.keep_zip,
            num_workers=args.num_workers,
            data_source='huggingface',
        )

    print(f'\nDone. Next: python scripts/build_dataset.py --tartanground_root {args.root}')


if __name__ == '__main__':
    main()
