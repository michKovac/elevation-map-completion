#!/usr/bin/env python3
"""
Cross-environment 5-fold experiment for elevation map completion.

Fold design (leave-one-environment-out):
    fold k  →  TEST  = all trajectories of environment k (never seen in training)
               VAL   = last (sorted) trajectory of each remaining environment
               TRAIN = all other trajectories of the remaining environments

Holding out a whole environment, rather than random samples, is what makes the
reported error a measure of generalisation to unseen terrain. Each experiment
directory keeps everything needed to reproduce its numbers, including an
auto-generated README with a snippet for loading the predictions.

Usage:
    # new experiment (trains all 5 folds sequentially, then evaluates + aggregates)
    python scripts/train.py --name resnet34

    # run a subset of folds, e.g. in parallel across two GPUs
    CUDA_VISIBLE_DEVICES=0 python scripts/train.py --exp_dir runs/cv_resnet34_x --folds 0 1 2
    CUDA_VISIBLE_DEVICES=1 python scripts/train.py --exp_dir runs/cv_resnet34_x --folds 3 4

    # re-run only evaluation / only the cross-fold summary
    python scripts/train.py --exp_dir runs/cv_resnet34_x --stage eval
    python scripts/train.py --exp_dir runs/cv_resnet34_x --stage aggregate

    # fast smoke test of the whole pipeline (12 files per trajectory)
    python scripts/train.py --name smoke --debug 12 --epochs 2

Hyper-parameters come from configs/default.toml. CLI overrides (--epochs,
--aug_ray_p, ...) apply only when creating a new experiment — afterwards the
config is frozen in experiment.json. Interrupted folds resume from last.pth.
"""
import argparse
import json
import sys
from pathlib import Path

from elevcomp.cv import HPARAM_OVERRIDES, create_experiment, run_experiment


def _bool(s: str) -> bool:
    if s.lower() in ('1', 'true', 'yes', 'on'):
        return True
    if s.lower() in ('0', 'false', 'no', 'off'):
        return False
    raise argparse.ArgumentTypeError(f'expected true/false, got {s!r}')


def parse_cli():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[1],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--name', help='experiment name (creates a new experiment)')
    p.add_argument('--exp_dir', help='existing experiment directory (continue)')
    p.add_argument('--data_root',
                   default=str(Path(__file__).parent / '../datasets/elevation_dataset'),
                   help='dataset root containing the environment directories')
    p.add_argument('--folds', type=int, nargs='+',
                   help='subset of folds to run (default: all)')
    p.add_argument('--stage', default='all',
                   choices=['all', 'train', 'eval', 'aggregate'])
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--val_traj_per_env', type=int, default=1)
    p.add_argument('--debug', type=int, default=0, metavar='N',
                   help='use only N files per trajectory (smoke test, marks _DEBUG)')
    p.add_argument('--num_workers', type=int)

    # Hyper-parameter overrides — allowed only for NEW experiments.
    hp = p.add_argument_group('config overrides (new experiments only)')
    hp.add_argument('--epochs', type=int)
    hp.add_argument('--batch_size', type=int)
    hp.add_argument('--lr', type=float)
    hp.add_argument('--weight_decay', type=float)
    hp.add_argument('--grad_clip', type=float)
    hp.add_argument('--layer_size', type=int)
    hp.add_argument('--base_channels', type=int)
    hp.add_argument('--w_valid', type=float)
    hp.add_argument('--w_hole', type=float)
    hp.add_argument('--uncertainty', type=_bool, metavar='true|false')
    hp.add_argument('--uncertainty_beta', type=float)
    hp.add_argument('--uncertainty_warmup', type=int)
    hp.add_argument('--aug_hflip_p', type=float)
    hp.add_argument('--aug_vflip_p', type=float)
    hp.add_argument('--aug_rot90_p', type=float)
    hp.add_argument('--aug_ray_p', type=float)
    hp.add_argument('--aug_ray_n_max', type=int)
    hp.add_argument('--aug_ray_angle_deg', type=float, nargs=2,
                    metavar=('MIN', 'MAX'))
    return p.parse_args()


def main():
    args = parse_cli()
    if bool(args.name) == bool(args.exp_dir):
        sys.exit('Provide exactly one of --name (new) or --exp_dir (continue).')

    overrides = {k: getattr(args, k) for k in HPARAM_OVERRIDES
                 if getattr(args, k) is not None}

    if args.exp_dir:
        exp_dir = Path(args.exp_dir)
        if not (exp_dir / 'experiment.json').exists():
            sys.exit(f'{exp_dir} is not an experiment directory (no experiment.json).')
        if overrides:
            sys.exit(f'Hyper-parameter overrides {sorted(overrides)} are only '
                     f'allowed for new experiments — this one is frozen in '
                     f'experiment.json.')
    else:
        exp_dir = create_experiment(
            name=args.name, overrides=overrides, data_root=args.data_root,
            seed=args.seed, val_traj_per_env=args.val_traj_per_env,
            debug_limit=args.debug, command=' '.join(sys.argv))

    run_experiment(exp_dir, folds_sel=args.folds, stage=args.stage,
                   num_workers=args.num_workers)


if __name__ == '__main__':
    main()
