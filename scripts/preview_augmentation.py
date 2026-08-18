#!/usr/bin/env python3
"""
Preview the ray-cone augmentation on real samples: original coverage next to
1..n_max removed sectors of increasing width.

The sectors are produced by elevcomp.dataset._apply_ray_augmentation, i.e. the
same code path that runs during training, so the figure cannot drift away from
the augmentation it documents.

    python scripts/eval/figures/preview_ray_augmentation.py
    python scripts/eval/figures/preview_ray_augmentation.py --env Gascola --n_samples 4
"""
import argparse
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from elevcomp import paths
from elevcomp.dataset import _apply_ray_augmentation
from elevcomp.paths import reports_dir

# (label, n_max, width in degrees); None = untouched input.
# n_max is an upper bound: the augmentation draws 1..n_max sectors per sample,
# which is why the two right-hand columns do not always show the full count.
SETTINGS = [
    ('Original',        None, None),
    ('1 sector, 15°',      1,   15),
    ('1 sector, 25°',      1,   25),
    ('1–2 sectors, 25°',   2,   25),
    ('1–3 sectors, 25°',   3,   25),
]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--data_root', default=None,
                   help='dataset root (default: $ELEVCOMP_DATA_ROOT or datasets/elevation_dataset)')
    p.add_argument('--env', default='OldTownSummer', help='environment to sample from')
    p.add_argument('--n_samples', type=int, default=3, help='rows in the figure')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--out', default=None,
                   help='output PNG (default: reports/preview_ray_augmentation.png)')
    return p.parse_args()


def load_samples(base: Path, n: int, seed: int) -> list:
    files = sorted(base.glob('**/*.npz'))
    if len(files) < n:
        raise SystemExit(f'Found {len(files)} samples in {base}, need {n}. '
                         'Generate the dataset first — see docs/DATASET.md.')
    chosen = np.random.default_rng(seed).choice(len(files), size=n, replace=False)
    samples = []
    for idx in chosen:
        d = np.load(files[idx])
        samples.append({
            'pe': d['partial_elevation'].astype(np.float32),
            'pm': d['partial_mask'].astype(np.float32),
            'gt': d['gt_elevation'].astype(np.float32),
            'name': files[idx].name,
        })
    return samples


def main():
    args = parse_args()
    base = Path(args.data_root).expanduser() if args.data_root else paths.data_root()
    base = base / args.env
    samples = load_samples(base, args.n_samples, args.seed)

    fig, axes = plt.subplots(len(samples), len(SETTINGS),
                             figsize=(2.9 * len(SETTINGS), 2.8 * len(samples)),
                             squeeze=False, constrained_layout=True)

    for row, s in enumerate(samples):
        valid = s['gt'][np.isfinite(s['gt'])]
        vmin, vmax = np.percentile(valid, [1, 99])

        for col, (label, n_cones, angle) in enumerate(SETTINGS):
            if n_cones is None:
                pe, pm = s['pe'], s['pm']
            else:
                # the library helper draws from `random`; seed per cell so the
                # figure is reproducible
                random.seed(args.seed + row * 10 + col)
                pe, pm = _apply_ray_augmentation(s['pe'], s['pm'], n_cones, (angle, angle))

            ax = axes[row][col]
            ax.imshow(np.where(pm > 0, pe, np.nan), origin='lower', cmap='terrain',
                      vmin=vmin, vmax=vmax)
            ax.set_title(f'{label}\n{pm.mean() * 100:.1f}% coverage', fontsize=17)
            ax.axis('off')

    out = Path(args.out) if args.out else reports_dir() / 'preview_ray_augmentation.png'
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=200, bbox_inches='tight')
    plt.close()
    print(f'Saved {out}')


if __name__ == '__main__':
    main()
