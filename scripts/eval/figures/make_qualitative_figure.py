#!/usr/bin/env python
"""
Build the qualitative results figure: one median-error test sample per fold.

Rows  = held-out environments (folds).
Cols  = partial input (E⊙M), composite output, ground truth, absolute error,
        predicted uncertainty σ (β-NLL head).

Samples are picked at the per-fold median of nll_rmse_hole (from
fold_*/eval/per_sample_test.csv), so they are representative rather than curated.
Predictions are read from fold_*/predictions/test/*.npz (normalized float16,
aligned to the source NPZ) and denormalized with the fold stats in
predictions/meta.json. No GPU or model rebuild needed.

Usage:
    python tools/make_qualitative_figure.py \
        --exp_dir runs/cv_unet_resnet34_20260709_214611 \
        --out MDPI___Raimoc/figures/fig_results_qualitative.png
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from elevcomp import paths
from elevcomp.paths import project_root

ROOT = project_root()


def median_row(csv_path: Path, key: str = 'nll_rmse_hole') -> dict:
    with open(csv_path) as f:
        rows = [r for r in csv.DictReader(f) if r.get(key) not in (None, '', 'nan')]
    rows.sort(key=lambda r: float(r[key]))
    return rows[len(rows) // 2]


# Short, non-distracting environment labels (full names in the caption).
ENV_SHORT = {
    'ForestEnv': 'Forest', 'Gascola': 'Gascola',
    'ModularNeighborhood': 'Modular', 'OldTownSummer': 'Old Town',
    'SeasonalForestWinter': 'Winter',
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--exp_dir', required=True)
    p.add_argument('--data_root', default=str(paths.data_root()))
    p.add_argument('--out', default=str(ROOT / 'results' / 'figures' / 'fig_results_qualitative.png'))
    args = p.parse_args()

    exp_dir = Path(args.exp_dir)
    data_root = Path(args.data_root).resolve()
    meta = json.load(open(exp_dir / sorted(d.name for d in exp_dir.glob('fold_*'))[0]
                          / 'predictions' / 'meta.json'))
    fold_dirs = sorted(exp_dir.glob('fold_*'), key=lambda d: d.name)

    col_titles = ['Partial input', 'Composite', 'Ground truth',
                  'Absolute error [m]', 'Uncertainty $\\sigma$ [m]']
    n_rows, n_cols = len(fold_dirs), len(col_titles)
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(2.7 * n_cols, 2.7 * n_rows),
                             constrained_layout=True)

    for r, fdir in enumerate(fold_dirs):
        env = fdir.name.split('_', 2)[2]
        stats = json.load(open(fdir / 'stats.json'))
        mean, std = stats['mean'], stats['std']
        row = median_row(fdir / 'eval' / 'per_sample_test.csv')

        src = np.load(data_root / row['sample_id'])
        pe = src['partial_elevation'].astype(np.float32)
        pm = src['partial_mask'].astype(bool)
        gt = src['gt_elevation'].astype(np.float32)
        gm = src['gt_mask'].astype(bool)

        pred_npz = np.load(fdir / 'predictions' / row['pred_file'])
        pred = pred_npz['pred_nll'].astype(np.float32) * std + mean   # metres
        sigma = pred_npz['sigma_nll'].astype(np.float32) * std        # metres

        comp = np.where(pm, pe, pred)
        err = np.abs(comp - gt)

        # shared elevation scale from GT-valid cells
        gv = gt[gm]
        vmin, vmax = np.percentile(gv, 1), np.percentile(gv, 99)

        panels = [
            (np.where(pm, pe, np.nan), 'terrain', vmin, vmax),
            (np.where(gm, comp, np.nan), 'terrain', vmin, vmax),
            (np.where(gm, gt, np.nan), 'terrain', vmin, vmax),
            (np.where(gm, err, np.nan), 'magma', 0.0, float(np.percentile(err[gm], 95))),
            (np.where(gm, sigma, np.nan), 'viridis', 0.0, float(np.percentile(sigma[gm], 95))),
        ]
        for c, (img, cmap, lo, hi) in enumerate(panels):
            ax = axes[r, c]
            im = ax.imshow(img, origin='lower', cmap=cmap, vmin=lo, vmax=hi)
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(col_titles[c], fontsize=20)
            if c in (3, 4):
                cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
                cb.ax.tick_params(labelsize=14)
        cov = float(row.get('coverage', float('nan'))) * 100
        axes[r, 0].set_ylabel(f'{ENV_SHORT.get(env, env)}\n({cov:.0f}% obs.)', fontsize=18)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved → {args.out}')


if __name__ == '__main__':
    main()
