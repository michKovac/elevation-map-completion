#!/usr/bin/env python
"""
Appendix figure: visual comparison of classical baselines against the main model
(U-Net with ResNet-34 encoder), one representative test sample per environment.

Rows = held-out environments; columns = partial input, three classical baselines
(nearest, linear, Telea inpainting), our completed map, and ground truth. Every
map is the deployment output: observed cells are kept, only the holes are filled.
Samples are picked at the per-fold median hole RMSE of our model, so they are
representative rather than curated. Large fonts, minimal labels; no GPU needed
(predictions are read from the saved NPZ files).

Usage:
    python tools/make_baseline_comparison_figure.py \
        --exp_dir runs/cv_unet_resnet34_20260709_214611 \
        --out MDPI___Raimoc/figures/fig_baseline_comparison.png
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import sys
from elevcomp import paths
from elevcomp.paths import project_root
from elevcomp.baselines import BASELINES

ROOT = project_root()

# Short, non-distracting environment labels (full names in the caption).
ENV_LABEL = {
    'ForestEnv': 'Forest', 'Gascola': 'Gascola',
    'ModularNeighborhood': 'Modular', 'OldTownSummer': 'Old Town',
    'SeasonalForestWinter': 'Winter',
}
BASELINE_COLS = [('nearest', 'Nearest'), ('linear', 'Linear'), ('telea', 'Telea')]
FS = 22   # base font size (figure is downscaled to \textwidth, so this reads ~text size)


def median_row(csv_path: Path, key='nll_rmse_hole'):
    with open(csv_path) as f:
        rows = [r for r in csv.DictReader(f) if r.get(key) not in (None, '', 'nan')]
    rows.sort(key=lambda r: float(r[key]))
    return rows[len(rows) // 2]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--exp_dir', required=True)
    p.add_argument('--data_root', default=str(paths.data_root()))
    p.add_argument('--out', default=str(ROOT / 'results' / 'figures' / 'fig_baseline_comparison.png'))
    args = p.parse_args()

    exp_dir, data_root = Path(args.exp_dir), Path(args.data_root).resolve()
    fold_dirs = sorted(exp_dir.glob('fold_*'), key=lambda d: d.name)

    col_titles = ['Input'] + [t for _, t in BASELINE_COLS] + ['Ours', 'Ground truth']
    n_rows, n_cols = len(fold_dirs), len(col_titles)
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(2.5 * n_cols, 2.5 * n_rows + 0.4),
                             constrained_layout=True)

    for r, fdir in enumerate(fold_dirs):
        env = fdir.name.split('_', 2)[2]
        stats = json.load(open(fdir / 'stats.json'))
        m, s = stats['mean'], stats['std']
        row = median_row(fdir / 'eval' / 'per_sample_test.csv')

        src = np.load(data_root / row['sample_id'])
        pe = src['partial_elevation'].astype(np.float32)   # metres, NaN in holes
        pm = src['partial_mask'].astype(np.float32)
        gt = src['gt_elevation'].astype(np.float32)
        gm = src['gt_mask'].astype(bool)
        obs = pm > 0.5

        pred = np.load(fdir / 'predictions' / row['pred_file'])['pred_nll'].astype(np.float32) * s + m
        ours = np.where(obs, np.nan_to_num(pe), pred)      # composite

        maps = {'Input': np.where(obs, pe, np.nan)}
        for key, title in BASELINE_COLS:
            maps[title] = BASELINES[key](pe, pm)           # observed + fill
        maps['Ours'] = ours
        maps['Ground truth'] = gt

        gv = gt[gm]
        vmin, vmax = np.percentile(gv, 2), np.percentile(gv, 98)
        for c, title in enumerate(col_titles):
            ax = axes[r, c]
            img = np.where(gm, maps[title], np.nan) if title != 'Input' \
                else np.where(obs & gm, pe, np.nan)
            ax.imshow(img, origin='lower', cmap='terrain', vmin=vmin, vmax=vmax)
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
            if r == 0:
                ax.set_title(title, fontsize=FS, pad=10)
        axes[r, 0].set_ylabel(ENV_LABEL.get(env, env), fontsize=FS, labelpad=12)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved → {args.out}')


if __name__ == '__main__':
    main()
