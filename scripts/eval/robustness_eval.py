#!/usr/bin/env python3
"""
Sector-dropout robustness evaluation — the experiment that justifies (or
refutes) the ray-cone augmentation.

Evaluates trained checkpoints on the TEST split of selected folds while
removing an increasing number of angular sectors from the input, exactly like
the ray-cone training augmentation but DETERMINISTIC PER SAMPLE: sector
directions are seeded by (sample index, level), so every model sees identical
corrupted inputs and the comparison is paired by construction.

Reported per corruption level, in metres:
    rmse / mae            all ground-truth-valid cells
    rmse_hole             hole region of the CORRUPTED mask (original + new holes)
    rmse_removed          only the cells that were observed and got removed —
                          terrain the sensor would normally see; the most direct
                          measure of robustness to missing viewing directions

Usage:
    # base vs ray-augmented model, common folds, sectors of 30°
    python tools/robustness_eval.py \\
        --exp_dirs runs/cv_unet_20260706_140601 runs/cv_unet_nll_softaug_20260707_222512 \\
        --folds 0 1 2 --sectors 0 1 2 3 --width 30

Outputs (reports/robustness/):
    robustness.csv / robustness.json     per (model, fold, level) aggregates
    fig_robustness.png                   degradation curves
    table_robustness.tex                 LaTeX table for the paper
"""
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path


import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from elevcomp.paths import project_root
from elevcomp.metrics import compute_metrics
from elevcomp.model import build_model
from elevcomp.utils import write_csv

ROOT = project_root()
COLORS = ['#0072B2', '#E69F00', '#009E73', '#CC79A7']


# ─────────────────────────────────────────────────────────────────────────────
# Deterministic sector dropout
# ─────────────────────────────────────────────────────────────────────────────

_ANGLE_CACHE = {}


def _angle_map(shape):
    if shape not in _ANGLE_CACHE:
        h, w = shape
        cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
        yy, xx = np.meshgrid(np.arange(h, dtype=np.float32),
                             np.arange(w, dtype=np.float32), indexing='ij')
        _ANGLE_CACHE[shape] = np.degrees(np.arctan2(cy - yy, xx - cx)) % 360.0
    return _ANGLE_CACHE[shape]


def remove_sectors(pe, pm, n_sectors, width_deg, rng):
    """Remove n sectors of fixed width at rng-drawn directions (training-aug
    geometry: wedges anchored at the sensor origin in the grid center)."""
    if n_sectors == 0:
        return pe, pm
    amap = _angle_map(pe.shape)
    pe, pm = pe.copy(), pm.copy()
    for _ in range(n_sectors):
        direction = rng.uniform(0.0, 360.0)
        diff = (amap - direction + 180.0) % 360.0 - 180.0
        cone = np.abs(diff) <= width_deg / 2.0
        pe[cone] = np.nan
        pm[cone] = 0.0
    return pe, pm


class CorruptedTestSet(Dataset):
    """
    Test samples with deterministic sector dropout.

    Returns (inp, gt, gt_mask, removed_mask) where removed_mask marks cells
    that were observed in the original input and removed by the corruption.
    Directions are seeded by (seed, sample index, level) — independent of the
    model and of DataLoader worker scheduling.
    """

    def __init__(self, files, stats, pad_to, n_sectors, width_deg, level_id,
                 seed=42):
        self.files = files
        self.stats = stats
        self.pad_to = pad_to
        self.n_sectors = n_sectors
        self.width_deg = width_deg
        self.level_id = level_id
        self.seed = seed

    def __len__(self):
        return len(self.files)

    def _pad(self, arr):
        H, W = arr.shape
        ph, pw = (self.pad_to - H) // 2, (self.pad_to - W) // 2
        return np.pad(arr, ((ph, self.pad_to - H - ph),
                            (pw, self.pad_to - W - pw)))

    def __getitem__(self, idx):
        d = np.load(self.files[idx])
        pe = d['partial_elevation'].astype(np.float32)
        pm = d['partial_mask'].astype(np.float32)
        gt = d['gt_elevation'].astype(np.float32)
        gm = d['gt_mask'].astype(np.float32)

        rng = np.random.default_rng((self.seed, idx, self.level_id))
        pm0 = pm.copy()
        pe, pm = remove_sectors(pe, pm, self.n_sectors, self.width_deg, rng)
        removed = ((pm0 > 0.5) & (pm < 0.5)).astype(np.float32)

        m, s = self.stats['mean'], self.stats['std']
        pe_n = np.where(np.isfinite(pe) & (pm > 0.5), (pe - m) / s, 0.0)
        gt_n = np.where(np.isfinite(gt), (gt - m) / s, 0.0)

        inp = np.stack([self._pad(pe_n), self._pad(pm)])
        return (torch.from_numpy(inp),
                torch.from_numpy(self._pad(gt_n)[None]),
                torch.from_numpy(self._pad(gm)[None]),
                torch.from_numpy(self._pad(removed)[None]))


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def eval_level(model, ds, stats, device, batch_size, num_workers):
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=True)
    acc, n = {}, 0
    rem_se, rem_n = 0.0, 0          # pooled squared error on removed cells
    std_m = stats['std']

    for inp, gt, gm, removed in loader:
        inp = inp.to(device, non_blocking=True)
        gt, gm = gt.to(device), gm.to(device)
        removed = removed.to(device)
        pred = model(inp)[:, :1]

        for i in range(inp.shape[0]):
            m = compute_metrics(pred[i:i+1], gt[i:i+1], gm[i:i+1],
                                inp[i:i+1, 1:2], None,
                                denorm_std=stats['std'],
                                denorm_mean=stats['mean'])
            for k in ('rmse', 'mae', 'rmse_hole', 'mae_hole'):
                if k in m:
                    acc[k] = acc.get(k, 0.0) + m[k]
            n += 1

        rm = (removed * gm).bool()
        if rm.any():
            diff_m = (pred - gt)[rm] * std_m
            rem_se += float((diff_m ** 2).sum())
            rem_n += int(rm.sum())

    out = {k: v / n for k, v in acc.items()}
    out['rmse_removed'] = float(np.sqrt(rem_se / rem_n)) if rem_n else float('nan')
    out['removed_px_per_sample'] = rem_n / n
    return out


def main():
    p = argparse.ArgumentParser(description='Sector-dropout robustness evaluation.')
    p.add_argument('--exp_dirs', nargs='+', required=True,
                   help='CV experiment directories (their best.pth per fold is used)')
    p.add_argument('--folds', type=int, nargs='+', default=[0, 1, 2])
    p.add_argument('--sectors', type=int, nargs='+', default=[0, 1, 2, 3],
                   help='corruption levels = number of removed sectors')
    p.add_argument('--width', type=float, default=30.0, help='sector width [°]')
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--num_workers', type=int, default=8)
    p.add_argument('--limit', type=int, default=0,
                   help='use only N test samples per fold (0 = all; smoke test)')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--out', default=str(ROOT / 'reports' / 'robustness'))
    args = p.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for exp_arg in args.exp_dirs:
        exp_dir = Path(exp_arg) if Path(exp_arg).is_absolute() else ROOT / exp_arg
        exp = json.load(open(exp_dir / 'experiment.json'))
        folds = json.load(open(exp_dir / 'folds.json'))
        data_root = Path(exp['cv']['data_root'])
        label = exp['name']

        for fd in folds:
            if fd['fold'] not in args.folds:
                continue
            fdir = exp_dir / f'fold_{fd["fold"]}_{fd["test_env"]}'
            if not (fdir / 'best.pth').exists():
                print(f'  ! {label} fold {fd["fold"]}: no best.pth — skipping')
                continue
            ck = torch.load(fdir / 'best.pth', map_location=device)
            cfg, stats = ck['config'], ck['stats']
            model = build_model(cfg).to(device)
            model.load_state_dict(ck['model'])
            model.eval()

            files = [data_root / f for f in fd['test_files']]
            if args.limit:
                files = files[::max(1, len(files) // args.limit)][:args.limit]

            for lvl, n_sec in enumerate(args.sectors):
                ds = CorruptedTestSet(files, stats, cfg['pad_to'],
                                      n_sec, args.width, lvl, args.seed)
                m = eval_level(model, ds, stats, device,
                               args.batch_size, args.num_workers)
                rows.append({'model': label, 'fold': fd['fold'],
                             'test_env': fd['test_env'],
                             'n_sectors': n_sec, 'width_deg': args.width, **m})
                print(f'  {label:<28} fold {fd["fold"]} {n_sec}×{args.width:.0f}° | '
                      f'rmse {m["rmse"]:.3f}  hole {m["rmse_hole"]:.3f}  '
                      f'removed {m["rmse_removed"]:.3f}')

    write_csv(out_dir / 'robustness.csv', rows)
    with open(out_dir / 'robustness.json', 'w') as f:
        json.dump({'generated_at': datetime.now().isoformat(timespec='seconds'),
                   'args': vars(args), 'rows': rows}, f, indent=2)

    # ── degradation curves (mean over folds) ─────────────────────────────────
    models = sorted({r['model'] for r in rows})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    for ax, metric, title in zip(
            axes, ('rmse_removed', 'rmse_hole'),
            ('RMSE on removed sectors [m]', 'RMSE on all holes [m]')):
        for mi, mod in enumerate(models):
            xs = sorted({r['n_sectors'] for r in rows if r['model'] == mod})
            ys = [np.mean([r[metric] for r in rows
                           if r['model'] == mod and r['n_sectors'] == x])
                  for x in xs]
            ax.plot(xs, ys, 'o-', color=COLORS[mi % len(COLORS)], lw=2, ms=6,
                    label=mod)
        ax.set(xlabel=f'removed sectors × {args.width:.0f}°', title=title,
               xticks=args.sectors)
        ax.grid(alpha=0.25)
        ax.spines[['top', 'right']].set_visible(False)
        ax.legend(frameon=False, fontsize=8)
    fig.suptitle(f'Sector-dropout robustness (mean over folds {args.folds}, '
                 f'{len(rows) and rows[0]["test_env"] or ""}…)', fontsize=10)
    fig.savefig(out_dir / 'fig_robustness.png', dpi=200)
    plt.close(fig)

    # ── LaTeX table: rmse_removed, models × levels (mean over folds) ─────────
    lines = [
        '% Auto-generated by tools/robustness_eval.py',
        f'% RMSE on removed sectors [m]; sectors of {args.width:.0f}°, '
        f'mean over folds {args.folds}',
        r'\begin{tabular}{l' + 'c' * len(args.sectors) + '}',
        r'\toprule',
        'Model & ' + ' & '.join(
            (f'{n}$\\times${args.width:.0f}$^\\circ$' if n else 'clean (hole)')
            for n in args.sectors) + r' \\',
        r'\midrule',
    ]
    for mod in models:
        vals = []
        for n_sec in args.sectors:
            sel = [r for r in rows if r['model'] == mod and r['n_sectors'] == n_sec]
            v = np.mean([r['rmse_hole' if n_sec == 0 else 'rmse_removed']
                         for r in sel])
            vals.append(f'{v:.3f}')
        lines.append(mod.replace('_', r'\_') + ' & ' + ' & '.join(vals) + r' \\')
    lines += [r'\bottomrule', r'\end{tabular}']
    (out_dir / 'table_robustness.tex').write_text('\n'.join(lines) + '\n')

    print(f'\nWritten → {out_dir}/robustness.{{csv,json}} + fig_robustness.png '
          f'+ table_robustness.tex')


if __name__ == '__main__':
    main()
