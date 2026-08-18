#!/usr/bin/env python3
"""
R2 comment 1a: controlled sensor-degradation stress test (test time, no retrain).

The trained model is evaluated while the PARTIAL input is corrupted before the
network, in two independent ways plus one combined condition:
  - noise : additive Gaussian noise on observed elevation cells (proxy for range
            noise), standard deviation in metres;
  - drop  : random removal of a fraction of observed cells (sparse returns).
Hole RMSE is always measured on the ORIGINAL hole region (gt-valid, unobserved
in the clean input), so we isolate how input corruption degrades completion of
the same target region. Clean condition must reproduce the paper's 2.855 m.

    python tools/degradation_eval.py --limit 400

Outputs to runs/experiments_review/r2c1a_degradation/:
  degradation.json, table_degradation.tex
"""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
import torch

from elevcomp.paths import project_root, reports_dir
from elevcomp.dataset import ElevationDataset
from elevcomp.inference import nll_predict
from elevcomp.model import build_model
from elevcomp.folds import eval_loader

ROOT = project_root()

RUN = None   # resolved in main(); override with --exp_dir
OUT = reports_dir() / 'degradation'
DEV = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# (name, noise_std_m, drop_fraction)
CONDITIONS = [
    ('Clean',            0.0,  0.0),
    ('Noise 0.05 m',     0.05, 0.0),
    ('Noise 0.10 m',     0.10, 0.0),
    ('Noise 0.20 m',     0.20, 0.0),
    ('Noise 0.50 m',     0.50, 0.0),
    ('Drop 25%',         0.0,  0.25),
    ('Drop 50%',         0.0,  0.50),
    ('Drop 75%',         0.0,  0.75),
    ('Drop 90%',         0.0,  0.90),
    ('Noise 0.10 + Drop 50%', 0.10, 0.50),
]


def corrupt(inp, noise_m, drop, std_m, gen):
    """Return a corrupted copy of inp and the ORIGINAL observation mask."""
    inp = inp.clone()
    pe = inp[:, 0]; pm = inp[:, 1]
    obs0 = pm > 0.5                      # original observed cells
    if drop > 0:
        r = torch.rand(pe.shape, generator=gen, device=pe.device)
        dropped = obs0 & (r < drop)
        pe[dropped] = 0.0
        pm[dropped] = 0.0
    if noise_m > 0:
        n = torch.randn(pe.shape, generator=gen, device=pe.device) * (noise_m / std_m)
        cur = pm > 0.5
        pe[cur] = pe[cur] + n[cur]
    return inp, obs0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--limit', type=int, default=0, help='test samples per fold (0=all)')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--exp_dir', default=None,
                    help='cross-validation experiment directory '
                         '(default: $ELEVCOMP_EXPERIMENT or newest runs/cv_*)')
    args = ap.parse_args()

    global RUN
    RUN = Path(args.exp_dir).expanduser().resolve() if args.exp_dir else default_experiment()
    OUT.mkdir(parents=True, exist_ok=True)

    # per-condition list of per-FOLD mean hole RMSE (metres)
    fold_means = {c[0]: [] for c in CONDITIONS}

    for fold_dir in sorted(RUN.glob('fold_*')):
        ck = torch.load(fold_dir / 'best.pth', map_location=DEV)
        cfg, stats = ck['config'], ck['stats']
        std_m = stats['std']
        model = build_model(cfg).to(DEV); model.load_state_dict(ck['model']); model.eval()
        exp = json.load(open(fold_dir.parent / 'experiment.json'))
        data_root = Path(exp['cv']['data_root'])
        split = json.load(open(fold_dir / 'split_files.json'))
        files = [data_root / f for f in split['test_files']]
        if args.limit:
            files = files[::max(1, len(files) // args.limit)][:args.limit]
        ds = ElevationDataset(split='test', augment=False, stats=stats,
                              pad_to=cfg['pad_to'], files=files)
        loader = eval_loader(ds, cfg, batch_size=max(cfg['batch_size'] // 2, 8))
        print(f'[{fold_dir.name}] {len(files)} samples', flush=True)

        per_sample = {c[0]: [] for c in CONDITIONS}   # within this fold
        with torch.no_grad():
            for inp, gt, gm in loader:
                inp = inp.to(DEV); gt = gt.to(DEV); gm = gm.to(DEV)
                for name, noise_m, drop in CONDITIONS:
                    gen = torch.Generator(device=DEV).manual_seed(args.seed)
                    cinp, obs0 = corrupt(inp, noise_m, drop, std_m, gen)
                    pred, _ = nll_predict(model, cinp, True)
                    for i in range(inp.shape[0]):
                        hole = (gm[i, 0] > 0.5) & ~obs0[i]      # original holes
                        if hole.sum() < 20:
                            continue
                        e = (pred[i, 0] - gt[i, 0])[hole] * std_m
                        per_sample[name].append(float(torch.sqrt((e ** 2).mean())))
        for name, _, _ in CONDITIONS:
            if per_sample[name]:
                fold_means[name].append(float(np.mean(per_sample[name])))

    # aggregate over folds (population std, matching summary.json convention)
    rows = []
    clean = float(np.mean(fold_means['Clean']))
    cr = round(clean, 3)
    for name, _, _ in CONDITIONS:
        v = np.array(fold_means[name])
        m = float(v.mean()); s = float(v.std())         # ddof=0
        dd = round(m, 3) - cr                            # delta from displayed 3-dec values
        rows.append(dict(cond=name, hole_rmse=m, std=s, delta=dd,
                         nfolds=int(v.size)))
    json.dump(rows, open(OUT / 'degradation.json', 'w'), indent=2)

    def dcell(r):
        if r['cond'] == 'Clean':
            return '--'
        if abs(r['delta']) < 5e-4:                       # rounds to 0.000 at 3 dp
            return '$<0.001$'
        return f'${r["delta"]:+.3f}$'

    with open(OUT / 'table_degradation.tex', 'w') as f:
        f.write('% Auto-generated by tools/degradation_eval.py\n')
        f.write('\\begin{tabular}{lcc}\n\\toprule\n')
        f.write('Corruption & Hole RMSE [m] & $\\Delta$ [m] \\\\\n\\midrule\n')
        for r in rows:
            cond = r['cond'].replace('%', r'\%')        # escape for LaTeX
            val = f'{r["hole_rmse"]:.3f} $\\pm$ {r["std"]:.3f}'
            f.write(f'{cond} & {val} & {dcell(r)} \\\\\n')
        f.write('\\bottomrule\n\\end{tabular}\n')

    print('\n=== hole RMSE [m] under input degradation (mean +/- std over folds) ===')
    for r in rows:
        d = '' if r['cond'] == 'Clean' else f'  ({r["delta"]:+.3f})'
        print(f'  {r["cond"]:24s} {r["hole_rmse"]:.3f} +/- {r["std"]:.3f}{d}')
    print(f'\nCHECK: clean = {clean:.3f} m (paper hole RMSE = 2.855 m)')


if __name__ == '__main__':
    main()
