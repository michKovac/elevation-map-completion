#!/usr/bin/env python3
"""
R2 comment 2 (supporting evidence): does elevation completion need long-range
context? For each sample we keep only the observed cells within a radius r of the
holes (zeroing farther observations) and measure the hole RMSE. If the error
saturates at a moderate radius, the task is dominated by local/multiscale context,
which helps explain why decoder attention or a transformer backbone gives no gain.

The radius grid must be resolved against the actual hole geometry: holes are
scattered over the whole map, so almost every observed cell sits within a few
metres of some hole. Radii of 5 m and above therefore remove nothing at all and
return the full-map result bit for bit. The grid below is sub-metre, and the
table reports the share of observed cells each radius keeps, so the reader can
see how much context was actually taken away.

    python tools/context_radius_eval.py --limit 300

Outputs to runs/experiments_review/r2c2_context_radius/:
  context_radius.json, table_context_radius.tex, fig_context_radius.png
"""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
import torch
from scipy.ndimage import distance_transform_edt

from elevcomp.paths import project_root, reports_dir
from elevcomp.dataset import ElevationDataset
from elevcomp.inference import nll_predict
from elevcomp.model import build_model
from elevcomp.folds import eval_loader

ROOT = project_root()

RUN = None   # resolved in main(); override with --exp_dir
OUT = reports_dir() / 'context_radius'
DEV = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
RES = 0.2
RADII = [0.2, 0.4, 0.6, 1.0, 1.4, 2.0, None]   # None = full map


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--exp_dir', default=None,
                    help='cross-validation experiment directory '
                         '(default: $ELEVCOMP_EXPERIMENT or newest runs/cv_*)')
    args = ap.parse_args()

    global RUN
    RUN = Path(args.exp_dir).expanduser().resolve() if args.exp_dir else default_experiment()
    OUT.mkdir(parents=True, exist_ok=True)
    labels = [f'{r:g} m' if r else 'Full' for r in RADII]
    fold_means = {lab: [] for lab in labels}   # per-fold mean hole RMSE
    fold_kept = {lab: [] for lab in labels}    # per-fold mean share of observed cells kept

    for fold_dir in sorted(RUN.glob('fold_*')):
        ck = torch.load(fold_dir / 'best.pth', map_location=DEV)
        cfg, stats = ck['config'], ck['stats']
        std_m = stats['std']
        model = build_model(cfg).to(DEV).eval(); model.load_state_dict(ck['model'])
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

        per_sample = {lab: [] for lab in labels}   # within this fold
        kept_sample = {lab: [] for lab in labels}
        with torch.no_grad():
            for inp, gt, gm in loader:
                inp = inp.to(DEV); gt = gt.to(DEV); gm = gm.to(DEV)
                for i in range(inp.shape[0]):
                    pm0 = (inp[i, 1] > 0.5)
                    hole = (gm[i, 0] > 0.5) & ~pm0
                    if hole.sum() < 20:
                        continue
                    n_obs = int(pm0.sum())
                    hole_np = hole.cpu().numpy()
                    # distance [m] from each cell to nearest hole cell
                    dist = distance_transform_edt(~hole_np) * RES
                    dist = torch.from_numpy(dist).to(DEV)
                    # one batch holds every radius variant of this sample
                    ci = inp[i:i + 1].repeat(len(RADII), 1, 1, 1)
                    for k, r in enumerate(RADII):
                        if r is None:
                            kept_sample[labels[k]].append(1.0)
                            continue
                        far = pm0 & (dist > r)
                        ci[k, 0][far] = 0.0
                        ci[k, 1][far] = 0.0
                        kept = 1.0 - (int(far.sum()) / max(n_obs, 1))
                        kept_sample[labels[k]].append(kept)
                    pred, _ = nll_predict(model, ci, True)
                    for k, lab in enumerate(labels):
                        e = (pred[k, 0] - gt[i, 0])[hole] * std_m
                        per_sample[lab].append(float(torch.sqrt((e ** 2).mean())))
        for lab in labels:
            if per_sample[lab]:
                fold_means[lab].append(float(np.mean(per_sample[lab])))
                fold_kept[lab].append(float(np.mean(kept_sample[lab])))

    # aggregate over folds (population std, matching summary.json convention)
    rows = []
    for lab in labels:
        v = np.array(fold_means[lab])
        k = np.array(fold_kept[lab])
        rows.append(dict(radius=lab, hole_rmse=float(v.mean()),
                         std=float(v.std()), kept=float(k.mean()),
                         kept_std=float(k.std()), nfolds=int(v.size),
                         per_fold=[float(x) for x in v]))   # ddof=0
    json.dump(rows, open(OUT / 'context_radius.json', 'w'), indent=2)

    with open(OUT / 'table_context_radius.tex', 'w') as f:
        f.write('% Auto-generated by tools/context_radius_eval.py\n')
        f.write('\\begin{tabular}{lcc}\n\\toprule\n')
        f.write('Observed context radius & Observed cells kept [\\%] & '
                'Hole RMSE [m] \\\\\n\\midrule\n')
        for r in rows:
            f.write(f'{r["radius"]} & {100*r["kept"]:.1f} & '
                    f'{r["hole_rmse"]:.3f} $\\pm$ {r["std"]:.3f} \\\\\n')
        f.write('\\bottomrule\n\\end{tabular}\n')

    try:
        import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
        fin = [r for r in rows if r['radius'] != 'Full']
        xs = [float(r['radius'].split()[0]) for r in fin]
        ys = [r['hole_rmse'] for r in fin]
        es = [r['std'] for r in fin]
        plt.figure(figsize=(4.8, 3.3))
        plt.errorbar(xs, ys, yerr=es, color='black', marker='o',
                     markersize=6, linewidth=1.6, markerfacecolor='white', capsize=3,
                     label='restricted context')
        plt.axhline(rows[-1]['hole_rmse'], color='black', linestyle='--',
                    linewidth=1.2, label='full map')
        plt.xlabel('Observed context radius around holes [m]')
        plt.ylabel('Hole RMSE [m]')
        plt.legend(fontsize=8, frameon=False)
        plt.tight_layout(); plt.savefig(OUT / 'fig_context_radius.png', dpi=200)
        print('wrote fig_context_radius.png')
    except Exception as e:
        print(f'figure skipped: {e}')

    print('\n=== hole RMSE [m] vs observed context radius (mean +/- std over folds) ===')
    print(f'{"radius":>8s}  {"kept [%]":>9s}  hole RMSE')
    for r in rows:
        print(f'{r["radius"]:>8s}  {100*r["kept"]:9.1f}  '
              f'{r["hole_rmse"]:.3f} +/- {r["std"]:.3f}')
    print(f'CHECK: Full = {rows[-1]["hole_rmse"]:.3f} m (paper 2.855 m)')


if __name__ == '__main__':
    main()
