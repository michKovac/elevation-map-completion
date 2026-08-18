#!/usr/bin/env python3
"""
R1 comment 2 (downstream demonstration): turn completed elevation + uncertainty
into a slope-based traversability map and measure how well it recovers the
ground-truth traversability in the hole regions — the decision that a planner
would actually consume. No retraining; runs the trained folds at test time.

Three settings on hole cells (gt-valid, unobserved in the input):
  - Completed          : traversability from the composite (measured + predicted)
  - Completed + sigma-gate : cells with predicted sigma > tau are declared
                          non-traversable (conservative use of uncertainty)
  (a partial-only planner cannot assess hole cells at all: 0 % assessable.)

Metrics are pooled over the samples of one fold and then aggregated as
mean +/- population std over the five folds: traversability accuracy, false-safe
rate (gt unsafe predicted safe, the collision-risk error), unsafe recall and the
assessed fraction of hole cells.

With --tau_sweep the gate threshold is swept, which gives the safety-coverage
trade-off curve (assessed fraction vs false-safe rate).

    python scripts/traversability_eval.py --limit 250 --tau 1.0
    python scripts/traversability_eval.py --tau_sweep

Outputs to runs/experiments_review/r1c2_traversability_demo/:
  traversability_sweep.json, table_traversability_sweep.tex,
  table_traversability_sweep_full.tex, fig_traversability_sweep.png,
  fig_traversability.png (qualitative six-panel example)
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
from elevcomp.traversability import RESOLUTION as RES
from elevcomp.traversability import slope_deg
from elevcomp.traversability import traversability as trav

ROOT = project_root()

RUN = None   # resolved in main(); override with --exp_dir
OUT = reports_dir() / 'traversability'
DEV = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


DEFAULT_SWEEP = '0.3,0.5,0.75,1.0,1.5,2.0,3.0'
OP_POINTS = [(0.5, 'Conservative'), (1.0, 'Balanced'), (2.0, 'Permissive'),
             (float('inf'), 'No gate')]


def tau_label(tau):
    return 'inf' if not np.isfinite(tau) else f'{tau:g}'


def agg(values):
    """mean and population std (ddof=0) over the per-fold values."""
    v = np.asarray(values, dtype=np.float64)
    return float(v.mean()), float(v.std())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--limit', type=int, default=0, help='test samples per fold (0=all)')
    ap.add_argument('--tau', type=float, default=1.0, help='sigma gate threshold [m]')
    ap.add_argument('--tau_sweep', nargs='?', const=DEFAULT_SWEEP, default=None,
                    help=f'sweep the gate threshold, comma separated [m] (default {DEFAULT_SWEEP})')
    ap.add_argument('--slope_thresh', type=float, default=25.0)
    ap.add_argument('--fig_fold', default='fold_2_ModularNeighborhood')
    ap.add_argument('--fig_idx', type=int, default=70)
    ap.add_argument('--exp_dir', default=None,
                    help='cross-validation experiment directory '
                         '(default: $ELEVCOMP_EXPERIMENT or newest runs/cv_*)')
    args = ap.parse_args()

    global RUN
    RUN = Path(args.exp_dir).expanduser().resolve() if args.exp_dir else default_experiment()
    OUT.mkdir(parents=True, exist_ok=True)
    THR = args.slope_thresh

    # gate thresholds; infinity = no gate. The single --tau is always evaluated.
    if args.tau_sweep:
        taus = [float(t) for t in args.tau_sweep.split(',') if t.strip()]
    else:
        taus = []
    for t in (args.tau, float('inf')):
        if t not in taus:
            taus.append(t)
    taus = sorted(taus)

    # per-fold counters, one counter dict per gate threshold
    folds = []          # list of (fold_name, {tau: counters})
    fig_data = None

    for fold_dir in sorted(RUN.glob('fold_*')):
        C = {t: dict(n=0, correct=0, gt_unsafe=0, false_safe=0, accepted=0)
             for t in taus}
        folds.append((fold_dir.name, C))
        ck = torch.load(fold_dir / 'best.pth', map_location=DEV)
        cfg, stats = ck['config'], ck['stats']
        mean, std = stats['mean'], stats['std']
        model = build_model(cfg).to(DEV); model.load_state_dict(ck['model']); model.eval()
        exp = json.load(open(fold_dir.parent / 'experiment.json'))
        data_root = Path(exp['cv']['data_root'])
        split = json.load(open(fold_dir / 'split_files.json'))
        files = [data_root / f for f in split['test_files']]
        if args.limit:
            files = files[::max(1, len(files) // args.limit)][:args.limit]
        ds = ElevationDataset(split='test', augment=False, stats=stats,
                              pad_to=cfg['pad_to'], files=files)
        H0 = 251; ph = (cfg['pad_to'] - H0) // 2
        loader = eval_loader(ds, cfg, batch_size=max(cfg['batch_size'] // 2, 8))
        print(f'[{fold_dir.name}] {len(files)} test samples', flush=True)

        idx = 0
        with torch.no_grad():
            for inp, gt, gm in loader:
                inp = inp.to(DEV); gt = gt.to(DEV); gm = gm.to(DEV)
                pm = inp[:, 1:2]; pe = inp[:, :1]
                pred, sigma = nll_predict(model, inp, True)
                B = inp.shape[0]
                for i in range(B):
                    sl = slice(ph, ph + H0)
                    pm_b = (pm[i, 0].cpu().numpy()[sl, sl] > 0.5)
                    gm_b = (gm[i, 0].cpu().numpy()[sl, sl] > 0.5)
                    pe_m = pe[i, 0].cpu().numpy()[sl, sl] * std + mean
                    gt_m = gt[i, 0].cpu().numpy()[sl, sl] * std + mean
                    pr_m = pred[i, 0].cpu().numpy()[sl, sl] * std + mean
                    sg_m = sigma[i, 0].cpu().numpy()[sl, sl] * std
                    comp = np.where(pm_b, pe_m, pr_m)
                    hole = gm_b & ~pm_b

                    gt_t = trav(slope_deg(gt_m, gm_b), THR)
                    cp_t = trav(slope_deg(comp, gm_b), THR)

                    ev = hole & (gt_t >= 0) & (cp_t >= 0)
                    if ev.sum() == 0:
                        idx += 1; continue
                    gtu = ev & (gt_t == 0)
                    n_ev = int(ev.sum()); n_gtu = int(gtu.sum())
                    fig_gate = None
                    for t in taus:
                        gate = cp_t.copy()
                        gate[(sg_m > t) & (gate >= 0)] = 0      # uncertain -> non-trav
                        c = C[t]
                        c['n'] += n_ev
                        c['correct'] += int((ev & (gate == gt_t)).sum())
                        c['gt_unsafe'] += n_gtu
                        c['false_safe'] += int((gtu & (gate == 1)).sum())
                        c['accepted'] += int((ev & (sg_m <= t)).sum())
                        if t == args.tau:
                            fig_gate = gate

                    if fig_data is None and fold_dir.name == args.fig_fold and idx == args.fig_idx:
                        fig_data = dict(pe=np.where(pm_b, pe_m, np.nan), comp=comp,
                                        sig=np.where(gm_b, sg_m, np.nan),
                                        gt_t=gt_t, cp_t=cp_t, gate=fig_gate,
                                        env=fold_dir.name.split('_', 2)[-1])
                    idx += 1

    # per-fold rates, then mean and population std over the folds
    METRICS = ('acc', 'false_safe', 'unsafe_recall', 'assessed')
    per_fold = {tau_label(t): {m: [] for m in METRICS} for t in taus}
    fold_names = [name for name, _ in folds]
    for _, C in folds:
        for t in taus:
            c = C[t]
            fs = c['false_safe'] / max(c['gt_unsafe'], 1)
            d = per_fold[tau_label(t)]
            d['acc'].append(c['correct'] / max(c['n'], 1))
            d['false_safe'].append(fs)
            d['unsafe_recall'].append(1.0 - fs)
            d['assessed'].append(c['accepted'] / max(c['n'], 1))

    # Class balance of the evaluated hole cells. Gate-independent, so it is read off
    # the first threshold. This is the majority-class rate that a trivial "everything
    # traversable" classifier would reach (at a false-safe rate of 1.0).
    n_cells = [int(C[taus[0]]['n']) for _, C in folds]
    gt_unsafe_cells = [int(C[taus[0]]['gt_unsafe']) for _, C in folds]
    base_trav = [1.0 - u / max(n, 1) for u, n in zip(gt_unsafe_cells, n_cells)]
    base_mean, base_std = agg(base_trav)

    res = dict(slope_thresh=THR, tau=args.tau, folds=fold_names,
               n_cells=n_cells, gt_unsafe_cells=gt_unsafe_cells,
               base_traversable=dict(mean=base_mean, std=base_std,
                                     per_fold=base_trav,
                                     pooled=1.0 - sum(gt_unsafe_cells) / sum(n_cells)),
               taus={})
    for t in taus:
        k = tau_label(t)
        entry = {}
        for m in METRICS:
            mean, std = agg(per_fold[k][m])
            entry[m] = dict(mean=mean, std=std, per_fold=per_fold[k][m])
        res['taus'][k] = entry
    json.dump(res, open(OUT / 'traversability_sweep.json', 'w'), indent=2)

    def cell(t, metric, scale=1.0, dec=3):
        e = res['taus'][tau_label(t)][metric]
        return f'{scale*e["mean"]:.{dec}f} $\\pm$ {scale*e["std"]:.{dec}f}'

    # operating-point table (the one used in the manuscript)
    with open(OUT / 'table_traversability_sweep.tex', 'w') as f:
        f.write('% Auto-generated by scripts/traversability_eval.py --tau_sweep\n')
        f.write(f'% Hole-region traversability, slope threshold {THR:g} deg, '
                'mean +/- population std over the five folds.\n')
        f.write('\\begin{tabular}{llccc}\n\\toprule\n')
        f.write('Operating point & Gate & False-safe rate & Unsafe recall & '
                'Assessed [\\%] \\\\\n\\midrule\n')
        for t, name in OP_POINTS:
            if t not in taus:
                continue
            gate = 'none' if not np.isfinite(t) else f'$\\hat{{\\sigma}} \\le {t:g}$~m'
            f.write(f'{name} & {gate} & {cell(t, "false_safe")} & '
                    f'{cell(t, "unsafe_recall")} & {cell(t, "assessed", 100, 1)} \\\\\n')
        f.write('\\bottomrule\n\\end{tabular}\n')

    # full sweep table (all thresholds, for reference)
    with open(OUT / 'table_traversability_sweep_full.tex', 'w') as f:
        f.write('% Auto-generated by scripts/traversability_eval.py --tau_sweep\n')
        f.write('\\begin{tabular}{lcccc}\n\\toprule\n')
        f.write('Gate $\\tau$ [m] & Accuracy & False-safe rate & Unsafe recall & '
                'Assessed [\\%] \\\\\n\\midrule\n')
        for t in taus:
            lbl = '$\\infty$ (no gate)' if not np.isfinite(t) else f'{t:g}'
            f.write(f'{lbl} & {cell(t, "acc")} & {cell(t, "false_safe")} & '
                    f'{cell(t, "unsafe_recall")} & {cell(t, "assessed", 100, 1)} \\\\\n')
        f.write('\\bottomrule\n\\end{tabular}\n')

    # safety-coverage curve, readable in black and white
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        marks = ['o', 's', '^', 'v', 'D', 'P', 'X', '*', 'h']
        xs = [100 * res['taus'][tau_label(t)]['assessed']['mean'] for t in taus]
        ys = [res['taus'][tau_label(t)]['false_safe']['mean'] for t in taus]
        xe = [100 * res['taus'][tau_label(t)]['assessed']['std'] for t in taus]
        ye = [res['taus'][tau_label(t)]['false_safe']['std'] for t in taus]
        fig, ax = plt.subplots(figsize=(6.4, 4.4))
        ax.plot(xs, ys, '-', color='black', linewidth=1.2, zorder=1)
        ax.errorbar(xs, ys, xerr=xe, yerr=ye, fmt='none', ecolor='0.45',
                    elinewidth=0.9, capsize=2.5, zorder=2)
        for i, t in enumerate(taus):
            last = not np.isfinite(t)
            lbl = 'no gate' if last else f'$\\tau$ = {t:g} m'
            ax.plot(xs[i], ys[i], marker=marks[i % len(marks)], color='black',
                    markersize=7, markerfacecolor='white', markeredgewidth=1.2,
                    linestyle='none', zorder=3)
            ax.annotate(lbl, (xs[i], ys[i]), textcoords='offset points',
                        xytext=(-8, 9) if last else (8, -13), fontsize=9,
                        ha='right' if last else 'left')
        ax.set_xlabel('Assessed fraction of hole cells [%]')
        ax.set_ylabel('False-safe rate')
        ax.set_xlim(min(xs) - 12, 108)
        ax.grid(True, color='0.85', linewidth=0.6)
        ax.set_axisbelow(True)
        plt.tight_layout()
        plt.savefig(OUT / 'fig_traversability_sweep.png', dpi=300)
        plt.close(fig)
        print('wrote fig_traversability_sweep.png')
    except Exception as e:
        print(f'sweep figure skipped: {e}')

    # figure
    if fig_data is not None:
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            from matplotlib.colors import ListedColormap
            tmap = ListedColormap(['#b2182b', '#1a9850'])   # 0 red, 1 green
            fd = fig_data
            fig, ax = plt.subplots(1, 6, figsize=(20, 3.6))
            ax[0].imshow(fd['pe'], origin='lower', cmap='terrain'); ax[0].set_title('Partial input')
            ax[1].imshow(fd['comp'], origin='lower', cmap='terrain'); ax[1].set_title('Completed')
            im = ax[2].imshow(fd['sig'], origin='lower', cmap='magma'); ax[2].set_title(r'Uncertainty $\hat\sigma$ [m]')
            plt.colorbar(im, ax=ax[2], fraction=0.046)
            for a, key, t in ((ax[3], 'gt_t', 'GT traversability'),
                              (ax[4], 'cp_t', 'Completed traversability'),
                              (ax[5], 'gate', r'Completed + $\hat\sigma$-gate')):
                m = np.ma.masked_where(fd[key] < 0, fd[key])
                a.imshow(m, origin='lower', cmap=tmap, vmin=0, vmax=1); a.set_title(t)
            for a in ax:
                a.set_xticks([]); a.set_yticks([])
            plt.tight_layout()
            plt.savefig(OUT / 'fig_traversability.png', dpi=200)
            print('wrote fig_traversability.png')
        except Exception as e:
            print(f'figure skipped: {e}')
    else:
        print('figure sample not hit; adjust --fig_fold/--fig_idx')

    print(f'\n=== hole-region traversability, slope threshold {THR:g} deg ===')
    print('mean +/- population std over the '
          f'{len(fold_names)} folds, {sum(res["n_cells"]):,} evaluated hole cells')
    print(f'base rate: {100*base_mean:.1f} +/- {100*base_std:.1f} % of the evaluated hole '
          f'cells are traversable in the ground truth '
          f'(pooled {100*res["base_traversable"]["pooled"]:.1f} %) — the accuracy of a '
          f'trivial all-traversable classifier, whose false-safe rate would be 1.000')
    print(f'{"tau [m]":>10s}  {"accuracy":>16s}  {"false-safe":>16s}  '
          f'{"unsafe recall":>16s}  {"assessed [%]":>16s}')
    for t in taus:
        e = res['taus'][tau_label(t)]
        lbl = 'no gate' if not np.isfinite(t) else f'{t:g}'
        print(f'{lbl:>10s}  {e["acc"]["mean"]:.3f} +/- {e["acc"]["std"]:.3f}  '
              f'{e["false_safe"]["mean"]:.3f} +/- {e["false_safe"]["std"]:.3f}  '
              f'{e["unsafe_recall"]["mean"]:.3f} +/- {e["unsafe_recall"]["std"]:.3f}  '
              f'{100*e["assessed"]["mean"]:6.1f} +/- {100*e["assessed"]["std"]:.1f}')
    print('\nper-fold false-safe rate')
    for i, name in enumerate(fold_names):
        vals = '  '.join(f'{res["taus"][tau_label(t)]["false_safe"]["per_fold"][i]:.3f}'
                         for t in taus)
        print(f'  {name:28s} {vals}')


if __name__ == '__main__':
    main()
