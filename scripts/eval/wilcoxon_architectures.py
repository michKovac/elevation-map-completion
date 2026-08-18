#!/usr/bin/env python
"""
Paired significance test of the main model against the other architectures on
hole RMSE, respecting the correlation structure of the data.

Unit of analysis: TRAJECTORY (default). Per-sample values are averaged per test
trajectory before pairing, because consecutive frames of a trajectory observe
overlapping terrain and are not independent (the same correlation the
leave-one-environment-out design controls for). A per-sample test would commit
pseudoreplication and vastly overstate significance, so it is reported only as a
labeled reference, not as evidence.

For each pair (main vs other) it reports the mean and median trajectory-level
difference, the Wilcoxon signed-rank statistic and p-value, the Holm-adjusted
p-value across the pairwise comparisons, the paired t-test p-value, and Cohen's
d (paired). Δ = main - other; negative means the main model has lower hole RMSE.

Usage:
    python tools/wilcoxon_architectures.py \
        --main runs/cv_unet_resnet34_20260709_214611 \
        --others runs/cv_unet_nll_softaug_20260707_222512 \
                 runs/cv_attunet_20260708_152544 \
                 runs/cv_segformer_20260709_135345 \
        --unit trajectory
"""
import argparse
import csv
import glob
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats


def load(exp_dir: str, metric: str) -> dict:
    """sample_id -> (env, trajectory, value)."""
    out = {}
    for f in glob.glob(f'{exp_dir}/fold_*/eval/per_sample_test.csv'):
        with open(f) as fh:
            for r in csv.DictReader(fh):
                v = r.get(metric)
                if v not in (None, '', 'nan'):
                    out[r['sample_id']] = (r['env'], r['trajectory'], float(v))
    return out


def aggregate(d: dict, unit: str) -> dict:
    """Reduce per-sample values to the chosen independent unit."""
    if unit == 'sample':
        return {k: v[2] for k, v in d.items()}
    keyfn = {'trajectory': lambda e, t: (e, t), 'env': lambda e, t: e}[unit]
    acc = defaultdict(list)
    for _, (env, tr, val) in d.items():
        acc[keyfn(env, tr)].append(val)
    return {k: float(np.mean(v)) for k, v in acc.items()}


def holm(pvals: list) -> list:
    """Holm-Bonferroni adjusted p-values (order preserved)."""
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    adj = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * pvals[i])
        adj[i] = min(running, 1.0)
    return adj


def name(exp_dir: str) -> str:
    return json.load(open(Path(exp_dir) / 'experiment.json'))['name']


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--main', required=True)
    p.add_argument('--others', nargs='+', required=True)
    p.add_argument('--metric', default='nll_rmse_hole')
    p.add_argument('--unit', default='trajectory',
                   choices=['trajectory', 'env', 'sample'])
    args = p.parse_args()

    main_raw = load(args.main, args.metric)
    main_u = aggregate(main_raw, args.unit)
    n_unit = len(main_u)
    print(f'main = {name(args.main)}   metric = {args.metric}   '
          f'unit = {args.unit} (n = {n_unit})\n')

    rows, pvals = [], []
    for other in args.others:
        o_raw = load(other, args.metric)
        o_u = aggregate(o_raw, args.unit)
        keys = sorted(set(main_u) & set(o_u))
        a = np.array([main_u[k] for k in keys])
        b = np.array([o_u[k] for k in keys])
        d = a - b
        try:
            W, pw = stats.wilcoxon(a, b)
        except ValueError:
            W, pw = float('nan'), 1.0
        _, pt = stats.ttest_rel(a, b)
        cohen = d.mean() / d.std(ddof=1)
        rows.append([name(other), len(keys), d.mean(), np.median(d), W, pw, pt, cohen])
        pvals.append(pw)

    p_holm = holm(pvals)
    print(f'{"vs model":<18}{"n":>4}{"meanD":>8}{"medD":>8}{"W":>7}'
          f'{"p_wilc":>9}{"p_holm":>9}{"p_ttest":>9}{"CohenD":>8}')
    print('-' * 78)
    for r, ph in zip(rows, p_holm):
        nm, n, md, mdn, W, pw, pt, cd = r
        print(f'{nm:<18}{n:>4}{md:>8.3f}{mdn:>8.3f}{W:>7.0f}'
              f'{pw:>9.3f}{ph:>9.3f}{pt:>9.3f}{cd:>8.3f}')

    print('\nD = main - other [m] (negative -> main has lower hole RMSE).')
    print('p_holm = Holm-Bonferroni adjusted across the pairwise comparisons.')

    # Reference only: per-sample test (pseudoreplication -> inflated; do NOT cite).
    print('\n[reference, per-sample -- inflated by within-trajectory correlation,'
          ' not used as evidence]')
    for other in args.others:
        o_raw = load(other, args.metric)
        keys = sorted(set(main_raw) & set(o_raw))
        a = np.array([main_raw[k][2] for k in keys])
        b = np.array([o_raw[k][2] for k in keys])
        _, pw = stats.wilcoxon(a, b)
        print(f'  {name(other):<18} n={len(keys):>6}  p_wilcoxon={pw:.1e}')


if __name__ == '__main__':
    main()
