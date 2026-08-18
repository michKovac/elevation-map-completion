#!/usr/bin/env python3
"""
R2 comment 4: post-hoc scalar (temperature) calibration of the beta-NLL sigma.

For each CV fold, a single scale gamma is fit on that fold's VALIDATION hole
pixels so that the validation coverage@1sigma matches the nominal Gaussian value
of 0.68 (gamma = 68th percentile of err/sigma). This coverage-matching estimator
is robust to the few near-zero-sigma pixels that make the mean(err^2/sigma^2)
maximum-likelihood scale explode. gamma is applied unchanged to the held-out TEST
pixels — no test-set leakage. A positive scale is monotone, so Pearson r and AUSE
are unchanged.

The test coverages are pooled over ALL hole pixels of the fold, accumulated
streamingly, which is the pixel set behind table_uncertainty.tex. cov_raw is
taken from summary.json so that the two tables cannot disagree, and the streaming
pass must reproduce it (asserted per fold, together with the pixel count).
(Before 2026-08-12 they were computed on the stored 200-pixels-per-sample
subsample in calibration_pixels_test.npz, which made two per-environment values
differ from table_uncertainty.tex by 0.01 after rounding.)

    python tools/temperature_scaling.py   (use the venv)

Outputs to runs/experiments_review/r2c4_temperature_scaling/:
  temp_scaling.json, per_env.csv, table_temperature_scaling.tex
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
from elevcomp.calibration import MIN_HOLE_PX

ROOT = project_root()

RUN = None   # resolved in main(); override with --exp_dir
OUT = reports_dir() / 'temperature_scaling'
DEV = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Same cuDNN regime as the evaluation pass of core/cv.py:193. Without it the kernel
# selection differs and the pooled coverage moves by up to 2e-4, because a knife-edge
# count (sigma > |err|) amplifies last-bit differences of the forward pass.
torch.backends.cudnn.benchmark = True


def load_fold(fold_dir):
    """Load one trained fold: (model, cfg, stats, data_root, split)."""
    ck = torch.load(fold_dir / 'best.pth', map_location=DEV)
    cfg, stats = ck['config'], ck['stats']
    model = build_model(cfg).to(DEV)
    model.load_state_dict(ck['model'])
    model.eval()
    exp = json.load(open(fold_dir.parent / 'experiment.json'))
    data_root = Path(exp['cv']['data_root'])
    split = json.load(open(fold_dir / 'split_files.json'))
    return model, cfg, stats, data_root, split


def hole_pixels(model, cfg, stats, files, split_name):
    """
    Yield (|err| [m], sigma [m]) per sample over its hole pixels.

    Same pixel set as the pooled calibration of core/cv.py: gt-valid and
    unobserved cells, samples with fewer than MIN_HOLE_PX hole cells skipped.
    """
    unc = cfg.get('uncertainty', True)
    std_m = stats['std']
    ds = ElevationDataset(split=split_name, augment=False, stats=stats,
                          pad_to=cfg['pad_to'], files=files)
    loader = eval_loader(ds, cfg, batch_size=max(cfg['batch_size'] // 2, 8))
    with torch.no_grad():
        for inp, gt, gm in loader:
            inp = inp.to(DEV); gt = gt.to(DEV); gm = gm.to(DEV)
            pm = inp[:, 1:2]
            pred, sigma = nll_predict(model, inp, unc)
            for i in range(inp.shape[0]):
                gm_np = gm[i, 0].cpu().numpy().astype(bool)
                pm_np = pm[i, 0].cpu().numpy() > 0.5
                hole = gm_np & ~pm_np
                if hole.sum() < MIN_HOLE_PX:
                    continue
                e = np.abs(pred[i, 0].cpu().numpy() - gt[i, 0].cpu().numpy())[hole] * std_m
                s = sigma[i, 0].cpu().numpy()[hole] * std_m
                yield e.astype(np.float32), s.astype(np.float32)


def val_pixels(model, cfg, stats, data_root, split):
    """Return (err[m], sigma[m]) over all validation hole pixels."""
    files = [data_root / f for f in split['val_files']]
    errs, sigs = [], []
    for e, s in hole_pixels(model, cfg, stats, files, 'val'):
        errs.append(e); sigs.append(s)
    return np.concatenate(errs), np.concatenate(sigs)


def test_stats(model, cfg, stats, data_root, split, gamma):
    """
    Pooled test statistics over ALL hole pixels, accumulated streamingly.

    Coverage is counted exactly as core/calibration.py:StreamingCorr does
    (sigma > |err| in float64), so cov_raw must reproduce the cov68 value of
    summary.json; cov_cal applies the same test to gamma*sigma.
    """
    n = cov_raw = cov_cal = 0.0
    sig_sum = nll_raw_sum = nll_cal_sum = 0.0
    files = [data_root / f for f in split['test_files']]
    for e, s in hole_pixels(model, cfg, stats, files, 'test'):
        y = e.astype(np.float64); x = s.astype(np.float64)
        n += x.size
        cov_raw += float((x > y).sum())
        cov_cal += float((gamma * x > y).sum())
        sig_sum += x.sum()
        nll_raw_sum += gnll_sum(y, x)
        nll_cal_sum += gnll_sum(y, gamma * x)
    return dict(cov_raw=cov_raw / n, cov_cal=cov_cal / n,
                nll_raw=nll_raw_sum / n, nll_cal=nll_cal_sum / n,
                sig_raw=sig_sum / n, sig_cal=gamma * sig_sum / n,
                n_pixels=int(n))


def gnll_sum(err, sig):
    """Summed Gaussian NLL over the given pixels (sigma floored for stability)."""
    sig = np.maximum(sig, 1e-6)
    return float(np.sum(0.5 * np.log(2 * np.pi * sig ** 2) + err ** 2 / (2 * sig ** 2)))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--exp_dir', default=None,
                    help='cross-validation experiment directory '
                         '(default: $ELEVCOMP_EXPERIMENT or newest runs/cv_*)')
    args = ap.parse_args()

    global RUN
    RUN = Path(args.exp_dir).expanduser().resolve() if args.exp_dir else default_experiment()
    OUT.mkdir(parents=True, exist_ok=True)

    summary = json.load(open(RUN / 'summary' / 'summary.json'))
    recs = []
    for fold_dir in sorted(RUN.glob('fold_*')):
        model, cfg, stats, data_root, split = load_fold(fold_dir)
        env = split['test_env']
        print(f'[{fold_dir.name}] validation forward pass...', flush=True)
        ev, sv = val_pixels(model, cfg, stats, data_root, split)
        # coverage-matching temperature: gamma s.t. P(err/sigma < gamma) = 0.68 on val
        gamma = float(np.quantile(ev / np.maximum(sv, 1e-6), 0.68))
        del ev, sv
        print(f'[{fold_dir.name}] test forward pass (all hole pixels)...', flush=True)
        stat = test_stats(model, cfg, stats, data_root, split, gamma)
        ref = summary['folds'][fold_dir.name.split('_')[1]]['calibration']['nll']['global']
        # Structural check: exactly the pixel set behind table_uncertainty.tex.
        assert stat['n_pixels'] == ref['n_pixels'], (
            f'{fold_dir.name}: {stat["n_pixels"]} pixels != summary {ref["n_pixels"]}')
        # Numerical check: float32 inference is not bit-reproducible across runs and
        # GPUs, so a few hundred of the ~2e8 pixels flip the sigma > |err| test. With
        # the cuDNN regime above the observed gap is ~2e-6; the tolerance is set far
        # looser than that but still 50x below the 0.005 rounding step of the table,
        # so a wrong pixel set or a wrong definition would still trip it.
        assert abs(stat['cov_raw'] - ref['cov68']) < 1e-4, (
            f'{fold_dir.name}: cov_raw {stat["cov_raw"]} != summary cov68 {ref["cov68"]}')
        # Report the stored raw coverage, so this table and table_uncertainty.tex
        # cannot disagree; gamma and cov_cal come from the pass above.
        rec = dict(fold=fold_dir.name, env=env, gamma=gamma, cov_raw=ref['cov68'],
                   cov_raw_recomputed=stat['cov_raw'], cov_cal=stat['cov_cal'],
                   nll_raw=stat['nll_raw'], nll_cal=stat['nll_cal'],
                   sig_raw=stat['sig_raw'], sig_cal=stat['sig_cal'],
                   n_pixels=stat['n_pixels'])
        recs.append(rec)
        print(f'  {env}: gamma={gamma:.3f}  cov {rec["cov_raw"]:.3f}->{rec["cov_cal"]:.3f}'
              f'  NLL {rec["nll_raw"]:.3f}->{rec["nll_cal"]:.3f}'
              f'  [{rec["n_pixels"]:,} hole pixels; recomputed raw '
              f'{rec["cov_raw_recomputed"]:.5f} vs stored {ref["cov68"]:.5f}]', flush=True)

    json.dump(recs, open(OUT / 'temp_scaling.json', 'w'), indent=2)
    import statistics as st
    def m(k): return st.mean(r[k] for r in recs)
    def sd(k): return st.pstdev([r[k] for r in recs])   # population std over folds

    with open(OUT / 'per_env.csv', 'w') as f:
        f.write('env,gamma,cov_raw,cov_cal,nll_raw,nll_cal\n')
        for r in recs:
            f.write(f'{r["env"]},{r["gamma"]:.3f},{r["cov_raw"]:.3f},{r["cov_cal"]:.3f},'
                    f'{r["nll_raw"]:.3f},{r["nll_cal"]:.3f}\n')

    ENV_SHORT = {'ForestEnv': 'Forest', 'Gascola': 'Gascola',
                 'ModularNeighborhood': 'Modular', 'OldTownSummer': 'Old Town',
                 'SeasonalForestWinter': 'Winter'}
    with open(OUT / 'table_temperature_scaling.tex', 'w') as f:
        f.write('% Auto-generated by tools/temperature_scaling.py\n')
        f.write('% gamma fit on validation (coverage-matching), applied to held-out test (no leakage).\n')
        f.write('% Coverages pooled over all test hole pixels — same pixel set as table_uncertainty.tex.\n')
        f.write('\\begin{tabular}{lccc}\n\\toprule\n')
        f.write('Test environment & $\\gamma$ & Cov@1$\\hat{\\sigma}$ (raw) & Cov@1$\\hat{\\sigma}$ (cal.) \\\\\n\\midrule\n')
        for r in recs:
            f.write(f'{ENV_SHORT.get(r["env"], r["env"])} & {r["gamma"]:.2f} & '
                    f'{r["cov_raw"]:.2f} & {r["cov_cal"]:.2f} \\\\\n')
        f.write('\\midrule\n')
        f.write(f'Mean & {m("gamma"):.2f} $\\pm$ {sd("gamma"):.2f} & '
                f'{m("cov_raw"):.2f} $\\pm$ {sd("cov_raw"):.2f} & '
                f'{m("cov_cal"):.2f} $\\pm$ {sd("cov_cal"):.2f} \\\\\n')
        f.write('\\bottomrule\n\\end{tabular}\n')

    print(f'\nMEAN  cov {m("cov_raw"):.3f} -> {m("cov_cal"):.3f}  |  mean gamma {m("gamma"):.3f}')
    print('(ideal coverage = 0.68; Pearson r and AUSE unchanged by a positive scale;'
          ' NLL kept in temp_scaling.json only — unstable due to sigma-floor outliers)')


if __name__ == '__main__':
    main()
