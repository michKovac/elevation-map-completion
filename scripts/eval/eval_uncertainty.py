#!/usr/bin/env python3
"""
Post-training uncertainty evaluation: β-NLL head vs TTA ensemble on a single
checkpoint (single-run workflow; the 5-fold experiment runs the same
comparison automatically per fold — see run_cv_experiment.py).

Loads the checkpoint, runs both inference modes on a split, and produces a
side-by-side comparison figure + printed aggregate metrics.

β-NLL uncertainty:
    Kendall & Gal, NeurIPS 2017. https://arxiv.org/abs/1703.04977
    Seitzer et al., ICLR 2022.   https://arxiv.org/abs/2203.09168

TTA uncertainty (D4 self-ensemble):
    Lim et al., CVPRW 2017.       https://arxiv.org/abs/1707.02921
    Shanmugam et al., ICCV 2021.  https://arxiv.org/abs/2011.11156

Usage:
    python tools/eval_uncertainty.py --checkpoint runs/unet_xxx/best.pth
    python tools/eval_uncertainty.py --checkpoint runs/unet_xxx/best.pth --split test --n_samples 12
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path


import numpy as np
import torch

from elevcomp.dataset import ElevationDataset
from elevcomp.figures import visualize_uncertainty
from elevcomp.inference import nll_predict, tta_predict
from elevcomp.metrics import compute_metrics
from elevcomp.model import build_model


# ── Aggregate metrics ─────────────────────────────────────────────────────────

@torch.no_grad()
def aggregate_stats(model, dataset, device, stats, has_uncertainty):
    """
    Run over the full dataset split and compute:
      RMSE / MAE (all GT-valid, hole-only, composite)
      Calibration: Pearson(σ, |err|) and % σ > |err|
    for both β-NLL and TTA.
    """
    sum_nll = defaultdict(float)
    sum_tta = defaultdict(float)
    err_nll_all, err_tta_all = [], []
    sig_nll_all, sig_tta_all = [], []

    n = len(dataset)
    for idx in range(n):
        sample = dataset[idx]
        inp    = sample[0].unsqueeze(0).to(device)
        gt_t   = sample[1].unsqueeze(0).to(device)
        gm_t   = sample[2].unsqueeze(0).to(device)
        pm_t   = inp[:, 1:2]
        pe_t   = inp[:, :1]

        pred_nll, sigma_nll = nll_predict(model, inp, has_uncertainty)
        pred_tta, sigma_tta = tta_predict(model, inp)

        m_nll = compute_metrics(pred_nll, gt_t, gm_t, pm_t, pe_t,
                                denorm_std=stats['std'], denorm_mean=stats['mean'])
        m_tta = compute_metrics(pred_tta, gt_t, gm_t, pm_t, pe_t,
                                denorm_std=stats['std'], denorm_mean=stats['mean'])

        for k, v in m_nll.items():
            sum_nll[k] += v
        for k, v in m_tta.items():
            sum_tta[k] += v

        # Calibration — pixel-level (hole region only)
        gm_np = gm_t[0, 0].cpu().numpy().astype(bool)
        pm_np = pm_t[0, 0].cpu().numpy().astype(bool)
        hole  = gm_np & ~pm_np

        pr_nll_np = pred_nll[0, 0].cpu().numpy()
        pr_tta_np = pred_tta[0, 0].cpu().numpy()
        gt_np     = gt_t[0, 0].cpu().numpy()

        err_nll_all.append(np.abs((pr_nll_np - gt_np) * stats['std'])[hole])
        err_tta_all.append(np.abs((pr_tta_np - gt_np) * stats['std'])[hole])

        if has_uncertainty:
            sig_nll_all.append((sigma_nll[0, 0].cpu().numpy() * stats['std'])[hole])
        sig_tta_all.append((sigma_tta[0, 0].cpu().numpy() * stats['std'])[hole])

    avg_nll = {k: v / n for k, v in sum_nll.items()}
    avg_tta = {k: v / n for k, v in sum_tta.items()}

    err_nll_all = np.concatenate(err_nll_all)
    err_tta_all = np.concatenate(err_tta_all)
    sig_tta_all = np.concatenate(sig_tta_all)

    w = 15   # column width for aligned printing

    def row(label, nll_val, tta_val, fmt='.4f'):
        print(f'  {label:<{w}} {nll_val:{fmt}}   {tta_val:{fmt}}')

    print('\n── Aggregate metrics ────────────────────────────────────────────────')
    print(f'  {"":>{w}} {"β-NLL":>8}   {"TTA":>8}')
    print(f'  {"─"*35}')
    row('MAE   [m]',        avg_nll.get('mae',      float('nan')), avg_tta.get('mae',      float('nan')))
    row('RMSE  [m]',        avg_nll.get('rmse',     float('nan')), avg_tta.get('rmse',     float('nan')))
    row('AbsRel',           avg_nll.get('abs_rel',  float('nan')), avg_tta.get('abs_rel',  float('nan')))
    row('MAE  hole [m]',    avg_nll.get('mae_hole', float('nan')), avg_tta.get('mae_hole', float('nan')))
    row('RMSE hole [m]',    avg_nll.get('rmse_hole',float('nan')), avg_tta.get('rmse_hole',float('nan')))
    row('MAE  comp [m]',    avg_nll.get('mae_comp', float('nan')), avg_tta.get('mae_comp', float('nan')))
    row('RMSE comp [m]',    avg_nll.get('rmse_comp',float('nan')), avg_tta.get('rmse_comp',float('nan')))
    row('SSIM',             avg_nll.get('ssim',     float('nan')), avg_tta.get('ssim',     float('nan')))
    print(f'  {"─"*35}')

    print(f'\n── Calibration (hole pixels) ────────────────────────────────────────')
    print(f'  {"":>{w}} {"β-NLL":>8}   {"TTA":>8}')
    print(f'  {"─"*35}')

    if has_uncertainty:
        sig_nll_all = np.concatenate(sig_nll_all)
        pearson_nll = np.corrcoef(sig_nll_all, err_nll_all)[0, 1]
        pct_nll     = (sig_nll_all > err_nll_all).mean() * 100
        mean_sig_nll = sig_nll_all.mean()
    else:
        pearson_nll = float('nan')
        pct_nll     = float('nan')
        mean_sig_nll = float('nan')

    pearson_tta = np.corrcoef(sig_tta_all, err_tta_all)[0, 1]
    pct_tta     = (sig_tta_all > err_tta_all).mean() * 100

    row('Mean σ [m]',       mean_sig_nll,   sig_tta_all.mean())
    row('Pearson(σ,|e|)',   pearson_nll,    pearson_tta)
    row('% σ>|e| (→68%)',  pct_nll,        pct_tta, fmt='.1f')
    print('─────────────────────────────────────────────────────────────────────')


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description='Compare β-NLL and TTA uncertainty on a trained checkpoint.'
    )
    p.add_argument('--checkpoint', required=True, help='Path to .pth checkpoint')
    p.add_argument('--split',      default='val', choices=['train', 'val', 'test'])
    p.add_argument('--n_samples',  type=int, default=8,  help='Samples in figure')
    p.add_argument('--seed',       type=int, default=42, help='RNG seed for sample selection')
    p.add_argument('--out',        default='', help='Output PNG path (default: next to checkpoint)')
    p.add_argument('--no_stats',   action='store_true', help='Skip aggregate metrics (faster)')
    args = p.parse_args()

    ckpt_path = Path(args.checkpoint)
    run_dir   = ckpt_path.parent

    # config.json lives in run root; uncertainty_head checkpoints are one level deeper
    for search_dir in (run_dir, run_dir.parent):
        cfg_path   = search_dir / 'config.json'
        stats_path = search_dir / 'stats.json'
        if cfg_path.exists():
            break
    else:
        raise FileNotFoundError(f'config.json not found near {ckpt_path}')

    with open(cfg_path)   as f: cfg   = json.load(f)
    with open(stats_path) as f: stats = json.load(f)

    has_uncertainty = cfg.get('uncertainty', False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device      : {device}')
    print(f'Checkpoint  : {ckpt_path}')
    print(f'Model       : {cfg["model"]}')
    print(f'Uncertainty : {has_uncertainty}  (β-NLL head)')
    print(f'Split       : {args.split}')

    model = build_model(cfg).to(device)
    ckpt  = torch.load(ckpt_path, map_location=device, weights_only=True)
    state = ckpt['model'] if isinstance(ckpt, dict) and 'model' in ckpt else ckpt
    model.load_state_dict(state)
    model.eval()

    dataset = ElevationDataset(
        cfg['data_dir'], split=args.split, augment=False,
        stats=stats, pad_to=cfg['pad_to'],
        fix_mask_rotation=cfg.get('fix_mask_rotation', False),
    )
    print(f'Dataset     : {len(dataset)} samples\n')

    rng     = np.random.default_rng(args.seed)
    indices = rng.choice(len(dataset), size=min(args.n_samples, len(dataset)), replace=False).tolist()

    out_path = Path(args.out) if args.out else run_dir / f'uncertainty_comparison_{args.split}.png'
    visualize_uncertainty(model, dataset, stats, indices, device, has_uncertainty, out_path)

    if not args.no_stats:
        aggregate_stats(model, dataset, device, stats, has_uncertainty)


if __name__ == '__main__':
    main()
