#!/usr/bin/env python3
"""
Test-Time Augmentation (TTA) uncertainty estimation on a single checkpoint.

Runs the model with all 8 elements of the D4 dihedral group (core/inference.py),
then reports per-pixel mean prediction and std (σ) as an ensemble-based
uncertainty estimate, with a per-sample visualization including the σ/|error|
calibration ratio panel.

D4 self-ensemble for super-resolution / image restoration:
    Lim et al., "Enhanced Deep Residual Networks for Single Image
    Super-Resolution", CVPRW 2017. https://arxiv.org/abs/1707.02921

TTA as predictive uncertainty:
    Shanmugam et al., "Better Aggregation in Test-Time Augmentation",
    ICCV 2021. https://arxiv.org/abs/2011.11156

Usage:
    python tools/tta_inference.py --checkpoint runs/unet_20260625_152534/best.pth
    python tools/tta_inference.py --checkpoint runs/attunet_xxx/best.pth --n_samples 12
    python tools/tta_inference.py --checkpoint runs/unet_xxx/best.pth --split test
"""
import argparse
import json
import sys
from pathlib import Path


import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from elevcomp.dataset import ElevationDataset
from elevcomp.inference import D4_TRANSFORMS, tta_predict
from elevcomp.model import build_model


# ── Visualization ─────────────────────────────────────────────────────────────

def visualize(model, dataset, stats, indices, device, out_path: Path):
    mean_s, std_s = stats['mean'], stats['std']

    def denorm(x):
        return x * std_s + mean_s

    n = len(indices)
    fig, axes = plt.subplots(n, 6, figsize=(21, 3.2 * n), constrained_layout=True)
    if n == 1:
        axes = axes[None]

    col_titles = [
        'Partial (stereo)', 'Pred (mean)', 'GT (simulation)',
        '|Error| [m]', 'σ TTA [m]', 'σ / |Error|',
    ]
    for col, title in enumerate(col_titles):
        axes[0, col].set_title(title, fontsize=9, fontweight='bold')

    for row, idx in enumerate(indices):
        sample = dataset[idx]
        inp = sample[0].unsqueeze(0).to(device)   # (1, 2, H, W)
        gt  = sample[1].unsqueeze(0)               # (1, 1, H, W)
        gm  = sample[2].unsqueeze(0)               # (1, 1, H, W)

        mean_pred, sigma = tta_predict(model, inp)

        pm  = inp[0, 1].cpu().numpy()
        pe  = denorm(inp[0, 0].cpu().numpy())
        pr  = denorm(mean_pred[0, 0].cpu().numpy())
        gte = denorm(gt[0, 0].numpy())
        sig = sigma[0, 0].cpu().numpy() * std_s    # convert to metres
        gm_np = gm[0, 0].numpy().astype(bool)

        partial_vis = np.where(pm > 0.5, pe, np.nan)
        err = np.abs(pr - gte)
        err[~gm_np] = np.nan
        sig[~gm_np] = np.nan

        # calibration ratio: σ / |error| — ideally ≈ 1
        ratio = np.where(err > 0, sig / (err + 1e-6), np.nan)
        ratio[~gm_np] = np.nan

        valid = gte[gm_np]
        vmin  = float(np.nanpercentile(valid, 1))
        vmax  = float(np.nanpercentile(valid, 99))
        emax  = float(np.nanpercentile(err[gm_np], 95)) if gm_np.any() else 1.0
        smax  = float(np.nanpercentile(sig[gm_np], 95)) if gm_np.any() else 1.0

        panels = [
            (partial_vis, 'terrain', vmin,  vmax,  None),
            (pr,          'terrain', vmin,  vmax,  None),
            (gte,         'terrain', vmin,  vmax,  None),
            (err,         'hot',     0.0,   emax,  None),
            (sig,         'plasma',  0.0,   smax,  None),
            (ratio,       'RdYlGn',  0.0,   2.0,   '0=overconf  1=ideal  2=underconf'),
        ]

        for j, (data, cmap, lo, hi, label) in enumerate(panels):
            im = axes[row, j].imshow(data, cmap=cmap, vmin=lo, vmax=hi, origin='lower')
            axes[row, j].axis('off')
            plt.colorbar(im, ax=axes[row, j], fraction=0.046, pad=0.02)
            if label and row == 0:
                axes[row, j].set_xlabel(label, fontsize=7)

        cov = float(pm.mean() * 100)
        mean_err = float(np.nanmean(err))
        mean_sig = float(np.nanmean(sig))
        axes[row, 0].set_ylabel(
            f'#{idx}\n{cov:.0f}% cov\nMAE={mean_err:.3f}m\nσ={mean_sig:.3f}m',
            fontsize=7, rotation=0, labelpad=80, va='center',
        )

    fig.suptitle(
        f'TTA uncertainty  ({len(D4_TRANSFORMS)} augmentations — D4 symmetry group)',
        fontsize=11,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved → {out_path}')


# ── Aggregate stats ───────────────────────────────────────────────────────────

@torch.no_grad()
def aggregate_stats(model, dataset, device, std_s, max_samples=200):
    """Compute calibration stats over many samples (metres)."""
    errors, sigmas = [], []

    n = min(len(dataset), max_samples)
    indices = torch.randperm(len(dataset))[:n].tolist()

    for idx in indices:
        sample = dataset[idx]
        inp = sample[0].unsqueeze(0).to(device)
        gt  = sample[1]
        gm  = sample[2].squeeze().numpy().astype(bool)

        mean_pred, sigma = tta_predict(model, inp)
        pr = mean_pred[0, 0].cpu().numpy()
        gt_np = gt[0].numpy()

        errors.append(np.abs(pr - gt_np)[gm] * std_s)
        sigmas.append(sigma[0, 0].cpu().numpy()[gm] * std_s)

    errors = np.concatenate(errors)
    sigmas = np.concatenate(sigmas)

    print(f'\n── Aggregate TTA stats ({n} samples) ──')
    print(f'  Mean |error|  : {errors.mean():.4f} m')
    print(f'  Mean σ        : {sigmas.mean():.4f} m')
    print(f'  Pearson(σ,err): {np.corrcoef(sigmas, errors)[0,1]:.3f}  (higher = better calibrated)')
    print(f'  % σ > |err|   : {(sigmas > errors).mean()*100:.1f}%  (ideally ~68% for Gaussian)')


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True, help='Path to .pth checkpoint')
    p.add_argument('--split',      default='val', choices=['train', 'val', 'test'])
    p.add_argument('--n_samples',  type=int, default=8)
    p.add_argument('--seed',       type=int, default=42)
    p.add_argument('--out',        default='', help='Output PNG path (default: next to checkpoint)')
    p.add_argument('--stats',      action='store_true', help='Also compute aggregate calibration stats')
    args = p.parse_args()

    ckpt_path = Path(args.checkpoint)
    run_dir   = ckpt_path.parent

    # ── Load config + stats ───────────────────────────────────────────────────
    # config.json lives in the run root; for uncertainty_head checkpoints the
    # checkpoint is one level deeper, so fall back to the parent directory.
    cfg_path = run_dir / 'config.json'
    if not cfg_path.exists():
        cfg_path = run_dir.parent / 'config.json'
    if not cfg_path.exists():
        raise FileNotFoundError(f'config.json not found in {run_dir} or its parent')
    with open(cfg_path) as f:
        cfg = json.load(f)

    stats_path = run_dir / 'stats.json'
    if not stats_path.exists():
        stats_path = run_dir.parent / 'stats.json'
    with open(stats_path) as f:
        stats = json.load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device     : {device}')
    print(f'Checkpoint : {ckpt_path}')
    print(f'Model      : {cfg["model"]}')
    print(f'Split      : {args.split}')
    print(f'TTA augs   : {len(D4_TRANSFORMS)}')

    # ── Build model ───────────────────────────────────────────────────────────
    # Build with the same config as training (preserves uncertainty heads).
    # tta_predict always takes out[:, :1], so uncertainty=True models work fine.
    model = build_model(cfg).to(device)
    ckpt  = torch.load(ckpt_path, map_location=device)
    # Support both train.py format ({'model': ...}) and raw state-dict format
    state_dict = ckpt['model'] if isinstance(ckpt, dict) and 'model' in ckpt else ckpt
    model.load_state_dict(state_dict)
    model.eval()

    # ── Dataset ───────────────────────────────────────────────────────────────
    dataset = ElevationDataset(
        cfg['data_dir'], split=args.split, augment=False,
        stats=stats, pad_to=cfg['pad_to'],
        fix_mask_rotation=cfg.get('fix_mask_rotation', False),
    )
    print(f'Dataset    : {len(dataset)} samples\n')

    torch.manual_seed(args.seed)
    indices = torch.randperm(len(dataset))[:args.n_samples].tolist()

    # ── Visualize ─────────────────────────────────────────────────────────────
    out_path = Path(args.out) if args.out else run_dir / f'tta_{args.split}.png'
    visualize(model, dataset, stats, indices, device, out_path)

    # ── Optional aggregate stats ──────────────────────────────────────────────
    if args.stats:
        aggregate_stats(model, dataset, device, stats['std'])


if __name__ == '__main__':
    main()
