#!/usr/bin/env python3
"""
Evaluate a trained checkpoint on the test split and save visualisations.

Usage:
    python evaluate.py --checkpoint runs/pconv_20240622_143022/best.pth
    python evaluate.py --checkpoint runs/pconv_20240622_143022/best.pth --visualize --n_vis 16
"""
import sys
from pathlib import Path
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from elevcomp.dataset import ElevationDataset
from elevcomp.losses import ElevationLoss
from elevcomp.metrics import compute_metrics
from elevcomp.model import build_model, count_parameters


# ─────────────────────────────────────────────────────────────────────────────

def load_checkpoint(path: Path, device: torch.device):
    ckpt  = torch.load(path, map_location=device)
    cfg   = ckpt['config']   # plain dict saved by train.py
    stats = ckpt['stats']
    model = build_model(cfg).to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()
    return model, cfg, stats, ckpt


def _denorm(x: np.ndarray, stats: dict) -> np.ndarray:
    return x * stats['std'] + stats['mean']


def save_visualization(inp, pred, gt, gt_mask, idx: int, out_dir: Path, stats: dict):
    """Save a 4-panel comparison figure (partial / predicted / GT / error)."""
    pm = inp[1]                               # partial mask, H×W
    pe = _denorm(inp[0], stats)               # partial elevation, H×W
    pr = _denorm(pred[0], stats)              # predicted, H×W
    gt_e = _denorm(gt[0], stats)              # ground truth, H×W
    gm = gt_mask[0].astype(bool)

    err = np.abs(pr - gt_e)
    err[~gm] = np.nan

    vmin = float(np.nanpercentile(gt_e, 1))
    vmax = float(np.nanpercentile(gt_e, 99))
    emax = float(np.nanpercentile(err[gm], 95)) if gm.any() else 1.0

    partial_vis = np.where(pm > 0.5, pe, np.nan)

    fig, axes = plt.subplots(1, 4, figsize=(20, 5), constrained_layout=True)
    panels = [
        ('Partial (stereo)', partial_vis, 'terrain',  vmin, vmax),
        ('Predicted',        pr,          'terrain',  vmin, vmax),
        ('GT (simulation)',  gt_e,        'terrain',  vmin, vmax),
        ('|Error| [m]',      err,         'hot',      0.0,  emax),
    ]

    for ax, (title, data, cmap, lo, hi) in zip(axes, panels):
        im = ax.imshow(data, cmap=cmap, vmin=lo, vmax=hi, origin='lower')
        ax.set_title(title, fontsize=11)
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02)

    obs_pct = float(pm.mean() * 100)
    fig.suptitle(
        f'Sample {idx:04d}  |  stereo coverage: {obs_pct:.1f}%',
        fontsize=12, fontweight='bold',
    )

    out_path = out_dir / f'sample_{idx:04d}.png'
    plt.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close()
    return out_path


# ─────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser('Evaluate elevation completion model')
    p.add_argument('--checkpoint', required=True, help='Path to .pth checkpoint')
    p.add_argument('--split',      default='test', choices=['train', 'val', 'test'])
    p.add_argument('--batch_size', type=int, default=8)
    p.add_argument('--num_workers',type=int, default=4)
    p.add_argument('--visualize',  action='store_true')
    p.add_argument('--n_vis',      type=int, default=16, help='Number of samples to visualise')
    p.add_argument('--device',     default='cuda' if torch.cuda.is_available() else 'cpu')
    args = p.parse_args()

    device   = torch.device(args.device)
    ckpt_path = Path(args.checkpoint)
    run_dir   = ckpt_path.parent

    # ── Load ─────────────────────────────────────────────────────────────────
    model, cfg, stats, ckpt = load_checkpoint(ckpt_path, device)
    n_params = count_parameters(model)
    print(f'\nModel      : {cfg["model"]}  |  params: {n_params:,}')
    print(f'Checkpoint : epoch {ckpt["epoch"]}  |  best val RMSE: {ckpt.get("best_rmse", "?"):.4f} m')

    # ── Dataset ───────────────────────────────────────────────────────────────
    # Single runs carry data_dir in their config (fractional split); CV fold
    # checkpoints instead ship exact file lists in split_files.json.
    split_json = run_dir / 'split_files.json'
    if 'data_dir' in cfg:
        ds = ElevationDataset(cfg['data_dir'], args.split, augment=False,
                              stats=stats, pad_to=cfg['pad_to'])
    elif split_json.exists():
        split = json.load(open(split_json))
        exp = json.load(open(run_dir.parent / 'experiment.json'))
        data_root = Path(exp['cv']['data_root'])
        files = [data_root / f for f in split[f'{args.split}_files']]
        ds = ElevationDataset(split=args.split, augment=False, stats=stats,
                              pad_to=cfg['pad_to'], files=files)
    else:
        raise FileNotFoundError(
            'Config has no data_dir and no split_files.json found — '
            'cannot reconstruct the evaluation split.')
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    print(f'Split      : {args.split}  |  samples: {len(ds)}\n')

    loss_fn = ElevationLoss(cfg['w_valid'], cfg['w_hole'])
    metrics_acc: dict = {}
    total_loss = 0.0
    vis_buffer = []

    # ── Evaluation loop ───────────────────────────────────────────────────────
    with torch.no_grad():
        for inp, gt, gt_mask in loader:
            inp, gt, gt_mask = (
                inp.to(device), gt.to(device), gt_mask.to(device),
            )
            partial_mask = inp[:, 1:2]
            # Channel 0 is always the prediction (channel 1 = log_var when the
            # model was trained with uncertainty=true).
            pred = model(inp)[:, :1]

            total_loss += loss_fn(pred, gt, gt_mask, partial_mask).item()
            m = compute_metrics(
                pred, gt, gt_mask, partial_mask,
                denorm_std=stats['std'], denorm_mean=stats['mean'],
            )
            for k, v in m.items():
                metrics_acc[k] = metrics_acc.get(k, 0.0) + v

            if args.visualize and len(vis_buffer) < args.n_vis:
                for b in range(inp.shape[0]):
                    if len(vis_buffer) < args.n_vis:
                        vis_buffer.append((
                            inp[b].cpu().numpy(),
                            pred[b].cpu().numpy(),
                            gt[b].cpu().numpy(),
                            gt_mask[b].cpu().numpy(),
                        ))

    n = max(len(loader), 1)
    final = {k: v / n for k, v in metrics_acc.items()}
    final['loss'] = total_loss / n

    # ── Print metrics ─────────────────────────────────────────────────────────
    WIDTH = 20
    print('=' * 44)
    print(f'{"Metric":<{WIDTH}}  {"Value":>10}')
    print('-' * 44)
    units = {'mae': ' m', 'rmse': ' m', 'mae_hole': ' m', 'rmse_hole': ' m', 'abs_rel': '', 'ssim': '', 'loss': ''}
    for k, v in final.items():
        u = units.get(k, '')
        print(f'  {k:<{WIDTH-2}}  {v:>8.4f}{u}')
    print('=' * 44)

    # ── Save metrics ──────────────────────────────────────────────────────────
    out_json = run_dir / f'metrics_{args.split}.json'
    with open(out_json, 'w') as f:
        json.dump(final, f, indent=2)
    print(f'\nMetrics saved → {out_json}')

    # ── Visualisations ────────────────────────────────────────────────────────
    if args.visualize and vis_buffer:
        vis_dir = run_dir / f'vis_{args.split}'
        vis_dir.mkdir(exist_ok=True)
        print(f'\nSaving {len(vis_buffer)} visualisations to {vis_dir} …')
        for i, (inp_np, pred_np, gt_np, gm_np) in enumerate(vis_buffer):
            path = save_visualization(inp_np, pred_np, gt_np, gm_np, i, vis_dir, stats)
            print(f'  [{i+1:02d}/{len(vis_buffer)}] {path.name}')


if __name__ == '__main__':
    main()
