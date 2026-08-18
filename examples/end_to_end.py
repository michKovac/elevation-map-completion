#!/usr/bin/env python3
"""
End-to-end demonstration: one raw sample to a traversability decision.

    input .npz  ->  completed elevation + per-cell sigma
                ->  composite map (measured where observed, predicted elsewhere)
                ->  slope  ->  traversability  ->  uncertainty-gated traversability

Writes a six-panel figure and prints the metrics on hole cells, i.e. the cells
the model had to invent and where the decision actually matters.

    python examples/end_to_end.py --checkpoint runs/cv_.../fold_0_*/best.pth
    python examples/end_to_end.py --checkpoint <ckpt> --sample examples/sample/<file>.npz

With no --sample it uses the first file in examples/sample/.
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from elevcomp.data.io import load_sample
from elevcomp.inference import nll_predict
from elevcomp.model import build_model
from elevcomp.traversability import (DEFAULT_SLOPE_THRESHOLD_DEG, NON_TRAVERSABLE, TRAVERSABLE,
                                     UNKNOWN, apply_sigma_gate, false_safe_rate, slope_deg,
                                     traversability)

SAMPLE_DIR = Path(__file__).resolve().parent / 'sample'


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--checkpoint', required=True, help='trained fold checkpoint (best.pth)')
    p.add_argument('--sample', default=None, help='input .npz (default: first in examples/sample/)')
    p.add_argument('--tau', type=float, default=1.0, help='sigma gate threshold [m]')
    p.add_argument('--slope_thresh', type=float, default=DEFAULT_SLOPE_THRESHOLD_DEG)
    p.add_argument('--out', default='end_to_end.png')
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    return p.parse_args()


def complete(sample: dict, checkpoint: Path, device: torch.device) -> tuple:
    """
    Run the model on one sample.

    Returns (prediction, sigma) in metres, cropped back to the native grid. The
    network sees normalised elevation padded to a power-of-two canvas, so both
    steps are undone here.
    """
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    cfg, stats = ckpt['config'], ckpt['stats']
    mean, std = stats['mean'], stats['std']

    model = build_model(cfg).to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()

    elev, mask = sample['partial_elevation'], sample['partial_mask']
    size = elev.shape[0]
    pad = (cfg['pad_to'] - size) // 2

    norm = np.where(mask, (np.nan_to_num(elev) - mean) / std, 0.0).astype(np.float32)
    inp = np.zeros((2, cfg['pad_to'], cfg['pad_to']), dtype=np.float32)
    inp[0, pad:pad + size, pad:pad + size] = norm
    inp[1, pad:pad + size, pad:pad + size] = mask.astype(np.float32)

    pred, sigma = nll_predict(model, torch.from_numpy(inp)[None].to(device),
                              has_uncertainty=cfg.get('uncertainty', False))

    crop = slice(pad, pad + size)
    pred_m = pred[0, 0].cpu().numpy()[crop, crop] * std + mean
    sigma_m = (sigma[0, 0].cpu().numpy()[crop, crop] * std if sigma is not None
               else np.zeros_like(pred_m))
    return pred_m, sigma_m


def main():
    args = parse_args()

    sample_path = Path(args.sample) if args.sample else next(iter(sorted(SAMPLE_DIR.glob('*.npz'))), None)
    if sample_path is None:
        raise SystemExit(f'No sample given and none found in {SAMPLE_DIR}')
    sample = load_sample(sample_path)

    pred, sigma = complete(sample, Path(args.checkpoint), torch.device(args.device))

    observed = sample['partial_mask']
    gt_valid = sample['gt_mask']
    gt = sample['gt_elevation']

    # the map a planner would consume: keep what was measured, fill the rest
    composite = np.where(observed, np.nan_to_num(sample['partial_elevation']), pred)

    gt_trav = traversability(slope_deg(gt, gt_valid), args.slope_thresh)
    pred_trav = traversability(slope_deg(composite, gt_valid), args.slope_thresh)
    gated_trav = apply_sigma_gate(pred_trav, sigma, args.tau)

    # hole cells: unobserved but with valid ground truth — what completion is for
    holes = (~observed) & gt_valid
    rmse = float(np.sqrt(np.mean((pred[holes] - gt[holes]) ** 2)))

    print(f'sample          : {sample_path.name}')
    print(f'observed        : {100 * observed.mean():.1f}% of the window')
    print(f'hole RMSE       : {rmse:.3f} m')
    print(f'mean sigma      : {sigma[holes].mean():.3f} m')
    # cells the metrics are computed on: holes where both maps have a decision
    evaluated = holes & (gt_trav != UNKNOWN) & (pred_trav != UNKNOWN)
    n_eval = max(int(evaluated.sum()), 1)

    for name, tmap, tau in (('no gate', pred_trav, float('inf')),
                            (f'sigma <= {args.tau} m', gated_trav, args.tau)):
        fsr, n_fs, n_unsafe = false_safe_rate(gt_trav, tmap, evaluated)
        acc = int((evaluated & (tmap == gt_trav)).sum()) / n_eval
        # "assessed" is the share the gate leaves confidently decided, which is
        # what trades against safety
        assessed = 100 * int((evaluated & (sigma <= tau)).sum()) / n_eval
        print(f'{name:16s}: accuracy {acc:.3f}, false-safe {fsr:.3f} '
              f'({n_fs}/{n_unsafe}), assessed {assessed:.1f}% of hole cells')

    panels = [
        ('Input (observed)', np.where(observed, sample['partial_elevation'], np.nan), 'terrain', None),
        ('Completed', composite, 'terrain', None),
        ('Ground truth', np.where(gt_valid, gt, np.nan), 'terrain', None),
        ('Predicted σ [m]', np.where(holes, sigma, np.nan), 'magma', None),
        (f'Traversability ({args.slope_thresh:g}°)', pred_trav.astype(float), 'RdYlGn', (0, 1)),
        (f'Gated (σ ≤ {args.tau:g} m)', gated_trav.astype(float), 'RdYlGn', (0, 1)),
    ]

    finite = gt[np.isfinite(gt)]
    vmin, vmax = np.percentile(finite, [1, 99])

    fig, axes = plt.subplots(2, 3, figsize=(15, 10), constrained_layout=True)
    for ax, (title, data, cmap, clim) in zip(axes.ravel(), panels):
        data = np.ma.masked_invalid(data)
        if clim is None and cmap == 'terrain':
            im = ax.imshow(data, origin='lower', cmap=cmap, vmin=vmin, vmax=vmax)
        elif clim is None:
            im = ax.imshow(data, origin='lower', cmap=cmap)
        else:
            im = ax.imshow(np.ma.masked_where(data < 0, data), origin='lower',
                           cmap=cmap, vmin=clim[0], vmax=clim[1])
        ax.set_title(title)
        ax.axis('off')
        fig.colorbar(im, ax=ax, shrink=0.8)

    fig.suptitle(f'{sample_path.name} — hole RMSE {rmse:.2f} m', fontsize=13)
    plt.savefig(args.out, dpi=150, bbox_inches='tight')
    print(f'\nSaved {args.out}')


if __name__ == '__main__':
    main()
