#!/usr/bin/env python3
"""
Example training run for elevation map completion.

A compact, readable demonstration of the method rather than the experiment
harness of the paper: it trains one model on one data root with the masked
beta-NLL loss and the ray-cone augmentation, and reports hole-region error and
uncertainty calibration on the held-out split.

The paper trains this same model five times in a leave-one-environment-out
protocol and reports the mean; pointing --data_root at a single environment
directory reproduces one such run, pointing it at the dataset root trains on
everything with a random split.

    # quick check that the pipeline runs (minutes, CPU is enough)
    python scripts/train.py --epochs 2 --warmup 1 --limit 200

    # a real run on one environment
    python scripts/train.py --data_root datasets/elevation_dataset/OldTownSummer

Writes runs/<name>_<timestamp>/ containing best.pth and last.pth. Checkpoints
carry their own config and normalisation statistics, so examples/end_to_end.py
needs nothing else.
"""
import argparse
import json
import time
import tomllib
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.amp import GradScaler

from elevcomp.calibration import MIN_HOLE_PX
from elevcomp.dataset import ElevationDataset, get_dataloaders
from elevcomp.inference import nll_predict
from elevcomp.losses import ElevationLoss
from elevcomp.model import build_model, count_parameters
from elevcomp.paths import CONFIG_DIR, data_root, runs_dir
from elevcomp.training import evaluate, make_state, save_checkpoint, train_one_epoch
from elevcomp.utils import seed_everything


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--data_root', default=None,
                   help='dataset root or one environment directory '
                        '(default: $ELEVCOMP_DATA_ROOT or datasets/elevation_dataset)')
    p.add_argument('--name', default='resnet34', help='run name')
    p.add_argument('--out', default=None, help='output directory (default: runs/<name>_<ts>)')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')

    o = p.add_argument_group('config overrides (defaults come from configs/default.toml)')
    o.add_argument('--epochs', type=int)
    o.add_argument('--batch_size', type=int)
    o.add_argument('--lr', type=float)
    o.add_argument('--warmup', type=int, dest='uncertainty_warmup',
                   help='epochs of plain L1 before beta-NLL is enabled')
    o.add_argument('--aug_ray_p', type=float, help='ray-cone augmentation probability')
    o.add_argument('--num_workers', type=int)
    p.add_argument('--limit', type=int, default=0,
                   help='use at most N samples in total (smoke tests)')
    return p.parse_args()


def load_config(args) -> dict:
    with open(CONFIG_DIR / 'default.toml', 'rb') as f:
        cfg = tomllib.load(f)
    for key in ('epochs', 'batch_size', 'lr', 'uncertainty_warmup', 'aug_ray_p',
                'num_workers'):
        value = getattr(args, key, None)
        if value is not None:
            cfg[key] = value
    cfg['data_dir'] = str(Path(args.data_root).expanduser() if args.data_root else data_root())
    return cfg


def hole_metrics(model, dataset, cfg, stats, device) -> dict:
    """
    Hole-region error and uncertainty calibration on the held-out split.

    Everything is measured in metres on the cells the model had to invent:
    unobserved by the cameras but covered by valid ground truth.
    """
    model.eval()
    mean, std = stats['mean'], stats['std']
    size = 251
    pad = (cfg['pad_to'] - size) // 2
    crop = slice(pad, pad + size)

    sq_err, n_px, covered, errs, sigmas = 0.0, 0, 0, [], []
    loader = torch.utils.data.DataLoader(dataset, batch_size=max(cfg['batch_size'] // 2, 4),
                                         shuffle=False, num_workers=cfg['num_workers'])

    for inp, gt, gt_mask in loader:
        inp, gt, gt_mask = inp.to(device), gt.to(device), gt_mask.to(device)
        pred, sigma = nll_predict(model, inp, cfg.get('uncertainty', False))

        hole = ((gt_mask > 0.5) & (inp[:, 1:2] < 0.5))[:, 0, crop, crop].cpu().numpy()
        err = ((pred - gt)[:, 0, crop, crop].cpu().numpy() * std)
        sq_err += float((err[hole] ** 2).sum())
        n_px += int(hole.sum())

        if sigma is not None:
            sig = sigma[:, 0, crop, crop].cpu().numpy() * std
            a, s = np.abs(err[hole]), sig[hole]
            covered += int((a <= s).sum())
            if len(a) > MIN_HOLE_PX:
                errs.append(a[::200])
                sigmas.append(s[::200])

    out = {'hole_rmse_m': float(np.sqrt(sq_err / max(n_px, 1))), 'hole_pixels': n_px}
    if errs:
        a, s = np.concatenate(errs), np.concatenate(sigmas)
        out['coverage_1sigma'] = covered / max(n_px, 1)
        out['corr_sigma_error'] = float(np.corrcoef(a, s)[0, 1])
    return out


def main():
    args = parse_args()
    seed_everything(args.seed)
    cfg = load_config(args)
    device = torch.device(args.device)

    out_dir = Path(args.out) if args.out else \
        runs_dir() / f"{args.name}_{datetime.now():%Y%m%d_%H%M%S}"
    out_dir.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, test_ds, stats, _ = get_dataloaders(cfg)
    if args.limit:
        test_ds = torch.utils.data.Subset(test_ds, range(min(args.limit, len(test_ds))))
    print(f'data      : {cfg["data_dir"]}')
    print(f'samples   : {len(train_loader.dataset)} train / '
          f'{len(val_loader.dataset)} val / {len(test_ds)} test')

    model = build_model(cfg).to(device)
    print(f'model     : U-Net ResNet-34, {count_parameters(model) / 1e6:.2f}M parameters')

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['lr'],
                                  weight_decay=cfg['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg['epochs'], eta_min=cfg['lr'] * 0.01)
    loss_fn = ElevationLoss(cfg['w_valid'], cfg['w_hole'], cfg['uncertainty_beta'])

    use_amp = cfg.get('amp', True) and device.type == 'cuda'
    scaler = GradScaler('cuda') if use_amp else None
    amp_dtype = torch.float16 if use_amp else None

    warmup = cfg.get('uncertainty_warmup', 0)
    print(f'schedule  : {warmup} epochs L1 warmup, then beta-NLL '
          f'(beta={cfg["uncertainty_beta"]}) to epoch {cfg["epochs"]}\n')

    best = float('inf')
    for epoch in range(1, cfg['epochs'] + 1):
        # the log-variance head is only supervised after the warmup; before it
        # the loss falls back to plain masked L1
        unc = cfg['uncertainty'] and epoch > warmup

        t0 = time.time()
        tr_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, scaler,
                                  device, None, epoch, cfg['grad_clip'], unc, amp_dtype)
        val = evaluate(model, val_loader, loss_fn, device, stats, unc)
        scheduler.step()

        tag = 'beta-NLL' if unc else 'L1      '
        print(f'epoch {epoch:3d}/{cfg["epochs"]}  {tag}  '
              f'train {tr_loss:7.4f}  val {val["loss"]:7.4f}  '
              f'RMSE {val.get("rmse", float("nan")):6.3f} m  ({time.time() - t0:.0f}s)',
              flush=True)

        state = make_state(epoch, model, optimizer, scheduler, scaler, best, stats, cfg)
        save_checkpoint(state, out_dir / 'last.pth')
        if val.get('rmse', float('inf')) < best:
            best = val['rmse']
            save_checkpoint(state, out_dir / 'best.pth')

    print('\nheld-out split, hole cells only:')
    ckpt = torch.load(out_dir / 'best.pth', map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model'])
    results = hole_metrics(model, test_ds, cfg, stats, device)
    for k, v in results.items():
        print(f'  {k:18s} {v:.4f}' if isinstance(v, float) else f'  {k:18s} {v}')

    json.dump({'config': cfg, 'stats': stats, 'test': results},
              open(out_dir / 'results.json', 'w'), indent=2)
    print(f'\nSaved {out_dir}')


if __name__ == '__main__':
    main()
