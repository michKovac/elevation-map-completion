#!/usr/bin/env python3
"""
Train PConv-UNet / SimpleUNet / AttentionUNet for elevation map completion
on a SINGLE dataset directory (configs/single_run.toml: data_dir). For the 5-fold
cross-environment experiment used in the paper, see run_cv_experiment.py.

Heteroscedastic aleatoric uncertainty (optional, uncertainty=true in config):
    Kendall & Gal, "What Uncertainties Do We Need in Bayesian Deep Learning
    for Computer Vision?", NeurIPS 2017. https://arxiv.org/abs/1703.04977

β-NLL loss (prevents variance collapse, activated after warmup epochs):
    Seitzer et al., "On the Pitfalls of Heteroscedastic Uncertainty Estimation
    with Probabilistic Neural Networks", ICLR 2022.
    https://arxiv.org/abs/2203.09168  |  https://github.com/martius-lab/beta-nll

Single GPU:
    python train.py
    python train.py --model unet --epochs 100

Multi-GPU DDP (2× RTX 5090):
    torchrun --nproc_per_node=2 train.py

Resume from checkpoint:
    python train.py --resume runs/pconv_20240622_143022/last.pth

All keys in configs/single_run.toml can be overridden from the command line.
"""
import json
import os
import time
import tomllib
from datetime import datetime
from pathlib import Path
from elevcomp.paths import CONFIG_DIR

import torch
import torch.distributed as dist
from torch.amp import GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter

from elevcomp.dataset import get_dataloaders
from elevcomp.figures import save_val_images
from elevcomp.losses import ElevationLoss
from elevcomp.model import build_model, count_parameters
from elevcomp.training import (evaluate, make_state, save_checkpoint,
                           train_one_epoch)


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

def load_config(cli_args=None) -> dict:
    """
    Load configs/single_run.toml, then apply any CLI overrides.
    Non-null CLI values always win over configs/single_run.toml defaults.
    """
    cfg_path = CONFIG_DIR / 'single_run.toml'
    with open(cfg_path, 'rb') as f:
        cfg = tomllib.load(f)

    # Empty string resume means no resume (TOML has no null)
    if cfg.get('resume') == '':
        cfg['resume'] = None

    if cli_args:
        for k, v in vars(cli_args).items():
            if v is not None:
                cfg[k] = v

    return cfg


def parse_cli():
    import argparse
    p = argparse.ArgumentParser(add_help=True)
    p.add_argument('--data_dir')
    p.add_argument('--model',         choices=['pconv', 'unet', 'attunet', 'unet_pytorch',
                                               'segformer'])
    p.add_argument('--layer_size',    type=int)
    p.add_argument('--base_channels', type=int)
    p.add_argument('--epochs',        type=int)
    p.add_argument('--batch_size',    type=int)
    p.add_argument('--lr',            type=float)
    p.add_argument('--weight_decay',  type=float)
    p.add_argument('--grad_clip',     type=float)
    p.add_argument('--w_valid',       type=float)
    p.add_argument('--w_hole',        type=float)
    p.add_argument('--save_dir')
    p.add_argument('--save_every',    type=int)
    p.add_argument('--num_workers',   type=int)
    p.add_argument('--resume')
    p.add_argument('--no_amp',        action='store_true', default=False)
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# DDP helpers
# ─────────────────────────────────────────────────────────────────────────────

def setup_ddp():
    rank       = int(os.environ.get('LOCAL_RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    if world_size > 1:
        dist.init_process_group(backend='nccl')
    torch.cuda.set_device(rank)
    return rank, world_size


def cleanup_ddp(world_size: int):
    if world_size > 1:
        dist.destroy_process_group()


def is_main(rank: int) -> bool:
    return rank == 0


def reduce_metrics(metrics: dict, world_size: int) -> dict:
    if world_size == 1:
        return metrics
    for k, v in metrics.items():
        t = torch.tensor(v, device='cuda')
        dist.all_reduce(t, op=dist.ReduceOp.AVG)
        metrics[k] = float(t)
    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_cli()
    cfg  = load_config(args)
    if args.no_amp:
        cfg['amp'] = False

    rank, world_size = setup_ddp()
    device = torch.device(f'cuda:{rank}')

    # ── Run directory ─────────────────────────────────────────────────────────
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir   = Path(cfg['save_dir']) / f'{cfg["model"]}_{timestamp}'
    vis_dir   = run_dir / 'vis'
    if is_main(rank):
        run_dir.mkdir(parents=True, exist_ok=True)
        print(f'\n{"="*60}')
        print(f'Run dir  : {run_dir}')
        print(f'Model    : {cfg["model"]}  |  layer_size: {cfg.get("layer_size", 6)}')
        print(f'Epochs   : {cfg["epochs"]}  |  batch: {cfg["batch_size"]}  |  lr: {cfg["lr"]}')
        _unc = cfg.get('uncertainty', False)
        _wu  = cfg.get('uncertainty_warmup', 0) if _unc else 0
        unc_str = f'on (warmup {_wu} ep)' if _unc else 'off'
        print(f'AMP      : {cfg["amp"]}  |  GPUs: {world_size}  |  Uncertainty: {unc_str}')
        print(f'{"="*60}\n')

    # ── Data ──────────────────────────────────────────────────────────────────
    train_loader, val_loader, _, stats, sampler = get_dataloaders(cfg, rank, world_size)
    if is_main(rank):
        print(f'Train batches : {len(train_loader)}  (samples: {len(train_loader.dataset)})')
        print(f'Val   batches : {len(val_loader)}  (samples: {len(val_loader.dataset)})\n')

    # ── Model ─────────────────────────────────────────────────────────────────
    model = build_model(cfg).to(device)
    if world_size > 1:
        find_unused = cfg['model'] in ('attunet', 'unet_pytorch', 'segformer',
                                       'segformer')
        model = DDP(model, device_ids=[rank], find_unused_parameters=find_unused)

    if is_main(rank):
        raw = model.module if world_size > 1 else model
        print(f'Parameters : {count_parameters(raw):,}\n')

    # ── Optimizer / scheduler / scaler ───────────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg['epochs'], eta_min=cfg['lr'] * 0.01,
    )
    # AMP precision: fp16 + scaler (default) | bf16 (no scaler, wider exponent
    # range) | off (fp32).
    _amp_on = cfg.get('amp', True)
    _amp_str = str(cfg.get('amp_dtype', 'fp16')).lower()
    if not _amp_on:
        autocast_dtype, scaler = None, None
    elif _amp_str in ('bf16', 'bfloat16'):
        autocast_dtype, scaler = torch.bfloat16, None
    else:
        autocast_dtype, scaler = torch.float16, GradScaler('cuda')
    uncertainty      = cfg.get('uncertainty', False)
    unc_warmup       = cfg.get('uncertainty_warmup', 0) if uncertainty else 0
    unc_beta         = cfg.get('uncertainty_beta', 0.5)
    loss_fn          = ElevationLoss(cfg['w_valid'], cfg['w_hole'], beta=unc_beta)

    # ── Resume ────────────────────────────────────────────────────────────────
    start_epoch = 0
    best_rmse   = float('inf')
    history     = []

    if cfg.get('resume'):
        ckpt = torch.load(cfg['resume'], map_location=device)
        raw  = model.module if isinstance(model, DDP) else model
        raw.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        if scaler is not None and 'scaler' in ckpt:
            scaler.load_state_dict(ckpt['scaler'])
        start_epoch = ckpt['epoch'] + 1
        best_rmse   = ckpt.get('best_rmse', float('inf'))
        if is_main(rank):
            print(f'Resumed from epoch {ckpt["epoch"]}  (best RMSE: {best_rmse:.4f} m)\n')

    # ── Persist config + stats ────────────────────────────────────────────────
    if is_main(rank):
        with open(run_dir / 'config.json', 'w') as f:
            json.dump(cfg, f, indent=2)
        with open(run_dir / 'stats.json', 'w') as f:
            json.dump(stats, f, indent=2)
        writer = SummaryWriter(run_dir / 'tb')

    # ── Training loop ─────────────────────────────────────────────────────────
    for epoch in range(start_epoch, cfg['epochs']):
        t0 = time.time()

        use_unc = uncertainty and (epoch >= unc_warmup)

        train_loss = train_one_epoch(
            model, train_loader, optimizer, loss_fn,
            scaler, device, sampler, epoch, cfg['grad_clip'], use_unc,
            autocast_dtype,
        )
        val_m = evaluate(model, val_loader, loss_fn, device, stats, use_unc)

        if world_size > 1:
            val_m = reduce_metrics(val_m, world_size)

        scheduler.step()

        if is_main(rank):
            lr_now  = scheduler.get_last_lr()[0]
            elapsed = time.time() - t0

            row = {'epoch': epoch, 'train_loss': train_loss,
                   'lr': lr_now, 'time_s': elapsed, **val_m}
            history.append(row)

            hole_str = (
                f"  rmse_hole {val_m['rmse_hole']:.3f}m"
                if 'rmse_hole' in val_m else ''
            )
            print(
                f"Ep {epoch+1:03d}/{cfg['epochs']} | "
                f"train {train_loss:.4f} | "
                f"val rmse {val_m['rmse']:.3f}m  mae {val_m['mae']:.3f}m"
                f"{hole_str} | "
                f"ssim {val_m.get('ssim', 0.0):.3f} | "
                f"lr {lr_now:.2e} | {elapsed:.1f}s"
            )

            writer.add_scalar('train/loss', train_loss, epoch)
            writer.add_scalar('train/lr',   lr_now,     epoch)
            for k, v in val_m.items():
                writer.add_scalar(f'val/{k}', v, epoch)

            new_best = val_m['rmse'] < best_rmse
            if new_best:
                best_rmse = val_m['rmse']

            state = make_state(epoch, model, optimizer, scheduler, scaler, best_rmse, stats, cfg)
            save_checkpoint(state, run_dir / 'last.pth')

            if new_best:
                save_checkpoint(state, run_dir / 'best.pth')
                print(f'  ★ New best RMSE: {best_rmse:.4f} m  → saved best.pth')

            if (epoch + 1) % cfg['save_every'] == 0:
                save_checkpoint(state, run_dir / f'epoch_{epoch+1:03d}.pth')

            vis_every = cfg.get('vis_every', 25)
            n_vis     = cfg.get('n_vis', 8)
            if epoch == unc_warmup and unc_warmup > 0:
                print(f'  ★ Warmup complete — uncertainty (NLL) activated from epoch {epoch+1}')

            if (epoch + 1) % vis_every == 0 or epoch == 0:
                png = save_val_images(
                    model, val_loader, device, stats,
                    epoch, vis_dir, n_vis, use_unc, writer,
                )
                print(f'  → vis saved: {png}')

            with open(run_dir / 'history.json', 'w') as f:
                json.dump(history, f, indent=2)

    # ── Done ──────────────────────────────────────────────────────────────────
    if is_main(rank):
        writer.close()
        print(f'\n{"="*60}')
        print(f'Training complete.')
        print(f'Best val RMSE : {best_rmse:.4f} m')
        print(f'Run dir       : {run_dir}')
        print(f'{"="*60}\n')

    cleanup_ddp(world_size)


if __name__ == '__main__':
    main()
