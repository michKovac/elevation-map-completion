"""
Stage-2 training: fit only the log_var_head on a frozen backbone.

Loads a checkpoint trained with plain L1 (uncertainty=False), rebuilds the model
with uncertainty=True (adds log_var_head), loads matching weights via strict=False,
freezes everything except log_var_head, then trains the head with Laplacian NLL
where the prediction is detached — making the loss a well-posed regression:

    log_var* = log(|pred_fixed - gt|)

No gradient instability because the prediction is fixed.

Heteroscedastic aleatoric uncertainty framework:
    Kendall & Gal, "What Uncertainties Do We Need in Bayesian Deep Learning
    for Computer Vision?", NeurIPS 2017. https://arxiv.org/abs/1703.04977

Usage:
    python train_uncertainty_head.py --checkpoint runs/unet_xxx/best.pth
    python train_uncertainty_head.py --checkpoint runs/attunet_xxx/best.pth --epochs 100
"""
import sys
from pathlib import Path
import argparse
import json
import os
import time
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.amp import GradScaler, autocast

from elevcomp.dataset import get_dataloaders
from elevcomp.model import build_model


# ── Loss ──────────────────────────────────────────────────────────────────────

def nll_loss(log_var, pred_detached, gt, gt_mask, partial_mask, w_hole):
    """
    Laplacian NLL for the log_var head only. pred is already detached.
    loss = exp(-log_var) * |pred - gt| + log_var   (per pixel, masked)
    """
    diff = (pred_detached - gt).abs()
    lv   = log_var.clamp(-10.0, 10.0)
    term = torch.exp(-lv) * diff + lv

    n_valid = gt_mask.sum().clamp(min=1)
    loss    = (term * gt_mask).sum() / n_valid

    if w_hole > 0 and partial_mask is not None:
        hole_mask = gt_mask * (1.0 - partial_mask)
        n_hole    = hole_mask.sum().clamp(min=1)
        loss      = loss + w_hole * (term * hole_mask).sum() / n_hole

    return loss


# ── Visualization ─────────────────────────────────────────────────────────────

@torch.no_grad()
def save_vis(model, val_loader, device, stats, epoch, vis_dir, n_vis=8):
    model.eval()
    dataset = val_loader.dataset
    indices = torch.randperm(len(dataset))[:min(n_vis, len(dataset))].tolist()
    samples = [dataset[i] for i in indices]

    inp_t = torch.stack([s[0] for s in samples]).to(device)
    gt_t  = torch.stack([s[1] for s in samples]).to(device)
    gm_t  = torch.stack([s[2] for s in samples]).to(device)

    out      = model(inp_t)
    pred_t   = out[:, :1]
    log_var_t = out[:, 1:2]

    mean_s, std_s = stats['mean'], stats['std']

    def denorm(x):
        return x * std_s + mean_s

    n = inp_t.shape[0]
    fig, axes = plt.subplots(n, 5, figsize=(18, 3.2 * n), constrained_layout=True)
    if n == 1:
        axes = axes[None]

    for col, title in enumerate(['Partial', 'Predicted', 'GT', '|Error| [m]', 'σ [m]']):
        axes[0, col].set_title(title, fontsize=10, fontweight='bold')

    for i in range(n):
        pm  = inp_t[i, 1].cpu().numpy()
        pe  = denorm(inp_t[i, 0].cpu().numpy())
        pr  = denorm(pred_t[i, 0].cpu().numpy())
        gte = denorm(gt_t[i, 0].cpu().numpy())
        gm  = gm_t[i, 0].cpu().numpy().astype(bool)
        sig = np.exp(0.5 * log_var_t[i, 0].cpu().numpy()) * std_s

        partial_vis = np.where(pm > 0.5, pe, np.nan)
        err = np.abs(pr - gte)
        err[~gm] = np.nan
        sig[~gm] = np.nan

        valid = gte[gm] if gm.any() else gte.ravel()
        vmin  = float(np.nanpercentile(valid, 1))
        vmax  = float(np.nanpercentile(valid, 99))
        emax  = float(np.nanpercentile(err[gm], 95)) if gm.any() else 1.0
        smax  = float(np.nanpercentile(sig[gm], 95)) if gm.any() else 1.0

        for j, (data, cmap, lo, hi) in enumerate([
            (partial_vis, 'terrain', vmin, vmax),
            (pr,          'terrain', vmin, vmax),
            (gte,         'terrain', vmin, vmax),
            (err,         'hot',     0.0,  emax),
            (sig,         'plasma',  0.0,  smax),
        ]):
            im = axes[i, j].imshow(data, cmap=cmap, vmin=lo, vmax=hi, origin='lower')
            axes[i, j].axis('off')
            plt.colorbar(im, ax=axes[i, j], fraction=0.046, pad=0.02)

        axes[i, 0].set_ylabel(f'#{i}  {pm.mean()*100:.0f}%', fontsize=8)

    fig.suptitle(f'Uncertainty head — epoch {epoch + 1}', fontsize=11)
    vis_dir.mkdir(exist_ok=True)
    path = vis_dir / f'epoch_{epoch + 1:04d}.png'
    fig.savefig(path, dpi=100, bbox_inches='tight')
    plt.close(fig)
    return path


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--epochs',     type=int,   default=50)
    p.add_argument('--lr',         type=float, default=1e-3)
    p.add_argument('--w_hole',     type=float, default=1.0)
    p.add_argument('--vis_every',  type=int,   default=10)
    p.add_argument('--n_vis',      type=int,   default=8)
    p.add_argument('--amp',        action='store_true', default=True)
    args = p.parse_args()

    ckpt_path = Path(args.checkpoint)
    run_dir   = ckpt_path.parent

    # ── Config + stats ────────────────────────────────────────────────────────
    with open(run_dir / 'config.json') as f:
        cfg = json.load(f)
    with open(run_dir / 'stats.json') as f:
        stats = json.load(f)

    # Force uncertainty=True so model gets log_var_head
    cfg['uncertainty'] = False   # build base first to load weights
    cfg['resume']      = None

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ── Data ─────────────────────────────────────────────────────────────────
    train_loader, val_loader, _, _, _ = get_dataloaders(cfg, rank=0, world_size=1)
    print(f'Train: {len(train_loader.dataset)}  Val: {len(val_loader.dataset)}')

    # ── Model: load backbone weights, then add log_var_head ───────────────────
    # Step 1: build without uncertainty, load checkpoint
    base_model = build_model(cfg).to(device)
    ckpt       = torch.load(ckpt_path, map_location=device)
    # Support both train.py format ({'model': ...}) and raw state-dict format
    raw_sd = ckpt['model'] if isinstance(ckpt, dict) and 'model' in ckpt else ckpt
    # strict=False: ignore any extra uncertainty keys the checkpoint may already have
    base_model.load_state_dict(raw_sd, strict=False)
    base_state = base_model.state_dict()

    # Step 2: build with uncertainty=True (adds log_var_head)
    cfg['uncertainty'] = True
    model = build_model(cfg).to(device)
    # For attunet the segmentation head changes shape (classes 1→2), so exclude it
    # from base_state; the new head is properly initialised by build_model.
    if cfg.get('model') == 'attunet':
        base_state = {k: v for k, v in base_state.items()
                      if 'segmentation_head' not in k}
    # Load backbone weights; log_var_head stays at init (zeros weight, bias=-5)
    missing, unexpected = model.load_state_dict(base_state, strict=False)
    print(f'Loaded backbone. Missing (log_var_head): {missing}')

    # Step 3: freeze everything except log_var_head
    n_frozen = 0
    for name, param in model.named_parameters():
        if 'log_var_head' not in name:
            param.requires_grad = False
            n_frozen += param.numel()
        else:
            param.requires_grad = True
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Frozen: {n_frozen:,}  Trainable (log_var_head): {n_trainable:,}')

    # ── Optimizer ─────────────────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01,
    )
    scaler = GradScaler('cuda') if args.amp else None

    out_dir = run_dir / 'uncertainty_head'
    out_dir.mkdir(exist_ok=True)
    vis_dir = out_dir / 'vis'

    print(f'\nTraining log_var_head for {args.epochs} epochs → {out_dir}\n')

    best_loss = float('inf')
    history   = []

    for epoch in range(args.epochs):
        model.train()
        t0 = time.time()
        total_loss = 0.0

        for inp, gt, gt_mask in train_loader:
            inp, gt, gt_mask = (
                inp.to(device, non_blocking=True),
                gt.to(device, non_blocking=True),
                gt_mask.to(device, non_blocking=True),
            )
            partial_mask = inp[:, 1:2]

            optimizer.zero_grad(set_to_none=True)
            with autocast('cuda', enabled=(scaler is not None)):
                out     = model(inp)
                pred    = out[:, :1].detach()   # frozen — no gradient to backbone
                log_var = out[:, 1:2]
                loss    = nll_loss(log_var, pred, gt, gt_mask, partial_mask, args.w_hole)

            if scaler:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    filter(lambda p: p.requires_grad, model.parameters()), 1.0
                )
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    filter(lambda p: p.requires_grad, model.parameters()), 1.0
                )
                optimizer.step()

            total_loss += loss.item()

        # ── Val loss ──────────────────────────────────────────────────────────
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inp, gt, gt_mask in val_loader:
                inp, gt, gt_mask = (
                    inp.to(device, non_blocking=True),
                    gt.to(device, non_blocking=True),
                    gt_mask.to(device, non_blocking=True),
                )
                out     = model(inp)
                pred    = out[:, :1].detach()
                log_var = out[:, 1:2]
                val_loss += nll_loss(log_var, pred, gt, gt_mask, inp[:, 1:2], args.w_hole).item()

        scheduler.step()

        train_l = total_loss / max(len(train_loader), 1)
        val_l   = val_loss   / max(len(val_loader), 1)
        elapsed = time.time() - t0
        lr_now  = scheduler.get_last_lr()[0]

        print(f'Ep {epoch+1:03d}/{args.epochs} | train {train_l:.4f} | val {val_l:.4f} | lr {lr_now:.2e} | {elapsed:.1f}s')
        history.append({'epoch': epoch, 'train_loss': train_l, 'val_loss': val_l})

        # Save best — wrapped in {'model': ...} so tta_inference.py can load it
        if val_l < best_loss:
            best_loss = val_l
            torch.save({'model': model.state_dict()}, out_dir / 'best.pth')
            print(f'  ★ New best val loss: {best_loss:.4f}')

        # Visualize
        if (epoch + 1) % args.vis_every == 0 or epoch == 0:
            png = save_vis(model, val_loader, device, stats, epoch, vis_dir, args.n_vis)
            print(f'  → vis: {png}')

        with open(out_dir / 'history.json', 'w') as f:
            json.dump(history, f, indent=2)

    # Save final full model (backbone + log_var_head)
    torch.save({'model': model.state_dict()}, out_dir / 'last.pth')
    print(f'\nDone. Best val NLL: {best_loss:.4f}')
    print(f'Checkpoint with uncertainty: {out_dir}/best.pth')


if __name__ == '__main__':
    main()
