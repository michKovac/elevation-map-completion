"""
Shared matplotlib figures (informative / diagnostic — paper-final figures are
rebuilt from the persisted per-sample predictions and metrics).

Conventions:
    elevation panels : 'terrain' colormap, robust [p1, p99] range from GT
    |error| panels   : 'hot',    0 … p95
    σ panels         : 'plasma', 0 … p95
    method colours   : Okabe & Ito colourblind-safe pair (β-NLL vs TTA)
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from .calibration import SPARS_FRACS
from .inference import nll_predict, split_output, tta_predict

# Okabe & Ito colourblind-safe pair used in all comparison figures.
C_NLL = '#0072B2'   # blue   — β-NLL head
C_TTA = '#E69F00'   # orange — TTA ensemble
C_REF = '#888888'   # neutral reference (oracle / diagonal)


# ─────────────────────────────────────────────────────────────────────────────
# Training-progress grid (called periodically from the training loop)
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def save_val_images(model, val_loader, device, stats, epoch, vis_dir, n_vis,
                    uncertainty=False, writer=None):
    """
    Grab n_vis random val samples, run inference, save a grid PNG.
    Columns: Partial | Predicted | Composite | GT | |Error| pred | |Error| comp [| σ]
    """
    model.eval()

    dataset = val_loader.dataset
    n_total = len(dataset)
    indices = torch.randperm(n_total)[:min(n_vis, n_total)].tolist()
    samples = [dataset[i] for i in indices]

    inp_t = torch.stack([s[0] for s in samples]).to(device)
    gt_t  = torch.stack([s[1] for s in samples]).to(device)
    gm_t  = torch.stack([s[2] for s in samples]).to(device)

    out_t = model(inp_t)
    pred_t, lv_t = split_output(out_t, uncertainty)

    mean, std = stats['mean'], stats['std']

    def denorm(x):
        return x * std + mean

    n_cols = 7 if uncertainty else 6
    n      = inp_t.shape[0]
    fig, axes = plt.subplots(n, n_cols, figsize=(3.5 * n_cols, 3.2 * n), constrained_layout=True)
    if n == 1:
        axes = axes[None]

    col_titles = [
        'Partial (stereo)', 'Predicted', 'Composite',
        'GT (simulation)', '|Error| pred [m]', '|Error| comp [m]',
    ]
    if uncertainty:
        col_titles.append('Uncertainty σ [m]')
    for col, title in enumerate(col_titles):
        axes[0, col].set_title(title, fontsize=9, fontweight='bold')

    for i in range(n):
        pm  = inp_t[i, 1].cpu().numpy()
        pe  = denorm(inp_t[i, 0].cpu().numpy())
        pr  = denorm(pred_t[i, 0].cpu().numpy())
        gte = denorm(gt_t[i, 0].cpu().numpy())
        gm  = gm_t[i, 0].cpu().numpy().astype(bool)

        # Composite: keep stereo where measured, fill holes with prediction
        comp = np.where(pm > 0.5, pe, pr)

        partial_vis = np.where(pm > 0.5, pe, np.nan)
        err_pred = np.abs(pr   - gte)
        err_comp = np.abs(comp - gte)
        err_pred[~gm] = np.nan
        err_comp[~gm] = np.nan

        valid_vals = gte[gm] if gm.any() else gte.ravel()
        vmin = float(np.nanpercentile(valid_vals, 1))
        vmax = float(np.nanpercentile(valid_vals, 99))
        emax = float(np.nanpercentile(
            np.concatenate([err_pred[gm], err_comp[gm]]), 95
        )) if gm.any() else 1.0

        panels = [
            (partial_vis, 'terrain', vmin, vmax),
            (pr,          'terrain', vmin, vmax),
            (comp,        'terrain', vmin, vmax),
            (gte,         'terrain', vmin, vmax),
            (err_pred,    'hot',     0.0,  emax),
            (err_comp,    'hot',     0.0,  emax),
        ]

        if uncertainty:
            # σ = exp(0.5 * log_var), converted to metres
            sigma = np.exp(0.5 * lv_t[i, 0].cpu().numpy()) * std
            sigma[~gm] = np.nan
            smax = float(np.nanpercentile(sigma[gm], 95)) if gm.any() else 1.0
            panels.append((sigma, 'plasma', 0.0, smax))

        for j, (data, cmap, lo, hi) in enumerate(panels):
            im = axes[i, j].imshow(data, cmap=cmap, vmin=lo, vmax=hi, origin='lower')
            axes[i, j].axis('off')
            plt.colorbar(im, ax=axes[i, j], fraction=0.046, pad=0.02)

        cov      = float(pm.mean() * 100)
        mae_pred = float(np.nanmean(err_pred))
        mae_comp = float(np.nanmean(err_comp))
        axes[i, 0].set_ylabel(
            f'#{i}  {cov:.0f}% cov\npred {mae_pred:.3f}m\ncomp {mae_comp:.3f}m',
            fontsize=7, rotation=0, labelpad=72, va='center',
        )

    fig.suptitle(f'Epoch {epoch + 1}  —  validation samples', fontsize=11)

    vis_dir.mkdir(exist_ok=True)
    png_path = vis_dir / f'epoch_{epoch + 1:04d}.png'
    fig.savefig(png_path, dpi=100, bbox_inches='tight')
    plt.close(fig)

    if writer is not None:
        img = plt.imread(str(png_path))
        img_t = torch.from_numpy(img[..., :3]).permute(2, 0, 1).unsqueeze(0)
        writer.add_images('val/predictions', img_t, epoch, dataformats='NCHW')

    return png_path


# ─────────────────────────────────────────────────────────────────────────────
# Uncertainty comparison grid: β-NLL head vs D4 TTA ensemble
# ─────────────────────────────────────────────────────────────────────────────

def visualize_uncertainty(model, dataset, stats, indices, device,
                          has_uncertainty, out_path):
    """
    Per-sample rows comparing both uncertainty methods.
    Columns (8, or 6 if the model has no β-NLL head):
        Partial | GT | β-NLL pred | |Err| NLL | σ NLL | TTA pred | |Err| TTA | σ TTA
    """
    mean_s, std_s = stats['mean'], stats['std']

    def denorm(x):
        return x * std_s + mean_s

    n      = len(indices)
    n_cols = 8 if has_uncertainty else 6

    if has_uncertainty:
        col_titles = [
            'Partial (stereo)', 'GT',
            'Pred (β-NLL)', '|Error| NLL [m]', 'σ NLL [m]',
            'Pred (TTA)',   '|Error| TTA [m]', 'σ TTA [m]',
        ]
    else:
        col_titles = [
            'Partial (stereo)', 'GT',
            'Pred', '|Error| [m]',
            'Pred (TTA)', 'σ TTA [m]',
        ]

    fig, axes = plt.subplots(
        n, n_cols,
        figsize=(n_cols * 2.6, n * 3.2),
        constrained_layout=True,
    )
    if n == 1:
        axes = axes[None]

    for col_i, title in enumerate(col_titles):
        axes[0, col_i].set_title(title, fontsize=8, fontweight='bold')

    for row, idx in enumerate(indices):
        sample  = dataset[idx]
        inp     = sample[0].unsqueeze(0).to(device)
        gt_t    = sample[1].unsqueeze(0)
        gm_t    = sample[2].unsqueeze(0)

        pred_nll, sigma_nll = nll_predict(model, inp, has_uncertainty)
        pred_tta, sigma_tta = tta_predict(model, inp)

        pm     = inp[0, 1].cpu().numpy()
        pe     = denorm(inp[0, 0].cpu().numpy())
        pr_nll = denorm(pred_nll[0, 0].cpu().numpy())
        pr_tta = denorm(pred_tta[0, 0].cpu().numpy())
        gte    = denorm(gt_t[0, 0].numpy())
        gm_np  = gm_t[0, 0].numpy().astype(bool)

        sig_tta = sigma_tta[0, 0].cpu().numpy() * std_s
        sig_tta[~gm_np] = np.nan

        err_nll = np.abs(pr_nll - gte)
        err_tta = np.abs(pr_tta - gte)
        err_nll[~gm_np] = np.nan
        err_tta[~gm_np] = np.nan

        if has_uncertainty:
            sig_nll = sigma_nll[0, 0].cpu().numpy() * std_s
            sig_nll[~gm_np] = np.nan

        # Color ranges — shared across both methods for fair comparison
        valid_gt = gte[gm_np] if gm_np.any() else gte.ravel()
        vmin = float(np.nanpercentile(valid_gt, 1))
        vmax = float(np.nanpercentile(valid_gt, 99))

        err_vals = np.concatenate([err_nll[gm_np], err_tta[gm_np]]) if gm_np.any() else np.array([1.0])
        emax = float(np.nanpercentile(err_vals, 95))

        sig_parts = [sig_tta[gm_np]] if gm_np.any() else [np.array([1.0])]
        if has_uncertainty and gm_np.any():
            sig_parts.append(sig_nll[gm_np])
        smax = float(np.nanpercentile(np.concatenate(sig_parts), 95))

        partial_vis = np.where(pm > 0.5, pe, np.nan)

        if has_uncertainty:
            panels = [
                (partial_vis, 'terrain', vmin, vmax),
                (gte,         'terrain', vmin, vmax),
                (pr_nll,      'terrain', vmin, vmax),
                (err_nll,     'hot',     0.0,  emax),
                (sig_nll,     'plasma',  0.0,  smax),
                (pr_tta,      'terrain', vmin, vmax),
                (err_tta,     'hot',     0.0,  emax),
                (sig_tta,     'plasma',  0.0,  smax),
            ]
        else:
            panels = [
                (partial_vis, 'terrain', vmin, vmax),
                (gte,         'terrain', vmin, vmax),
                (pr_nll,      'terrain', vmin, vmax),
                (err_nll,     'hot',     0.0,  emax),
                (pr_tta,      'terrain', vmin, vmax),
                (sig_tta,     'plasma',  0.0,  smax),
            ]

        for col_i, (data, cmap, lo, hi) in enumerate(panels):
            im = axes[row, col_i].imshow(data, cmap=cmap, vmin=lo, vmax=hi, origin='lower')
            axes[row, col_i].axis('off')
            plt.colorbar(im, ax=axes[row, col_i], fraction=0.046, pad=0.02)

        axes[row, 0].set_ylabel(f'#{idx}', fontsize=8)

    title_str = 'Uncertainty: β-NLL head vs TTA (8× D4)' if has_uncertainty else 'Uncertainty: TTA (8× D4)'
    fig.suptitle(title_str, fontsize=11, fontweight='bold')
    fig.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {out_path}')


# ─────────────────────────────────────────────────────────────────────────────
# Per-fold diagnostic figures (CV experiment)
# ─────────────────────────────────────────────────────────────────────────────

def fig_learning_curves(fold_dir: Path, out: Path):
    """Train loss + validation RMSE curves from history.json."""
    hist = json.load(open(fold_dir / 'history.json'))
    ep = [r['epoch'] + 1 for r in hist]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)

    a1.plot(ep, [r['train_loss'] for r in hist], color=C_REF, lw=2)
    a1.set(xlabel='epoch', ylabel='train loss', title='Training loss')

    a2.plot(ep, [r['rmse'] for r in hist], color=C_NLL, lw=2, label='val RMSE')
    if 'rmse_hole' in hist[-1]:
        a2.plot(ep, [r.get('rmse_hole', np.nan) for r in hist],
                color=C_TTA, lw=2, label='val RMSE (hole)')
    a2.set(xlabel='epoch', ylabel='RMSE [m]', title='Validation error')
    a2.legend(frameon=False)

    for ax in (a1, a2):
        ax.grid(alpha=0.25)
        ax.spines[['top', 'right']].set_visible(False)
    fig.savefig(out, dpi=200)
    plt.close(fig)


def fig_calibration(calibration: dict, pix: dict, uncertainty: bool, out: Path):
    """Sparsification curves + σ-decile reliability for both σ methods."""
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    fr = np.array(calibration['sparsification_fractions'])

    oracle = calibration['tta']['oracle_mean_curve']
    if oracle:
        a1.plot(fr, oracle, color=C_REF, lw=2,
                ls='--', label='oracle (sorted by |error|, TTA)')
    if uncertainty and calibration['nll']['mean_curve']:
        a1.plot(fr, calibration['nll']['mean_curve'], color=C_NLL, lw=2,
                label='β-NLL σ')
    if calibration['tta']['mean_curve']:
        a1.plot(fr, calibration['tta']['mean_curve'], color=C_TTA, lw=2,
                label='TTA σ')
    a1.set_xlabel('fraction of most-uncertain pixels removed', fontsize=15)
    a1.set_ylabel('normalized RMSE (hole)', fontsize=15)
    a1.set_title('Sparsification (mean over test samples)', fontsize=16)
    a1.legend(frameon=False, fontsize=13)

    def reliability(sig, err, color, label):
        if sig.size < 100:
            return
        sig, err = sig.astype(np.float64), err.astype(np.float64)
        qs = np.quantile(sig, np.linspace(0, 1, 11))
        xs, ys = [], []
        for lo, hi in zip(qs[:-1], qs[1:]):
            m = (sig >= lo) & (sig <= hi)
            if m.sum() > 10:
                xs.append(sig[m].mean())
                ys.append(np.sqrt((err[m] ** 2).mean()))
        a2.plot(xs, ys, 'o-', color=color, lw=2, ms=5, label=label)

    reliability(pix['sig_tta'], pix['err_tta'], C_TTA, 'TTA σ')
    if uncertainty:
        reliability(pix['sig_nll'], pix['err_nll'], C_NLL, 'β-NLL σ')
    lim = a2.get_xlim()[1] if a2.has_data() else 1.0
    a2.plot([0, lim], [0, lim], color=C_REF, ls='--', lw=1.5,
            label='ideal (RMSE = σ)')
    a2.set_xlabel('predicted σ [m] (decile bins)', fontsize=15)
    a2.set_ylabel('empirical RMSE [m]', fontsize=15)
    a2.set_title('Reliability (hole pixels)', fontsize=16)
    a2.legend(frameon=False, fontsize=13)

    for ax in (a1, a2):
        ax.grid(alpha=0.25)
        ax.spines[['top', 'right']].set_visible(False)
        ax.tick_params(labelsize=13)
    fig.savefig(out, dpi=200)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Cross-fold summary figure
# ─────────────────────────────────────────────────────────────────────────────

def fig_test_by_env(per_env: dict, out: Path):
    """Test RMSE per held-out environment (bar = fold; mean line for reference)."""
    envs = list(per_env.keys())
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), constrained_layout=True)
    for ax, metric, title in zip(
            axes, ('rmse', 'rmse_hole'),
            ('Test RMSE (all GT pixels)', 'Test RMSE (hole region)')):
        vals = [per_env[e]['nll'][metric]['mean'] for e in envs]
        errs = [per_env[e]['nll'][metric]['std'] for e in envs]
        ax.bar(envs, vals, yerr=errs, color=C_NLL, width=0.6, capsize=3,
               error_kw={'ecolor': C_REF, 'lw': 1.2})
        for x, v in zip(envs, vals):
            ax.text(x, v, f' {v:.2f}', ha='center', va='bottom', fontsize=14)
        mean = float(np.mean(vals))
        ax.axhline(mean, color=C_REF, ls='--', lw=1.2)
        ax.text(len(envs) - 0.4, mean, f'mean {mean:.2f}', fontsize=13,
                va='bottom', ha='right', color='#555555')
        ax.set_ylabel('RMSE [m]', fontsize=16)
        ax.set_title(title, fontsize=17)
        ax.tick_params(axis='x', rotation=20, labelsize=14)
        ax.tick_params(axis='y', labelsize=14)
        ax.grid(axis='y', alpha=0.25)
        ax.spines[['top', 'right']].set_visible(False)
    fig.savefig(out, dpi=200)
    plt.close(fig)
