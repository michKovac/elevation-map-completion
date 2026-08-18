"""
Training primitives shared by train.py (single runs) and core/cv.py (5-fold CV).

Heteroscedastic aleatoric uncertainty (optional, uncertainty=true in config):
    Kendall & Gal, "What Uncertainties Do We Need in Bayesian Deep Learning
    for Computer Vision?", NeurIPS 2017. https://arxiv.org/abs/1703.04977

β-NLL loss (prevents variance collapse, activated after warmup epochs):
    Seitzer et al., "On the Pitfalls of Heteroscedastic Uncertainty Estimation
    with Probabilistic Neural Networks", ICLR 2022.
    https://arxiv.org/abs/2203.09168  |  https://github.com/martius-lab/beta-nll
"""
import torch
from torch.amp import autocast
from torch.nn.parallel import DistributedDataParallel as DDP

from .inference import split_output
from .metrics import compute_metrics


def train_one_epoch(model, loader, optimizer, loss_fn, scaler, device, sampler,
                    epoch, grad_clip, uncertainty, amp_dtype=None):
    """One optimization epoch; returns mean training loss.

    amp_dtype: torch.float16 (with `scaler`), torch.bfloat16 (no scaler, wider
    range), or None
    (full fp32). Autocast is enabled whenever amp_dtype is set.
    """
    model.train()
    if sampler is not None:
        sampler.set_epoch(epoch)

    total_loss = 0.0
    for inp, gt, gt_mask in loader:
        inp, gt, gt_mask = (
            inp.to(device, non_blocking=True),
            gt.to(device, non_blocking=True),
            gt_mask.to(device, non_blocking=True),
        )
        partial_mask = inp[:, 1:2]

        optimizer.zero_grad(set_to_none=True)
        with autocast('cuda', enabled=(amp_dtype is not None),
                      dtype=(amp_dtype or torch.float16)):
            out = model(inp)
            pred, log_var = split_output(out, uncertainty)
            loss = loss_fn(pred, gt, gt_mask, partial_mask, log_var)

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        total_loss += loss.item()

    return total_loss / max(len(loader), 1)


@torch.no_grad()
def evaluate(model, loader, loss_fn, device, stats, uncertainty):
    """Batch-averaged validation metrics + loss (used inside the training loop)."""
    model.eval()
    metrics_acc: dict = {}
    total_loss = 0.0

    for inp, gt, gt_mask in loader:
        inp, gt, gt_mask = (
            inp.to(device, non_blocking=True),
            gt.to(device, non_blocking=True),
            gt_mask.to(device, non_blocking=True),
        )
        partial_mask = inp[:, 1:2]
        out = model(inp)
        pred, log_var = split_output(out, uncertainty)

        total_loss += loss_fn(pred, gt, gt_mask, partial_mask, log_var).item()
        m = compute_metrics(
            pred, gt, gt_mask, partial_mask,
            partial_elev=inp[:, :1],
            denorm_std=stats['std'], denorm_mean=stats['mean'],
        )
        for k, v in m.items():
            metrics_acc[k] = metrics_acc.get(k, 0.0) + v

    n = max(len(loader), 1)
    result = {k: v / n for k, v in metrics_acc.items()}
    result['loss'] = total_loss / n
    return result


def make_state(epoch, model, optimizer, scheduler, scaler, best_rmse, stats, cfg):
    """Full checkpoint state — self-contained (config + stats travel with weights)."""
    raw_model = model.module if isinstance(model, DDP) else model
    state = {
        'epoch':     epoch,
        'model':     raw_model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'best_rmse': best_rmse,
        'stats':     stats,
        'config':    cfg,
    }
    if scaler is not None:
        state['scaler'] = scaler.state_dict()
    return state


def save_checkpoint(state: dict, path):
    """Atomic save: write to .tmp, then rename."""
    tmp = path.with_suffix('.tmp')
    torch.save(state, tmp)
    tmp.rename(path)
