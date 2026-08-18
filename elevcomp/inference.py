"""
Inference helpers: single-pass (β-NLL) prediction and D4 TTA self-ensemble.

Single source of truth for both uncertainty methods, used by the training
driver and by examples/end_to_end.py.

β-NLL heteroscedastic uncertainty (model predicts mean + log-variance):
    Kendall & Gal, "What Uncertainties Do We Need in Bayesian Deep Learning
    for Computer Vision?", NeurIPS 2017. https://arxiv.org/abs/1703.04977
    Seitzer et al., ICLR 2022. https://arxiv.org/abs/2203.09168

D4 self-ensemble for image restoration:
    Lim et al., "Enhanced Deep Residual Networks for Single Image
    Super-Resolution", CVPRW 2017. https://arxiv.org/abs/1707.02921

TTA as predictive uncertainty:
    Shanmugam et al., "Better Aggregation in Test-Time Augmentation",
    ICCV 2021. https://arxiv.org/abs/2011.11156
"""
import torch

# Each entry: (k_rot90, do_hflip) — 4 rotations × 2 flips = the full D4 group.
D4_TRANSFORMS = [(k, f) for k in range(4) for f in (False, True)]


def split_output(out: torch.Tensor, use_uncertainty: bool):
    """Split model output into (pred, log_var). log_var is None without the head."""
    pred = out[:, :1]                                    # always 1-channel prediction
    log_var = out[:, 1:2] if use_uncertainty else None   # β-NLL head channel
    return pred, log_var


def d4_apply(t: torch.Tensor, k: int, hflip: bool) -> torch.Tensor:
    """Apply rot90(k) then optional horizontal flip to a (..., H, W) tensor."""
    if k:
        t = torch.rot90(t, k, dims=(-2, -1))
    if hflip:
        t = torch.flip(t, dims=(-1,))
    return t


def d4_invert(t: torch.Tensor, k: int, hflip: bool) -> torch.Tensor:
    """Invert d4_apply so predictions align with the original orientation."""
    if hflip:
        t = torch.flip(t, dims=(-1,))
    if k:
        t = torch.rot90(t, -k, dims=(-2, -1))
    return t


@torch.no_grad()
def nll_predict(model: torch.nn.Module, inp: torch.Tensor, has_uncertainty: bool):
    """
    Single forward pass.

    Returns:
        pred  : (B, 1, H, W) prediction in normalized units
        sigma : (B, 1, H, W) aleatoric σ = exp(0.5·log_var), normalized units
                (None when the model has no uncertainty head)
    """
    out = model(inp)
    pred, log_var = split_output(out, has_uncertainty)
    sigma = torch.exp(0.5 * log_var) if has_uncertainty else None
    return pred, sigma


@torch.no_grad()
def tta_predict(model: torch.nn.Module, inp: torch.Tensor):
    """
    D4 test-time-augmentation ensemble (8 forward passes).

    Returns:
        mean_pred : (B, 1, H, W) mean prediction across the 8 transforms
        sigma     : (B, 1, H, W) std across the 8 transforms (epistemic proxy)
    """
    preds = []
    for k, hflip in D4_TRANSFORMS:
        out = model(d4_apply(inp, k, hflip))
        preds.append(d4_invert(out[:, :1], k, hflip))
    stack = torch.stack(preds, dim=0)   # (8, B, 1, H, W)
    return stack.mean(0), stack.std(0)
