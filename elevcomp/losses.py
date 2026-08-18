"""
Masked elevation completion losses.

Heteroscedastic aleatoric uncertainty framework:
    Kendall & Gal, "What Uncertainties Do We Need in Bayesian Deep Learning
    for Computer Vision?", NeurIPS 2017. https://arxiv.org/abs/1703.04977

β-NLL loss (fixes training instability of plain NLL):
    Seitzer et al., "On the Pitfalls of Heteroscedastic Uncertainty Estimation
    with Probabilistic Neural Networks", ICLR 2022.
    https://arxiv.org/abs/2203.09168
    https://github.com/martius-lab/beta-nll
"""
import torch
import torch.nn as nn


def _beta_nll_loss(
    mean: torch.Tensor,
    variance: torch.Tensor,
    target: torch.Tensor,
    beta: float = 0.5,
) -> torch.Tensor:
    """
    β-NLL loss per pixel (B, 1, H, W) — masking applied by the caller.

    Verbatim from Seitzer et al., "On the Pitfalls of Heteroscedastic Uncertainty
    Estimation with Probabilistic Neural Networks", ICLR 2022.
    Source: https://github.com/martius-lab/beta-nll

    Standard Gaussian NLL weighted by variance.detach()**beta, which stops
    the gradient through the variance weighting term.  This prevents the
    degenerate solution where the network predicts high variance everywhere
    to reduce the loss without improving predictions (the pitfall from the paper).

    beta=0   → standard NLL (unstable, the pitfall)
    beta=0.5 → recommended default (equal gradient weighting)
    beta=1   → uniform weighting across samples
    """
    loss = 0.5 * ((target - mean) ** 2 / variance + variance.log())
    if beta > 0:
        loss = loss * (variance.detach() ** beta)
    return loss   # (B, 1, H, W) — original sums over D; we keep spatial dims


class ElevationLoss(nn.Module):
    """
    Masked loss for elevation completion.

    Two terms (in normalized elevation units):
        L_valid : loss over all GT-valid pixels (gt_mask == 1)
        L_hole  : loss over hole pixels (partial_mask == 0 ∩ gt_mask == 1)

    Total loss = w_valid * L_valid + w_hole * L_hole

    When log_var is provided:
        Uses β-NLL (Seitzer et al., ICLR 2022) — Gaussian NLL weighted by
        variance^beta with stop-gradient, which fixes the training instability
        of plain NLL.  variance = exp(log_var), beta configured via uncertainty_beta.

    When log_var is None (warmup phase or uncertainty=False):
        Falls back to plain L1 — identical behaviour to before uncertainty was added.
    """

    def __init__(self, w_valid: float = 1.0, w_hole: float = 0.6, beta: float = 0.5):
        super().__init__()
        self.w_valid = w_valid
        self.w_hole  = w_hole
        self.beta    = beta

    def forward(
        self,
        pred: torch.Tensor,                 # (B, 1, H, W)
        gt: torch.Tensor,                   # (B, 1, H, W)
        gt_mask: torch.Tensor,              # (B, 1, H, W) binary
        partial_mask: torch.Tensor = None,  # (B, 1, H, W) binary, optional
        log_var: torch.Tensor = None,       # (B, 1, H, W) from model, optional
    ) -> torch.Tensor:

        if log_var is not None:
            # β-NLL: variance = exp(log_var), clamped for numerical stability
            variance = log_var.clamp(-10.0, 10.0).exp()
            term = _beta_nll_loss(pred, variance, gt, beta=self.beta)
        else:
            # Plain L1 (warmup phase or uncertainty disabled)
            term = (pred - gt).abs()

        n_valid = gt_mask.sum().clamp(min=1)
        loss    = self.w_valid * (term * gt_mask).sum() / n_valid

        if self.w_hole > 0 and partial_mask is not None:
            hole_mask = gt_mask * (1.0 - partial_mask)
            n_hole    = hole_mask.sum().clamp(min=1)
            loss      = loss + self.w_hole * (term * hole_mask).sum() / n_hole

        return loss
