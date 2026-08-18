"""
Evaluation metrics for elevation map completion.

All metrics are computed in **denormalized** (real-world metre) units.
Two evaluation regions are reported:
    overall  — all GT-valid pixels (gt_mask == 1)
    hole     — pixels unseen by stereo (partial_mask == 0) ∩ gt_mask == 1

SSIM
    Wang et al., "Image Quality Assessment: From Error Visibility to Structural
    Similarity", IEEE Trans. Image Process., 13(4):600–612, 2004.
    https://doi.org/10.1109/TIP.2003.819861

AbsRel / MAE / RMSE
    Standard depth-estimation metrics following:
    Eigen et al., "Depth Map Prediction from a Single Image using a Multi-Scale
    Deep Network", NeurIPS 2014. https://arxiv.org/abs/1406.2283
"""
import torch
import torch.nn.functional as F


_SSIM_K = 11   # Gaussian window size (shared with the valid-window check below)


@torch.no_grad()
def _ssim_map(pred: torch.Tensor, gt: torch.Tensor, data_range: float) -> torch.Tensor:
    """
    Per-pixel single-scale SSIM map (Wang et al. 2004, IEEE TIP).
    Gaussian window k=11, σ=1.5; stability constants k1=0.01, k2=0.03.
    No external dependency — matches torchmetrics default parameters.
    The caller decides which pixels to average over (see compute_metrics).
    """
    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2
    k  = _SSIM_K
    sigma = 1.5
    coords = torch.arange(k, dtype=torch.float32, device=pred.device) - k // 2
    gauss  = torch.exp(-coords ** 2 / (2 * sigma ** 2))
    gauss  = gauss / gauss.sum()
    kernel = (gauss[:, None] * gauss[None, :]).unsqueeze(0).unsqueeze(0)  # (1,1,k,k)

    mu_p = F.conv2d(pred, kernel, padding=k // 2)
    mu_g = F.conv2d(gt,   kernel, padding=k // 2)

    mu_pp = F.conv2d(pred * pred, kernel, padding=k // 2) - mu_p ** 2
    mu_gg = F.conv2d(gt   * gt,   kernel, padding=k // 2) - mu_g ** 2
    mu_pg = F.conv2d(pred * gt,   kernel, padding=k // 2) - mu_p * mu_g

    ssim_map = (
        (2 * mu_p * mu_g + C1) * (2 * mu_pg + C2)
        / ((mu_p ** 2 + mu_g ** 2 + C1) * (mu_pp + mu_gg + C2))
    )
    return ssim_map


@torch.no_grad()
def compute_metrics(
    pred: torch.Tensor,                  # (B, 1, H, W) normalized
    gt: torch.Tensor,                    # (B, 1, H, W) normalized
    gt_mask: torch.Tensor,               # (B, 1, H, W) binary
    partial_mask: torch.Tensor = None,   # (B, 1, H, W) binary
    partial_elev: torch.Tensor = None,   # (B, 1, H, W) normalized — channel 0 of inp
    *,
    denorm_std: float = 1.0,
    denorm_mean: float = 0.0,
) -> dict:
    """
    Returns a dict of Python floats.
    Keys:
        mae, rmse, abs_rel          — pred vs GT over all GT-valid pixels
        mae_hole, rmse_hole         — pred vs GT over hole region only
        mae_comp, rmse_comp         — composite vs GT over all GT-valid pixels
        ssim                        — structural similarity (full image, pred)

    Composite = stereo measurement where partial_mask=1, pred where partial_mask=0.
    This is the realistic deployment output: known stereo data is never overwritten.
    """
    p = pred * denorm_std + denorm_mean
    g = gt   * denorm_std + denorm_mean

    valid = gt_mask.bool()
    diff  = (p - g).abs()

    mae     = diff[valid].mean()
    rmse    = ((p - g)[valid] ** 2).mean().sqrt()
    abs_rel = (diff[valid] / g[valid].abs().clamp(min=0.1)).mean()

    out = {
        'mae':     float(mae),
        'rmse':    float(rmse),
        'abs_rel': float(abs_rel),
    }

    if partial_mask is not None:
        hole = valid & ~partial_mask[:, :1].bool()
        if hole.any():
            out['mae_hole']  = float(diff[hole].mean())
            out['rmse_hole'] = float(((p - g)[hole] ** 2).mean().sqrt())

    # Composite metrics — only when we have the stereo input elevation
    if partial_mask is not None and partial_elev is not None:
        pm   = partial_mask[:, :1].bool()
        se   = partial_elev[:, :1] * denorm_std + denorm_mean  # stereo in metres
        comp = torch.where(pm, se, p)                          # stereo + pred fill
        diff_c = (comp - g).abs()
        if valid.any():
            out['mae_comp']  = float(diff_c[valid].mean())
            out['rmse_comp'] = float(((comp - g)[valid] ** 2).mean().sqrt())

    if pred.shape[-1] >= 16:
        # data_range from actual GT so SSIM constants scale correctly regardless of terrain
        dr = float((g[valid].max() - g[valid].min()).clamp(min=1.0)) if valid.any() else 1.0
        ssim_map = _ssim_map(p, g, data_range=dr)
        # Masked SSIM: average only over pixels whose full k×k window lies inside
        # the valid GT region — padding and invalid (zero-filled) GT cells would
        # otherwise contaminate the window statistics and depress the score.
        ones = torch.ones(1, 1, _SSIM_K, _SSIM_K, device=pred.device)
        frac = F.conv2d(gt_mask.float(), ones, padding=_SSIM_K // 2) / (_SSIM_K ** 2)
        win_valid = frac >= 1.0 - 1e-6
        if win_valid.any():
            out['ssim'] = float(ssim_map[win_valid].mean())
        elif valid.any():
            # Fallback for heavily fragmented masks: plain masked average.
            out['ssim'] = float(ssim_map[valid].mean())

    return out
