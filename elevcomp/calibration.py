"""
Uncertainty calibration metrics on hole pixels (gt_mask=1 & partial_mask=0).

Sparsification / AUSE (Area Under the Sparsification Error curve):
    Ilg et al., "Uncertainty Estimates and Multi-Hypotheses Networks for
    Optical Flow", ECCV 2018. https://arxiv.org/abs/1802.07095
    Poggi et al., "On the Uncertainty of Self-Supervised Monocular Depth
    Estimation", CVPR 2020. https://arxiv.org/abs/2005.06209

Coverage @1σ: for a well-calibrated Gaussian, σ > |error| for ≈68 % of pixels.
"""
import numpy as np

from .utils import pearson

SPARS_FRACS = np.linspace(0.0, 0.95, 20)   # fraction of pixels removed
MIN_HOLE_PX = 20                           # min hole pixels for calibration stats
PIX_SUBSAMPLE = 200                        # calibration pixels kept per sample


def sparsification_curve(err: np.ndarray, sig: np.ndarray) -> np.ndarray:
    """RMSE of remaining pixels after removing the top-f most-uncertain ones."""
    n = err.size
    order = np.argsort(-sig)
    se = err[order].astype(np.float64) ** 2
    cum = np.concatenate([[0.0], np.cumsum(se)])
    out = np.empty(len(SPARS_FRACS))
    for i, f in enumerate(SPARS_FRACS):
        r = min(int(f * n), n - 1)
        out[i] = np.sqrt((cum[-1] - cum[r]) / (n - r))
    return out


def calib_sample(err: np.ndarray, sig: np.ndarray) -> dict:
    """
    Per-sample calibration on hole pixels (both arrays in metres).

    Returns mean σ, Pearson r(σ,|e|), coverage@1σ, AUSE and the normalized
    sparsification curves (uncertainty-sorted and oracle |error|-sorted).
    """
    if err.size < MIN_HOLE_PX:
        return {'mean_sigma': np.nan, 'pearson': np.nan,
                'cov68': np.nan, 'ause': np.nan,
                'curve_unc': None, 'curve_oracle': None}
    c_unc = sparsification_curve(err, sig)
    c_or = sparsification_curve(err, err)
    base = max(c_unc[0], 1e-9)
    c_unc, c_or = c_unc / base, c_or / base
    return {
        'mean_sigma': float(sig.mean()),
        'pearson': pearson(sig, err),
        'cov68': float((sig > err).mean()),
        'ause': float(np.mean(c_unc - c_or)),
        'curve_unc': c_unc,
        'curve_oracle': c_or,
    }


class StreamingCorr:
    """Streaming Pearson correlation + coverage pooled over all hole pixels."""

    def __init__(self):
        self.n = self.sx = self.sy = self.sxx = self.syy = self.sxy = self.cov = 0.0

    def add(self, sig: np.ndarray, err: np.ndarray):
        x, y = sig.astype(np.float64), err.astype(np.float64)
        self.n += x.size
        self.sx += x.sum(); self.sy += y.sum()
        self.sxx += (x * x).sum(); self.syy += (y * y).sum()
        self.sxy += (x * y).sum()
        self.cov += float((x > y).sum())

    def result(self) -> dict:
        if self.n == 0:
            return {'pearson': None, 'cov68': None, 'mean_sigma': None,
                    'mean_abs_err': None, 'n_pixels': 0}
        num = self.n * self.sxy - self.sx * self.sy
        den = np.sqrt(max(self.n * self.sxx - self.sx ** 2, 0)
                      * max(self.n * self.syy - self.sy ** 2, 0))
        return {
            'pearson': float(num / den) if den > 0 else None,
            'cov68': self.cov / self.n,
            'mean_sigma': self.sx / self.n,
            'mean_abs_err': self.sy / self.n,
            'n_pixels': int(self.n),
        }
