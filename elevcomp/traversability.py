"""
Slope-based traversability from a completed elevation map.

This is the decision a planner actually consumes: a cell is traversable when the
local terrain slope stays below a threshold. Because completion invents the
elevation of unobserved cells, the predicted uncertainty can gate the decision —
cells the model is unsure about are declared non-traversable rather than
optimistically passable.

The safety-relevant metric is the false-safe rate: ground truth says unsafe, the
prediction says safe. Accuracy alone is misleading here, since most hole cells
are traversable in the ground truth and a trivial always-safe classifier would
score well on accuracy while scoring 1.0 on false-safe.
"""
import numpy as np

# Grid resolution of the dataset, metres per cell.
RESOLUTION = 0.2

# Slope a tracked or legged platform is assumed to handle.
DEFAULT_SLOPE_THRESHOLD_DEG = 25.0

TRAVERSABLE = 1
NON_TRAVERSABLE = 0
UNKNOWN = -1


def slope_deg(elevation: np.ndarray, valid: np.ndarray,
              resolution: float = RESOLUTION) -> np.ndarray:
    """
    Central-difference terrain slope in degrees, NaN where it cannot be computed.

    A cell needs all four neighbours valid, so the slope is undefined on the
    border and around holes in the validity mask.
    """
    slope = np.full(elevation.shape, np.nan, np.float32)

    dzx = (elevation[1:-1, 2:] - elevation[1:-1, :-2]) / (2 * resolution)
    dzy = (elevation[2:, 1:-1] - elevation[:-2, 1:-1]) / (2 * resolution)
    computable = (valid[1:-1, 1:-1] & valid[1:-1, 2:] & valid[1:-1, :-2]
                  & valid[2:, 1:-1] & valid[:-2, 1:-1])

    inner = slope[1:-1, 1:-1]
    gradient = np.sqrt(dzx ** 2 + dzy ** 2)
    inner[computable] = np.degrees(np.arctan(gradient[computable]))
    return slope


def traversability(slope: np.ndarray,
                   threshold_deg: float = DEFAULT_SLOPE_THRESHOLD_DEG) -> np.ndarray:
    """Slope map -> {1 traversable, 0 non-traversable, -1 unknown}."""
    out = np.full(slope.shape, UNKNOWN, np.int8)
    finite = np.isfinite(slope)
    out[finite & (slope <= threshold_deg)] = TRAVERSABLE
    out[finite & (slope > threshold_deg)] = NON_TRAVERSABLE
    return out


def apply_sigma_gate(trav_map: np.ndarray, sigma: np.ndarray, tau: float) -> np.ndarray:
    """
    Declare cells with predicted sigma above `tau` non-traversable.

    A conservative use of uncertainty: it can only turn traversable into
    non-traversable, never the reverse.
    """
    if not np.isfinite(tau):
        return trav_map
    gated = trav_map.copy()
    gated[(trav_map == TRAVERSABLE) & (sigma > tau)] = NON_TRAVERSABLE
    return gated


def false_safe_rate(gt_trav: np.ndarray, pred_trav: np.ndarray,
                    eval_mask: np.ndarray = None) -> tuple:
    """
    Fraction of truly unsafe cells that were predicted safe.

    Returns (rate, n_false_safe, n_gt_unsafe). Cells that are unknown in either
    map are excluded, as is anything outside `eval_mask`.
    """
    considered = (gt_trav != UNKNOWN) & (pred_trav != UNKNOWN)
    if eval_mask is not None:
        considered &= eval_mask

    gt_unsafe = considered & (gt_trav == NON_TRAVERSABLE)
    n_gt_unsafe = int(gt_unsafe.sum())
    if n_gt_unsafe == 0:
        return float('nan'), 0, 0

    n_false_safe = int((gt_unsafe & (pred_trav == TRAVERSABLE)).sum())
    return n_false_safe / n_gt_unsafe, n_false_safe, n_gt_unsafe
