"""
Classical (learning-free) hole-filling baselines for elevation map completion.

Each baseline takes the raw partial elevation (metres, NaN in holes) and the
partial mask, and returns a fully dense map: observed values are ALWAYS kept
verbatim, only holes are filled. Evaluation therefore uses exactly the same
per-sample metrics as the learned models (core/metrics.py, incl. masked SSIM).

Methods and references:
    nearest   — nearest-neighbour extrapolation via the exact Euclidean
                distance transform.
                Felzenszwalb & Huttenlocher, "Distance Transforms of Sampled
                Functions", Theory of Computing 8:415–428, 2012.
    linear    — piecewise-linear barycentric interpolation on a Delaunay
                triangulation (scipy.interpolate.griddata, Qhull); pixels
                outside the convex hull fall back to nearest-neighbour.
                Barber et al., "The Quickhull Algorithm for Convex Hulls",
                ACM TOMS 22(4):469–483, 1996.
    idw       — inverse-distance weighting over the k nearest observed pixels.
                Shepard, "A two-dimensional interpolation function for
                irregularly-spaced data", ACM National Conf., 1968.
    telea     — fast marching inpainting.
                Telea, "An Image Inpainting Technique Based on the Fast
                Marching Method", J. Graphics Tools 9(1):23–34, 2004.
    ns        — fluid-dynamics (Navier–Stokes) inpainting.
                Bertalmío et al., "Navier-Stokes, Fluid Dynamics, and Image
                and Video Inpainting", CVPR 2001.

Note on telea/ns: OpenCV inpainting operates on 8-bit images, so elevations
are quantized to 256 levels over the observed range before inpainting
(quantization ≈ range/255 ≈ centimetres — negligible vs. metre-scale errors).
"""
import numpy as np
from scipy import ndimage
from scipy.interpolate import griddata
from scipy.spatial import cKDTree

try:
    import cv2
    _HAS_CV2 = True
except ImportError:          # telea/ns simply unavailable without OpenCV
    _HAS_CV2 = False


def _observed(pe: np.ndarray, pm: np.ndarray):
    """Boolean observed mask + finite partial elevation (defensive NaN check)."""
    obs = (pm > 0.5) & np.isfinite(pe)
    return obs


def fill_nearest(pe: np.ndarray, pm: np.ndarray) -> np.ndarray:
    obs = _observed(pe, pm)
    if not obs.any():
        return np.zeros_like(pe)
    # Indices of the nearest observed pixel for every position.
    _, (iy, ix) = ndimage.distance_transform_edt(~obs, return_indices=True)
    return pe[iy, ix]


def fill_linear(pe: np.ndarray, pm: np.ndarray) -> np.ndarray:
    obs = _observed(pe, pm)
    if obs.sum() < 4:                       # Delaunay needs a non-degenerate set
        return fill_nearest(pe, pm)
    pts = np.argwhere(obs).astype(np.float32)
    vals = pe[obs].astype(np.float64)
    holes = np.argwhere(~obs).astype(np.float32)
    filled = pe.copy()
    interp = griddata(pts, vals, holes, method='linear')
    filled[~obs] = interp
    # Outside the convex hull griddata returns NaN → nearest-neighbour fallback.
    nan_left = ~np.isfinite(filled)
    if nan_left.any():
        filled[nan_left] = fill_nearest(pe, pm)[nan_left]
    return filled


def fill_idw(pe: np.ndarray, pm: np.ndarray, k: int = 8, power: float = 2.0) -> np.ndarray:
    obs = _observed(pe, pm)
    if not obs.any():
        return np.zeros_like(pe)
    pts = np.argwhere(obs)
    vals = pe[obs].astype(np.float64)
    holes = np.argwhere(~obs)
    if holes.size == 0:
        return pe.copy()
    tree = cKDTree(pts)
    k_eff = min(k, len(pts))
    dist, idx = tree.query(holes, k=k_eff, workers=1)
    dist = np.atleast_2d(dist.astype(np.float64))
    idx = np.atleast_2d(idx)
    w = 1.0 / np.maximum(dist, 1e-6) ** power
    est = (w * vals[idx]).sum(axis=1) / w.sum(axis=1)
    filled = pe.copy()
    filled[~obs] = est
    return filled


def _fill_cv2(pe: np.ndarray, pm: np.ndarray, flag) -> np.ndarray:
    obs = _observed(pe, pm)
    if not obs.any():
        return np.zeros_like(pe)
    lo, hi = float(pe[obs].min()), float(pe[obs].max())
    scale = max(hi - lo, 1e-6)
    img8 = np.zeros(pe.shape, dtype=np.uint8)
    img8[obs] = np.clip((pe[obs] - lo) / scale * 255.0, 0, 255).astype(np.uint8)
    hole_mask = (~obs).astype(np.uint8)
    out8 = cv2.inpaint(img8, hole_mask, inpaintRadius=5, flags=flag)
    filled = pe.copy()
    filled[~obs] = out8[~obs].astype(np.float32) / 255.0 * scale + lo
    return filled


def fill_telea(pe: np.ndarray, pm: np.ndarray) -> np.ndarray:
    return _fill_cv2(pe, pm, cv2.INPAINT_TELEA)


def fill_ns(pe: np.ndarray, pm: np.ndarray) -> np.ndarray:
    return _fill_cv2(pe, pm, cv2.INPAINT_NS)


# Registry: method name → fill function. Order = presentation order in tables.
BASELINES = {
    'nearest': fill_nearest,
    'linear':  fill_linear,
    'idw':     fill_idw,
}
if _HAS_CV2:
    BASELINES['telea'] = fill_telea
    BASELINES['ns'] = fill_ns
