"""
Point clouds to robot-centric elevation maps.

A cell's elevation is a low percentile of the point heights inside it rather
than the minimum: the minimum latches onto single stray points below the
surface, while a low percentile keeps the ground surface under vegetation and
stays robust to depth noise. The same statistic is used for the partial input
and for the dense ground truth so the two are directly comparable.
"""
import numpy as np

# Percentile of the per-cell height distribution taken as the cell elevation.
DEFAULT_PERCENTILE = 20.0

_AXES = {"x": 0, "y": 1, "z": 2}

# The depth clouds arrive as [depth, x, y] in camera convention; the elevation
# grid expects [east, north, up]. 'yxz_neg' is the mapping that produced the
# dataset of the paper.
AXIS_ALIASES = {
    "identity": "x,y,z",
    "yxz_neg": "y,x,-z",
}


def apply_axis_mapping(points: np.ndarray, mode: str) -> np.ndarray:
    """
    Permute and flip point-cloud axes.

    `mode` is either an alias from AXIS_ALIASES or an explicit spec such as
    "y,x,-z" naming the source axis of each output column.
    """
    spec = AXIS_ALIASES.get(mode, mode)
    parts = [p.strip() for p in spec.split(",")]
    if len(parts) != 3:
        raise ValueError(f"Axis mapping needs 3 components, got {mode!r}")

    cols = []
    for part in parts:
        sign = -1.0 if part.startswith("-") else 1.0
        axis = part.lstrip("+-")
        if axis not in _AXES:
            raise ValueError(f"Unknown axis {part!r} in mapping {mode!r}")
        cols.append(sign * points[:, _AXES[axis]])

    return np.stack(cols, axis=1).astype(np.float32)


def grid_shape(bounds, resolution: float) -> tuple:
    x_min, x_max, y_min, y_max, _, _ = np.asarray(bounds, dtype=np.float32)
    return (int((x_max - x_min) / resolution) + 1,
            int((y_max - y_min) / resolution) + 1)


def rasterize_elevation(
    points: np.ndarray,
    bounds,
    resolution: float,
    percentile: float = DEFAULT_PERCENTILE,
) -> tuple:
    """
    Rasterise a local point cloud onto an elevation grid.

    Returns (elevation, mask): float32 heights with NaN in empty cells, and a
    bool mask marking the cells that received at least one point.
    """
    points = np.asarray(points, dtype=np.float32)
    x_min, x_max, y_min, y_max, z_min, z_max = np.asarray(bounds, dtype=np.float32)
    x_size, y_size = grid_shape(bounds, resolution)

    elevation = np.full((x_size, y_size), np.nan, dtype=np.float32)
    mask = np.zeros((x_size, y_size), dtype=bool)
    if len(points) == 0:
        return elevation, mask

    inside = (
        (points[:, 0] >= x_min) & (points[:, 0] <= x_max) &
        (points[:, 1] >= y_min) & (points[:, 1] <= y_max) &
        (points[:, 2] >= z_min) & (points[:, 2] <= z_max)
    )
    pts = points[inside]
    if len(pts) == 0:
        return elevation, mask

    ix = np.clip(np.floor((pts[:, 0] - x_min) / resolution).astype(np.int32), 0, x_size - 1)
    iy = np.clip(np.floor((pts[:, 1] - y_min) / resolution).astype(np.int32), 0, y_size - 1)

    # Sorting by flat cell index turns the per-cell percentile into a scan over
    # contiguous slices.
    flat = ix * y_size + iy
    order = np.argsort(flat)
    flat_sorted = flat[order]
    z_sorted = pts[:, 2][order]

    cells, starts, counts = np.unique(flat_sorted, return_index=True, return_counts=True)
    for cell, start, count in zip(cells, starts, counts):
        elevation[cell // y_size, cell % y_size] = np.percentile(
            z_sorted[start:start + count], percentile)
        mask[cell // y_size, cell % y_size] = True

    return elevation, mask
