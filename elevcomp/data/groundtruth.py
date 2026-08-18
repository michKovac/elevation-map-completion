"""
Dense ground-truth elevation from the global semantic point cloud.

TartanGround ships one dense cloud per environment ({env}_sem.pcd), accumulated
over the whole scene. Cropping it around a robot pose gives the elevation the
robot would see with perfect, unoccluded sensing — the target the completion
model is trained against.
"""
import numpy as np
from scipy.spatial import cKDTree

from elevcomp.data.depth import world_to_local
from elevcomp.data.raster import DEFAULT_PERCENTILE, apply_axis_mapping, grid_shape, rasterize_elevation


def load_global_cloud(pcd_path) -> tuple:
    """
    Load an environment's global cloud and index it.

    Returns (points, tree). Building the KD-tree over tens of millions of points
    takes a while, so load once per environment and reuse across trajectories.
    """
    import open3d as o3d

    pcd = o3d.io.read_point_cloud(str(pcd_path))
    points = np.asarray(pcd.points, dtype=np.float32)
    if points.size == 0:
        raise ValueError(f"Empty global point cloud: {pcd_path}")
    return points, cKDTree(points)


def extract_gt_elevation(
    global_points: np.ndarray,
    global_tree: cKDTree,
    pose,
    bounds,
    resolution: float,
    percentile: float = DEFAULT_PERCENTILE,
    axis_mapping: str = "yxz_neg",
) -> tuple:
    """
    Crop the global cloud around `pose` and rasterise it.

    The KD-tree is queried with the radius of the sphere enclosing the bounds
    box (plus 1 m), then the exact box is applied during rasterisation.
    """
    x_min, x_max, y_min, y_max, z_min, z_max = np.asarray(bounds, dtype=np.float32)
    radius = np.sqrt(
        max(abs(x_min), abs(x_max)) ** 2 +
        max(abs(y_min), abs(y_max)) ** 2 +
        max(abs(z_min), abs(z_max)) ** 2
    ) + 1.0

    candidates = global_tree.query_ball_point(np.asarray(pose, dtype=np.float32)[:3], r=radius)
    if len(candidates) == 0:
        x_size, y_size = grid_shape(bounds, resolution)
        return (np.full((x_size, y_size), np.nan, dtype=np.float32),
                np.zeros((x_size, y_size), dtype=bool))

    points_local = world_to_local(global_points[np.asarray(candidates)], pose)
    points_local = apply_axis_mapping(points_local, axis_mapping)

    return rasterize_elevation(points_local, bounds, resolution, percentile=percentile)
