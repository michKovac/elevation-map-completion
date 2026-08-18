"""
Depth images to robot-centric point clouds.

Each TartanGround frame carries four depth images (front/left/right/back) and a
pose per camera. The cameras are unprojected with the dataset intrinsics and
transformed into the frame of one reference camera, giving the partial
observation a real robot would accumulate in a single instant.

    Patel et al., "TartanGround: A Large-Scale Dataset for Ground Robot
    Perception and Navigation", IROS 2025.
"""
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

CAMERAS = ("front", "left", "right", "back")

# TartanGround pinhole intrinsics: 640x640 images, 90 deg horizontal FOV.
FOCAL = 320.0
PRINCIPAL = 320.0

# Depth outside this range is discarded: below is sensor noise at the lens,
# above is where stereo error grows past anything useful for a 50 m window.
DEPTH_MIN = 0.1
DEPTH_MAX = 100.0


def load_depth(depth_path) -> np.ndarray:
    """Load a depth image stored either as .npy or as float32 packed into RGBA .png."""
    depth_path = str(depth_path)
    if depth_path.endswith(".npy"):
        return np.load(depth_path).astype(np.float32)

    depth_rgba = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    if depth_rgba is None:
        raise FileNotFoundError(f"Could not load depth image: {depth_path}")
    return depth_rgba.view("<f4").squeeze().astype(np.float32)


def load_poses(pose_file) -> np.ndarray:
    """Load an N x 7 pose table of [x, y, z, qx, qy, qz, qw]."""
    poses = np.loadtxt(pose_file, dtype=np.float32)
    if poses.ndim == 1:
        poses = poses.reshape(1, -1)
    if poses.shape[1] > 7:
        poses = poses[:, -7:]
    if poses.shape[1] != 7:
        raise ValueError(f"Expected poses with 7 columns, got {poses.shape}")
    return poses


def pose_to_transform(pose, normalize_quaternion: bool = False) -> np.ndarray:
    """
    [x, y, z, qx, qy, qz, qw] -> 4x4 world-from-local transform.

    `normalize_quaternion` divides by the float32 norm before handing the
    quaternion to scipy, which normalises in float64 anyway. The two paths
    therefore differ by roughly 1e-7 in the rotation matrix, and the published
    dataset was generated with normalisation on for the depth cameras and off
    for the ground-truth crop. Both flags are kept so that dataset can be
    reproduced bit-exactly; the resulting elevation differs by at most a few
    millimetres either way, which is four orders of magnitude below the error
    the model is measured at.
    """
    pose = np.asarray(pose, dtype=np.float32).reshape(-1)
    if pose.size != 7:
        raise ValueError(f"Expected pose [x,y,z,qx,qy,qz,qw], got {pose.shape}")

    q = pose[3:7]
    if normalize_quaternion:
        q = q / np.linalg.norm(q)

    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = Rotation.from_quat(q).as_matrix().astype(np.float32)
    T[:3, 3] = pose[:3]
    return T


def apply_transform(points: np.ndarray, T: np.ndarray) -> np.ndarray:
    """Apply a 4x4 transform to N x 3 row-vector points."""
    T = np.asarray(T, dtype=np.float32)
    return (np.asarray(points, dtype=np.float32) @ T[:3, :3].T + T[:3, 3]).astype(np.float32)


def world_to_local(points_world: np.ndarray, pose) -> np.ndarray:
    """Express world points in the local frame of `pose`."""
    T = pose_to_transform(pose)
    return (points_world - T[:3, 3]) @ T[:3, :3]


def depth_to_point_cloud(depth: np.ndarray) -> np.ndarray:
    """
    Unproject a depth image into the camera frame as N x 3 [depth, x, y].

    Pixel centres are sampled at +0.5 so the principal point falls between
    pixels rather than on a corner.
    """
    h, w = depth.shape
    u_grid, v_grid = np.meshgrid(
        np.arange(w, dtype=np.float32) + 0.5,
        np.arange(h, dtype=np.float32) + 0.5,
    )

    valid = (depth > DEPTH_MIN) & (depth < DEPTH_MAX)
    d = depth[valid]
    x = (u_grid[valid] - PRINCIPAL) * d / FOCAL
    y = (v_grid[valid] - PRINCIPAL) * d / FOCAL

    return np.stack([d, x, y], axis=1).astype(np.float32)


def depth_files(traj_path, cam: str) -> list:
    """Sorted depth files of one camera of one trajectory."""
    depth_dir = Path(traj_path) / f"depth_lcam_{cam}"
    if not depth_dir.exists():
        raise FileNotFoundError(f"Depth directory not found: {depth_dir}")

    files = sorted(list(depth_dir.glob("*.png")) + list(depth_dir.glob("*.npy")))
    if not files:
        raise FileNotFoundError(f"No depth files found in {depth_dir}")
    return files


def load_camera_poses(traj_path, cameras=CAMERAS) -> dict:
    """Pose table per camera of one trajectory."""
    poses_by_cam = {}
    for cam in cameras:
        pose_file = Path(traj_path) / f"pose_lcam_{cam}.txt"
        if not pose_file.exists():
            raise FileNotFoundError(f"Pose file not found: {pose_file}")
        poses_by_cam[cam] = load_poses(pose_file)
    return poses_by_cam


def accumulate_frame(
    traj_path,
    frame_idx: int,
    poses_by_cam: dict,
    files_by_cam: dict,
    bounds: np.ndarray,
    cameras=CAMERAS,
) -> tuple:
    """
    Merge the four camera clouds of one frame into the front-camera frame.

    Returns (points, reference_pose). Points outside `bounds` are dropped here so
    the merged cloud stays small; rasterisation applies the same box again.
    """
    reference_pose = poses_by_cam["front"][frame_idx]
    T_ref_world = np.linalg.inv(pose_to_transform(reference_pose, normalize_quaternion=True))

    x_min, x_max, y_min, y_max, z_min, z_max = np.asarray(bounds, dtype=np.float32)
    merged = []

    for cam in cameras:
        depth = load_depth(files_by_cam[cam][frame_idx])
        points = depth_to_point_cloud(depth)

        T_ref_cam = T_ref_world @ pose_to_transform(
            poses_by_cam[cam][frame_idx], normalize_quaternion=True)
        points = apply_transform(points, T_ref_cam)

        inside = (
            (points[:, 0] >= x_min) & (points[:, 0] <= x_max) &
            (points[:, 1] >= y_min) & (points[:, 1] <= y_max) &
            (points[:, 2] >= z_min) & (points[:, 2] <= z_max)
        )
        merged.append(points[inside])

    if not merged:
        return np.empty((0, 3), dtype=np.float32), reference_pose
    return np.concatenate(merged, axis=0).astype(np.float32), reference_pose
