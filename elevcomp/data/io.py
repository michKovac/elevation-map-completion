"""Reading and writing elevation completion samples."""
from pathlib import Path

import numpy as np

SAMPLE_KEYS = ("partial_elevation", "partial_mask", "gt_elevation", "gt_mask",
               "frame_idx", "reference_idx", "bounds", "resolution")


def save_sample(
    out_dir,
    sample_name: str,
    partial_elev: np.ndarray,
    partial_mask: np.ndarray,
    gt_elev: np.ndarray,
    gt_mask: np.ndarray = None,
    frame_idx: int = None,
    reference_idx: int = None,
    bounds=None,
    resolution: float = None,
) -> Path:
    """
    Write one sample as a compressed .npz.

    Masks follow the convention 1 = observed / valid, 0 = missing / invalid.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    partial_elev = np.asarray(partial_elev, dtype=np.float32)
    gt_elev = np.asarray(gt_elev, dtype=np.float32)
    partial_mask = np.asarray(partial_mask).astype(bool)
    gt_mask = (np.ones_like(gt_elev, dtype=bool) if gt_mask is None
               else np.asarray(gt_mask).astype(bool))

    for name, arr in (("partial_mask", partial_mask), ("gt_elevation", gt_elev),
                      ("gt_mask", gt_mask)):
        if arr.shape != partial_elev.shape:
            raise ValueError(f"{name} shape {arr.shape} != partial_elevation "
                             f"shape {partial_elev.shape}")

    save_path = out_dir / f"{sample_name}.npz"
    np.savez_compressed(
        save_path,
        partial_elevation=partial_elev,
        partial_mask=partial_mask.astype(np.uint8),
        gt_elevation=gt_elev,
        gt_mask=gt_mask.astype(np.uint8),
        frame_idx=np.array(-1 if frame_idx is None else frame_idx, dtype=np.int32),
        reference_idx=np.array(-1 if reference_idx is None else reference_idx, dtype=np.int32),
        bounds=np.asarray(bounds if bounds is not None else [], dtype=np.float32),
        resolution=np.array(-1.0 if resolution is None else resolution, dtype=np.float32),
    )
    return save_path


def load_sample(npz_path) -> dict:
    """Load one sample as a dict with bool masks."""
    npz_path = Path(npz_path)
    if not npz_path.exists():
        raise FileNotFoundError(f"File not found: {npz_path}")

    data = np.load(npz_path)
    return {
        "partial_elevation": data["partial_elevation"].astype(np.float32),
        "partial_mask": data["partial_mask"].astype(bool),
        "gt_elevation": data["gt_elevation"].astype(np.float32),
        "gt_mask": data["gt_mask"].astype(bool),
        "frame_idx": int(data["frame_idx"]),
        "reference_idx": int(data["reference_idx"]),
        "bounds": data["bounds"],
        "resolution": float(data["resolution"]),
    }
