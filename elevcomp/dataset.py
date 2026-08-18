"""
Elevation dataset loader for map completion training.

Dataset
    Elevation samples derived from TartanAir (OldTownSummer sequence).
    Wang et al., "TartanAir: A Dataset to Push the Limits of Visual SLAM",
    IROS 2020. https://arxiv.org/abs/2011.00359
    https://theairlab.org/tartanair-dataset/

    Each NPZ contains a 251×251 partial stereo elevation map and a dense
    ground-truth elevation map (50×50 m area, 0.2 m/pixel resolution).
"""
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, DistributedSampler


# ── Ray-cone augmentation ──────────────────────────────────────────────────────
# Precomputed angle maps cached per shape — arctan2 runs once, reused every sample.
_ANGLE_MAP_CACHE: dict = {}

def _angle_map(shape: tuple) -> np.ndarray:
    if shape not in _ANGLE_MAP_CACHE:
        h, w = shape
        cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
        yy, xx = np.meshgrid(np.arange(h, dtype=np.float32),
                             np.arange(w, dtype=np.float32), indexing='ij')
        amap = np.degrees(np.arctan2(cy - yy, xx - cx)) % 360.0
        _ANGLE_MAP_CACHE[shape] = amap
    return _ANGLE_MAP_CACHE[shape]


def _apply_ray_augmentation(
    pe: np.ndarray,
    pm: np.ndarray,
    n_cones_max: int,
    angle_range: tuple,
) -> tuple:
    """
    Remove 1..n_cones_max random angular sectors from the partial elevation.

    Each cone:
      - independent random direction  [0°, 360°)
      - independent random half-width [angle_range[0]/2, angle_range[1]/2]

    Applied on raw H×W arrays BEFORE normalisation/padding so the physical
    sensor origin (image centre) is preserved.

    Returns (pe_aug, pm_aug) — same dtype/shape as inputs.
    """
    amap = _angle_map(pe.shape)

    n = random.randint(1, n_cones_max)
    pe_aug = pe.copy()
    pm_aug = pm.copy()

    for _ in range(n):
        direction  = random.uniform(0.0, 360.0)
        half_width = random.uniform(angle_range[0] / 2.0, angle_range[1] / 2.0)

        diff = (amap - direction + 180.0) % 360.0 - 180.0
        cone = np.abs(diff) <= half_width

        pe_aug[cone] = np.nan
        pm_aug[cone] = 0.0

    return pe_aug, pm_aug


# ── Dataset ────────────────────────────────────────────────────────────────────

class ElevationDataset(Dataset):
    """
    Loads paired (partial_elevation, gt_elevation) NPZ samples.

    Each sample contains:
        partial_elevation : H×W float32, NaN where stereo coverage is missing
        partial_mask      : H×W uint8,   1 = observed by stereo
        gt_elevation      : H×W float32, dense reference from simulation
        gt_mask           : H×W uint8,   1 = valid GT cell

    Returns (inp, gt, gt_mask) where:
        inp     : (2, pad_to, pad_to) — [normalized_elevation, partial_mask]
        gt      : (1, pad_to, pad_to) — normalized GT elevation
        gt_mask : (1, pad_to, pad_to) — binary GT validity mask

    Ray-cone augmentation (training only):
        Randomly removes 1–n_cones_max angular wedges from the partial
        elevation, simulating missing camera directions.  Controlled via:
            aug_ray_p          — probability of applying (0 = off)
            aug_ray_n_max      — max number of cones (1..n_max chosen uniformly)
            aug_ray_angle_deg  — [min°, max°] cone width per sector
    """

    _SPLITS = {'train': (0.0, 0.80), 'val': (0.80, 0.90), 'test': (0.90, 1.0)}

    def __init__(
        self,
        data_dir: str = None,
        split: str = 'train',
        augment: bool = True,
        stats: dict = None,
        pad_to: int = 256,
        fix_mask_rotation: bool = False,
        aug_hflip_p: float = 0.5,
        aug_vflip_p: float = 0.5,
        aug_rot90_p: float = 0.75,
        aug_ray_p: float = 0.0,
        aug_ray_n_max: int = 3,
        aug_ray_angle_deg: tuple = (5, 40),
        files: list = None,
    ):
        # Explicit file list (cross-validation folds) bypasses the directory
        # glob and the fractional split — `split` then only controls augmentation.
        explicit_files = files is not None
        if explicit_files:
            files = [Path(f) for f in files]
            if len(files) == 0:
                raise ValueError('Explicit `files` list is empty.')
        else:
            # Support flat dir (*.npz) and subdirectory layout (e.g. P2000/*.npz)
            files = sorted(Path(data_dir).glob('*.npz'))
            if len(files) == 0:
                files = sorted(Path(data_dir).glob('**/*.npz'))
            if len(files) == 0:
                raise FileNotFoundError(f'No .npz files found in {data_dir}')

        self.fix_mask_rotation = fix_mask_rotation
        self.aug_hflip_p       = aug_hflip_p
        self.aug_vflip_p       = aug_vflip_p
        self.aug_rot90_p       = aug_rot90_p
        self.aug_ray_p         = aug_ray_p if (augment and split == 'train') else 0.0
        self.aug_ray_n_max     = max(1, int(aug_ray_n_max))
        self.aug_ray_angle_deg = tuple(aug_ray_angle_deg)

        if explicit_files:
            self.files = files
        else:
            n = len(files)
            lo, hi = self._SPLITS[split]
            self.files = files if n < 10 else files[int(lo * n) : int(hi * n)]
            if len(self.files) == 0:
                raise ValueError(f'Split "{split}" produced 0 files from {n} total.')

        self.augment = augment and split == 'train'
        self.pad_to  = pad_to
        self.stats   = stats if stats is not None else self._compute_stats()

    # ------------------------------------------------------------------
    def _compute_stats(self) -> dict:
        vals = []
        for f in self.files:
            e = np.load(f)['gt_elevation']
            vals.append(e[np.isfinite(e)])
        v = np.concatenate(vals)
        return {'mean': float(v.mean()), 'std': float(v.std())}

    def _norm(self, x: np.ndarray) -> np.ndarray:
        return (x - self.stats['mean']) / (self.stats['std'] + 1e-6)

    def _pad(self, arr: np.ndarray) -> np.ndarray:
        H, W = arr.shape
        ph = (self.pad_to - H) // 2
        pw = (self.pad_to - W) // 2
        return np.pad(
            arr,
            ((ph, self.pad_to - H - ph), (pw, self.pad_to - W - pw)),
            mode='constant',
            constant_values=0.0,
        )

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int):
        d  = np.load(self.files[idx])
        pe = d['partial_elevation'].astype(np.float32)
        pm = d['partial_mask'].astype(np.float32)
        gt = d['gt_elevation'].astype(np.float32)
        gm = d['gt_mask'].astype(np.float32)

        # Old ForestEnv dataset: masks saved with one extra rot90 vs elevation.
        if self.fix_mask_rotation:
            pm = np.rot90(pm, k=-1)
            gm = np.rot90(gm, k=-1)

        # ── Ray-cone augmentation (before normalisation / padding) ─────────────
        # Operates on raw H×W so the sensor origin (image centre) is preserved.
        if self.aug_ray_p > 0.0 and random.random() < self.aug_ray_p:
            pe, pm = _apply_ray_augmentation(
                pe, pm,
                n_cones_max=self.aug_ray_n_max,
                angle_range=self.aug_ray_angle_deg,
            )

        # ── Normalise & pad ────────────────────────────────────────────────────
        # Normalize first; then fill NaN holes with 0.0 (= dataset mean in
        # normalized space), so masked pixels carry a neutral signal.
        pe_n = np.where(np.isfinite(pe), self._norm(pe), 0.0)
        gt_n = np.where(np.isfinite(gt), self._norm(gt), 0.0)

        inp  = np.stack([self._pad(pe_n), self._pad(pm)], axis=0)
        gt_n = self._pad(gt_n)[None]
        gm   = self._pad(gm)[None]

        # ── Geometric augmentation (flip / rot90) ──────────────────────────────
        if self.augment:
            if random.random() < self.aug_hflip_p:
                inp, gt_n, gm = (inp[:, :, ::-1].copy(),
                                 gt_n[:, :, ::-1].copy(),
                                 gm[:, :, ::-1].copy())
            if random.random() < self.aug_vflip_p:
                inp, gt_n, gm = (inp[:, ::-1, :].copy(),
                                 gt_n[:, ::-1, :].copy(),
                                 gm[:, ::-1, :].copy())
            if random.random() < self.aug_rot90_p:
                k = random.randint(1, 3)
                inp  = np.rot90(inp,  k, (1, 2)).copy()
                gt_n = np.rot90(gt_n, k, (1, 2)).copy()
                gm   = np.rot90(gm,   k, (1, 2)).copy()

        return (
            torch.from_numpy(inp),
            torch.from_numpy(gt_n),
            torch.from_numpy(gm),
        )


# ── DataLoaders ────────────────────────────────────────────────────────────────

def get_dataloaders(cfg, rank: int = 0, world_size: int = 1):
    """
    Returns train_loader, val_loader, test_ds, stats, train_sampler.
    Ray-cone augmentation is active only for the train split.
    """
    fix_rot    = cfg.get('fix_mask_rotation', False)
    ray_p      = cfg.get('aug_ray_p', 0.0)
    ray_n_max  = cfg.get('aug_ray_n_max', 3)
    ray_angles = cfg.get('aug_ray_angle_deg', [5, 40])

    ray_kw = dict(
        fix_mask_rotation=fix_rot,
        aug_hflip_p=cfg.get('aug_hflip_p', 0.5),
        aug_vflip_p=cfg.get('aug_vflip_p', 0.5),
        aug_rot90_p=cfg.get('aug_rot90_p', 0.75),
        aug_ray_p=ray_p,
        aug_ray_n_max=ray_n_max,
        aug_ray_angle_deg=ray_angles,
    )

    train_ds = ElevationDataset(cfg['data_dir'], 'train', augment=True,
                                pad_to=cfg['pad_to'], **ray_kw)
    stats = train_ds.stats

    val_ds  = ElevationDataset(cfg['data_dir'], 'val',  augment=False,
                               stats=stats, pad_to=cfg['pad_to'], **ray_kw)
    test_ds = ElevationDataset(cfg['data_dir'], 'test', augment=False,
                               stats=stats, pad_to=cfg['pad_to'], **ray_kw)

    train_sampler = (
        DistributedSampler(train_ds, world_size, rank, shuffle=True)
        if world_size > 1 else None
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg['batch_size'],
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        num_workers=cfg['num_workers'],
        pin_memory=True,
        persistent_workers=(cfg['num_workers'] > 0),
        drop_last=(len(train_ds) >= cfg['batch_size']),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg['batch_size'] * 2,
        shuffle=False,
        num_workers=cfg['num_workers'],
        pin_memory=True,
    )

    return train_loader, val_loader, test_ds, stats, train_sampler
