"""
Environment discovery and leave-one-environment-out fold construction.

Fold design:
    fold k  →  TEST  = all trajectories of environment k (never seen in training)
               VAL   = last (sorted) trajectory/ies of each remaining environment
               TRAIN = all other trajectories of the remaining environments

Normalisation statistics are computed from TRAIN files only (no leakage from
the held-out environment); model selection uses VAL RMSE, so the test
environment never influences any training decision.
"""
from pathlib import Path

from torch.utils.data import DataLoader

from .dataset import ElevationDataset


def discover_environments(data_root: Path) -> dict:
    """
    Environments = subdirectories of data_root whose own subdirectories
    (trajectories) contain .npz samples. Flat directories (e.g. old 'train'
    dumps) are ignored. Returns {env: {trajectory: [relative file paths]}}.
    """
    envs = {}
    for env_dir in sorted(p for p in data_root.iterdir() if p.is_dir()):
        trajs = {}
        for traj_dir in sorted(p for p in env_dir.iterdir() if p.is_dir()):
            files = sorted(traj_dir.glob('*.npz'))
            if files:
                trajs[traj_dir.name] = [str(f.relative_to(data_root)) for f in files]
        if trajs:
            envs[env_dir.name] = trajs
    if not envs:
        raise FileNotFoundError(f'No environment/trajectory structure found in {data_root}')
    return envs


def build_folds(envs: dict, val_traj_per_env: int, debug_limit: int = 0) -> list:
    """
    Leave-one-environment-out folds. Validation trajectories are the LAST
    val_traj_per_env trajectories (sorted by name) of each training environment
    — deterministic and disjoint from training trajectories.
    """
    if debug_limit:
        envs = {e: {t: fs[:debug_limit] for t, fs in trajs.items()}
                for e, trajs in envs.items()}

    env_names = sorted(envs.keys())
    folds = []
    for k, test_env in enumerate(env_names):
        train_files, val_files, val_trajs = [], [], {}
        for env in env_names:
            if env == test_env:
                continue
            traj_names = sorted(envs[env].keys())
            v_names = traj_names[-val_traj_per_env:]
            val_trajs[env] = v_names
            for t in traj_names:
                (val_files if t in v_names else train_files).extend(envs[env][t])
        test_files = [f for t in sorted(envs[test_env]) for f in envs[test_env][t]]
        folds.append({
            'fold': k,
            'test_env': test_env,
            'val_trajectories': val_trajs,
            'n_train': len(train_files),
            'n_val': len(val_files),
            'n_test': len(test_files),
            'train_files': sorted(train_files),
            'val_files': sorted(val_files),
            'test_files': test_files,
        })
    return folds


def fold_paths(fold: dict, data_root: Path) -> dict:
    """Absolute file paths per split of one fold."""
    return {s: [data_root / f for f in fold[f'{s}_files']]
            for s in ('train', 'val', 'test')}


def make_datasets(cfg: dict, fold: dict, data_root: Path, stats: dict = None):
    """
    Build (train, val, test) datasets for one fold. When stats is None it is
    computed from the fold's TRAIN files (and returned for reuse).
    """
    paths = fold_paths(fold, data_root)
    aug_kw = dict(
        fix_mask_rotation=cfg.get('fix_mask_rotation', False),
        aug_hflip_p=cfg.get('aug_hflip_p', 0.5),
        aug_vflip_p=cfg.get('aug_vflip_p', 0.5),
        aug_rot90_p=cfg.get('aug_rot90_p', 0.75),
        aug_ray_p=cfg.get('aug_ray_p', 0.0),
        aug_ray_n_max=cfg.get('aug_ray_n_max', 3),
        aug_ray_angle_deg=cfg.get('aug_ray_angle_deg', [5, 40]),
    )
    train_ds = ElevationDataset(split='train', augment=True, stats=stats,
                                pad_to=cfg['pad_to'], files=paths['train'], **aug_kw)
    stats = train_ds.stats
    val_ds = ElevationDataset(split='val', augment=False, stats=stats,
                              pad_to=cfg['pad_to'], files=paths['val'], **aug_kw)
    test_ds = ElevationDataset(split='test', augment=False, stats=stats,
                               pad_to=cfg['pad_to'], files=paths['test'], **aug_kw)
    return train_ds, val_ds, test_ds, stats


def eval_loader(ds, cfg, batch_size=None):
    """Deterministic, augmentation-free loader for evaluation passes."""
    return DataLoader(
        ds, batch_size=batch_size or cfg['batch_size'] * 2, shuffle=False,
        num_workers=cfg['num_workers'], pin_memory=True,
    )
