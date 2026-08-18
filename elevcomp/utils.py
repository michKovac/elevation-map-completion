"""
Small shared helpers: reproducibility, environment fingerprint, CSV and
NaN-safe statistics used across the experiment drivers and tools.
"""
import csv
import platform
import random
import socket
import subprocess
from pathlib import Path

import numpy as np
import torch


# ── Reproducibility ───────────────────────────────────────────────────────────

def seed_everything(seed: int):
    """Seed python / numpy / torch (+CUDA) RNGs."""
    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def worker_init(worker_id: int):
    """DataLoader worker_init_fn — derive numpy/random seeds from torch's."""
    s = torch.initial_seed() % 2**32
    np.random.seed(s)
    random.seed(s)


def environment_info() -> dict:
    """Library versions, GPU names, git commit — persisted with every experiment."""
    try:
        git = subprocess.run(
            ['git', '-C', str(Path(__file__).parent), 'rev-parse', 'HEAD'],
            capture_output=True, text=True, timeout=5,
        )
        commit = git.stdout.strip() if git.returncode == 0 else None
    except Exception:
        commit = None
    try:
        import segmentation_models_pytorch as smp
        smp_ver = smp.__version__
    except Exception:
        smp_ver = None
    return {
        'python': platform.python_version(),
        'torch': torch.__version__,
        'numpy': np.__version__,
        'segmentation_models_pytorch': smp_ver,
        'cuda': torch.version.cuda,
        'cudnn': torch.backends.cudnn.version(),
        'cudnn_benchmark': torch.backends.cudnn.benchmark,
        'gpus': [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        'hostname': socket.gethostname(),
        'git_commit': commit,
    }


# ── NaN-safe statistics ───────────────────────────────────────────────────────

def nanmean(v) -> float:
    v = np.asarray(v, dtype=np.float64)
    return float(np.nanmean(v)) if np.isfinite(v).any() else float('nan')


def nanstd(v) -> float:
    v = np.asarray(v, dtype=np.float64)
    return float(np.nanstd(v)) if np.isfinite(v).any() else float('nan')


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    """Pearson correlation without RuntimeWarnings on (near-)constant inputs."""
    x = x.astype(np.float64) - x.mean()
    y = y.astype(np.float64) - y.mean()
    den = np.sqrt((x * x).sum() * (y * y).sum())
    return float((x * y).sum() / den) if den > 1e-12 else float('nan')


def aggregate_rows(rows: list) -> dict:
    """Per-metric mean/std/n over per-sample dicts (missing keys → NaN-skipped)."""
    keys = sorted({k for r in rows for k in r})
    out = {}
    for k in keys:
        v = np.array([r.get(k, np.nan) for r in rows], dtype=np.float64)
        out[k] = {'mean': nanmean(v), 'std': nanstd(v),
                  'n': int(np.isfinite(v).sum())}
    return out


# ── I/O ───────────────────────────────────────────────────────────────────────

def write_csv(path: Path, rows: list):
    """Write list-of-dicts to CSV; column order = first appearance."""
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)
