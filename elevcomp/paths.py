"""
Filesystem layout.

Code, data and training outputs are decoupled so the repository works from any
checkout location. Two environment variables override the defaults:

    ELEVCOMP_ROOT        where runs/ is written      (default: repo root)
    ELEVCOMP_DATA_ROOT   generated elevation dataset (default: <root>/datasets/elevation_dataset)
"""
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = REPO_ROOT / "configs"


def project_root() -> Path:
    """Root for generated artefacts."""
    return Path(os.environ.get("ELEVCOMP_ROOT", REPO_ROOT)).expanduser().resolve()


def runs_dir() -> Path:
    return project_root() / "runs"


def data_root() -> Path:
    """Root of the generated elevation dataset (one subdirectory per environment)."""
    env = os.environ.get("ELEVCOMP_DATA_ROOT")
    return Path(env).expanduser().resolve() if env else project_root() / "datasets" / "elevation_dataset"


