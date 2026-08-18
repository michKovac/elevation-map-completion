"""
Filesystem layout for scripts and evaluation tools.

Code, data and experiment outputs are decoupled so the repository works from any
checkout location. Three environment variables override the defaults:

    ELEVCOMP_ROOT        where runs/ and reports/ are written  (default: repo root)
    ELEVCOMP_DATA_ROOT   generated elevation dataset           (default: <root>/datasets/elevation_dataset)
    ELEVCOMP_EXPERIMENT  experiment directory used by the      (default: newest runs/cv_*)
                         evaluation tools
"""
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = REPO_ROOT / "configs"


def project_root() -> Path:
    """Root for generated artefacts (runs/, reports/)."""
    return Path(os.environ.get("ELEVCOMP_ROOT", REPO_ROOT)).expanduser().resolve()


def runs_dir() -> Path:
    return project_root() / "runs"


def reports_dir() -> Path:
    return project_root() / "reports"


def data_root() -> Path:
    """Root of the generated elevation dataset (one subdirectory per environment)."""
    env = os.environ.get("ELEVCOMP_DATA_ROOT")
    return Path(env).expanduser().resolve() if env else project_root() / "datasets" / "elevation_dataset"


def default_experiment() -> Path:
    """
    Experiment directory the evaluation tools operate on when --exp_dir is omitted:
    $ELEVCOMP_EXPERIMENT if set, otherwise the most recent runs/cv_* directory.
    """
    env = os.environ.get("ELEVCOMP_EXPERIMENT")
    if env:
        return Path(env).expanduser().resolve()

    candidates = sorted((p for p in runs_dir().glob("cv_*") if p.is_dir()),
                        key=lambda p: p.name)
    if not candidates:
        raise SystemExit(
            f"No experiment found in {runs_dir()}.\n"
            "Pass --exp_dir <dir>, set ELEVCOMP_EXPERIMENT, or train one first:\n"
            "  python scripts/train_cv.py --name resnet34"
        )
    return candidates[-1]
