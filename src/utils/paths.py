"""
Project and data path utilities.

Supports local development and Google Colab by resolving paths relative
to a configurable data root (project root by default).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.core.config import DatasetConfig, ExperimentConfig


def is_colab() -> bool:
    """Return True when running inside Google Colab."""
    try:
        import google.colab  # noqa: F401

        return True
    except ImportError:
        return False


def get_project_root() -> Path:
    """Return the repository root directory."""
    env_root = os.environ.get("ASL_PROJECT_ROOT")
    if env_root:
        return Path(env_root).resolve()

    # src/utils/paths.py -> project root is two levels up
    return Path(__file__).resolve().parents[2]


def get_data_root() -> Path:
    """Return the root directory for dataset files (videos, JSON, cache).

    Override with ASL_DATA_ROOT for Colab Drive mounts, e.g.:
        /content/drive/MyDrive/BitirmeProjesi
    """
    env_root = os.environ.get("ASL_DATA_ROOT")
    if env_root:
        return Path(env_root).resolve()
    return get_project_root()


def get_mediapipe_models_dir() -> Path:
    """Directory containing MediaPipe .task model files."""
    env_dir = os.environ.get("ASL_MEDIAPIPE_MODELS_DIR")
    if env_dir:
        return Path(env_dir).resolve()
    return get_project_root() / "models" / "mediapipe"


def resolve_path(path: str | Path, base: Path | None = None) -> Path:
    """Resolve a config path against a base directory."""
    path = Path(path)
    if path.is_absolute():
        return path
    base = base or get_data_root()
    return (base / path).resolve()


def apply_data_root(dataset_config: DatasetConfig, data_root: Path | None = None) -> None:
    """Convert relative dataset paths to absolute paths under data_root."""
    root = data_root or get_data_root()
    dataset_config.annotation_file = str(resolve_path(dataset_config.annotation_file, root))
    dataset_config.video_dir = str(resolve_path(dataset_config.video_dir, root))
    dataset_config.class_list_file = str(resolve_path(dataset_config.class_list_file, root))


def apply_experiment_paths(
    config: ExperimentConfig,
    data_root: Path | None = None,
    output_dir: str | Path | None = None,
    cache_dir: str | Path | None = None,
) -> None:
    """Apply Colab/local path overrides to an experiment config."""
    root = data_root or get_data_root()
    apply_data_root(config.dataset, root)

    if output_dir is not None:
        config.output_dir = str(resolve_path(output_dir, root))
        config.training.checkpoint_dir = str(
            resolve_path(config.training.checkpoint_dir, root)
        )

    if cache_dir is not None:
        os.environ["ASL_CACHE_DIR"] = str(resolve_path(cache_dir, root))


def get_effective_num_workers(requested: int) -> int:
    """Return a safe DataLoader worker count for the current environment."""
    if is_colab():
        return 0
    if os.name == "nt" and requested > 0:
        # Windows multiprocessing in DataLoader is often problematic
        return 0
    return requested


def ensure_dir(path: str | Path) -> Path:
    """Create a directory if needed and return its Path."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path
