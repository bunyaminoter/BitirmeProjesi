"""Core infrastructure: registry, configuration, and shared types."""

from src.core.registry import Registry
from src.core.config import (
    ExperimentConfig,
    DatasetConfig,
    ModelConfig,
    TrainingConfig,
    AugmentationConfig,
    load_config,
)
from src.core.types import (
    LandmarkResult,
    HandCrop,
    FrameBatch,
    BatchDict,
)

__all__ = [
    "Registry",
    "ExperimentConfig",
    "DatasetConfig",
    "ModelConfig",
    "TrainingConfig",
    "AugmentationConfig",
    "load_config",
    "LandmarkResult",
    "HandCrop",
    "FrameBatch",
    "BatchDict",
]
