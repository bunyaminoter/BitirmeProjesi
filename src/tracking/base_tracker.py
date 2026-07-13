"""
Abstract base class for experiment trackers.

Defines the interface for logging metrics, hyperparameters, and
artifacts during training. Implementations can wrap TensorBoard,
Weights & Biases, MLflow, or any other tracking backend.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional


class BaseExperimentTracker(ABC):
    """Abstract experiment tracker interface.

    Subclasses implement the tracking backend (TensorBoard, W&B, etc.).
    All tracking calls go through this interface so backends can be
    swapped without changing training code.
    """

    @abstractmethod
    def log_metrics(
        self,
        metrics: Dict[str, float],
        step: int,
        prefix: str = "",
    ) -> None:
        """Log scalar metrics.

        Args:
            metrics: Dictionary of metric name → value.
            step: Global step or epoch number.
            prefix: Optional prefix for metric names (e.g., 'train/').
        """
        ...

    @abstractmethod
    def log_hyperparameters(self, params: Dict[str, Any]) -> None:
        """Log experiment hyperparameters.

        Args:
            params: Dictionary of hyperparameter name → value.
        """
        ...

    @abstractmethod
    def log_artifact(self, path: str, name: Optional[str] = None) -> None:
        """Log a file artifact (checkpoint, config, etc.).

        Args:
            path: Path to the artifact file.
            name: Optional name for the artifact.
        """
        ...

    @abstractmethod
    def close(self) -> None:
        """Flush and close the tracker."""
        ...

    def __enter__(self) -> BaseExperimentTracker:
        """Context manager entry."""
        return self

    def __exit__(self, *args: Any) -> None:
        """Context manager exit."""
        self.close()
