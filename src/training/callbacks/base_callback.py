"""
Abstract base class for training callbacks.

Callbacks allow injecting custom behavior at specific points in the
training lifecycle without modifying the Trainer class.
"""

from __future__ import annotations

from typing import Any, Dict, Optional


class BaseCallback:
    """Abstract base class for training callbacks.

    Override any of the hook methods to inject custom behavior.
    All methods receive the Trainer instance and relevant context.
    """

    def on_train_start(self, trainer: Any) -> None:
        """Called at the beginning of training."""
        pass

    def on_train_end(self, trainer: Any) -> None:
        """Called at the end of training."""
        pass

    def on_epoch_start(self, trainer: Any, epoch: int) -> None:
        """Called at the beginning of each epoch."""
        pass

    def on_epoch_end(
        self,
        trainer: Any,
        epoch: int,
        metrics: Optional[Dict[str, float]] = None,
    ) -> None:
        """Called at the end of each epoch."""
        pass

    def on_validation_end(
        self,
        trainer: Any,
        epoch: int,
        metrics: Optional[Dict[str, float]] = None,
    ) -> None:
        """Called at the end of validation."""
        pass

    def on_batch_start(self, trainer: Any, batch_idx: int) -> None:
        """Called at the beginning of each training batch."""
        pass

    def on_batch_end(
        self,
        trainer: Any,
        batch_idx: int,
        loss: float = 0.0,
    ) -> None:
        """Called at the end of each training batch."""
        pass
