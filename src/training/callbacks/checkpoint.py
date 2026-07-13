"""
Checkpoint saving callback.

Saves model checkpoints at configurable intervals. Supports both
periodic saving and best-metric saving.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from src.training.callbacks.base_callback import BaseCallback


class CheckpointCallback(BaseCallback):
    """Save model checkpoints during training.

    Attributes:
        save_dir: Directory to save checkpoints.
        monitor: Metric name to monitor for best-model saving.
        save_every_n_epochs: Save a checkpoint every N epochs.
        save_best_only: Only save when the monitored metric improves.
        best_value: Best observed value of the monitored metric.
    """

    def __init__(
        self,
        save_dir: str | Path,
        monitor: str = "val_accuracy",
        mode: str = "max",
        save_every_n_epochs: int = 1,
        save_best_only: bool = True,
    ) -> None:
        """Initialize the checkpoint callback.

        Args:
            save_dir: Directory for checkpoint files.
            monitor: Metric to monitor for best-model saving.
            mode: 'max' if higher is better, 'min' if lower is better.
            save_every_n_epochs: Periodic save interval.
            save_best_only: If True, only save when metric improves.
        """
        super().__init__()
        self.save_dir = Path(save_dir)
        self.monitor = monitor
        self.mode = mode
        self.save_every_n_epochs = save_every_n_epochs
        self.save_best_only = save_best_only
        self.best_value: Optional[float] = None

    def on_validation_end(
        self,
        trainer: Any,
        epoch: int,
        metrics: Optional[Dict[str, float]] = None,
    ) -> None:
        """Save checkpoint after validation if criteria are met."""
        if metrics is None:
            return

        current = metrics.get(self.monitor)
        if current is None:
            return

        is_best = False
        if self.best_value is None:
            is_best = True
        elif self.mode == "max" and current > self.best_value:
            is_best = True
        elif self.mode == "min" and current < self.best_value:
            is_best = True

        if is_best:
            self.best_value = current
            trainer.best_metric = current
            trainer.save_checkpoint(self.save_dir / "best_model.pt")

        if not self.save_best_only and (epoch + 1) % self.save_every_n_epochs == 0:
            trainer.save_checkpoint(self.save_dir / f"checkpoint_epoch_{epoch + 1}.pt")
