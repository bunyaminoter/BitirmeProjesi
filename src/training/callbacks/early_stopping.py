"""
Early stopping callback.

Stops training when the monitored metric stops improving for a
specified number of epochs (patience).
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from src.training.callbacks.base_callback import BaseCallback


class EarlyStoppingCallback(BaseCallback):
    """Stop training when a metric stops improving.

    Attributes:
        monitor: Metric name to monitor.
        patience: Number of epochs to wait for improvement.
        mode: 'max' if higher is better, 'min' if lower is better.
        min_delta: Minimum change to qualify as an improvement.
        should_stop: Set to True when training should stop.
        counter: Epochs without improvement.
        best_value: Best observed metric value.
    """

    def __init__(
        self,
        monitor: str = "val_accuracy",
        patience: int = 15,
        mode: str = "max",
        min_delta: float = 0.001,
    ) -> None:
        """Initialize early stopping.

        Args:
            monitor: Metric to monitor.
            patience: Epochs to wait before stopping.
            mode: 'max' or 'min'.
            min_delta: Minimum improvement threshold.
        """
        super().__init__()
        self.monitor = monitor
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.should_stop = False
        self.counter = 0
        self.best_value: Optional[float] = None

    def on_validation_end(
        self,
        trainer: Any,
        epoch: int,
        metrics: Optional[Dict[str, float]] = None,
    ) -> None:
        """Check if training should stop after validation."""
        if metrics is None:
            return

        current = metrics.get(self.monitor)
        if current is None:
            return

        if self.best_value is None:
            self.best_value = current
            self.counter = 0
            return

        improved = False
        if self.mode == "max":
            improved = current > (self.best_value + self.min_delta)
        elif self.mode == "min":
            improved = current < (self.best_value - self.min_delta)

        if improved:
            self.best_value = current
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
