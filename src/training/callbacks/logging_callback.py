"""
Logging callback for training progress.

Logs training metrics to console using Rich formatting.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from src.training.callbacks.base_callback import BaseCallback


class LoggingCallback(BaseCallback):
    """Log training progress to console.

    Uses Rich library for formatted console output with tables
    and progress indicators.

    Attributes:
        log_every_n_steps: Log every N training steps.
    """

    def __init__(self, log_every_n_steps: int = 10) -> None:
        """Initialize the logging callback.

        Args:
            log_every_n_steps: Logging frequency in steps.
        """
        super().__init__()
        self.log_every_n_steps = log_every_n_steps

    def on_train_start(self, trainer: Any) -> None:
        """Log training start information."""
        # TODO: Log experiment config, model summary, device info
        pass

    def on_epoch_end(
        self,
        trainer: Any,
        epoch: int,
        metrics: Optional[Dict[str, float]] = None,
    ) -> None:
        """Log epoch training metrics."""
        # TODO: Log epoch summary with Rich table
        pass

    def on_validation_end(
        self,
        trainer: Any,
        epoch: int,
        metrics: Optional[Dict[str, float]] = None,
    ) -> None:
        """Log validation metrics."""
        # TODO: Log validation metrics with Rich formatting
        pass

    def on_train_end(self, trainer: Any) -> None:
        """Log training completion summary."""
        # TODO: Log final summary
        pass
