"""
Logging callback for training progress.

Logs training metrics to console using Rich formatting.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, Optional

from src.training.callbacks.base_callback import BaseCallback


class LoggingCallback(BaseCallback):
    """Log training progress to console and history.csv.

    Uses Rich library for formatted console output with tables
    and progress indicators, and saves metrics to a CSV for plotting.

    Attributes:
        log_every_n_steps: Log every N training steps.
        history: List of dictionaries storing epoch metrics.
    """

    def __init__(self, log_every_n_steps: int = 10) -> None:
        """Initialize the logging callback.

        Args:
            log_every_n_steps: Logging frequency in steps.
        """
        super().__init__()
        self.log_every_n_steps = log_every_n_steps
        self.history = []
        self.csv_path = None

    def on_train_start(self, trainer: Any) -> None:
        """Log training start information."""
        checkpoint_dir = Path(trainer.config.training.checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = checkpoint_dir / "history.csv"
        
        # Initialize CSV with headers
        with open(self.csv_path, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['epoch', 'train_loss', 'train_acc', 'val_loss', 'val_acc'])

    def on_epoch_end(
        self,
        trainer: Any,
        epoch: int,
        metrics: Optional[Dict[str, float]] = None,
    ) -> None:
        """Log epoch training metrics."""
        if metrics:
            # Add a new entry for this epoch
            if len(self.history) <= epoch:
                self.history.append({'epoch': epoch + 1})
            self.history[epoch].update({
                'train_loss': metrics.get('loss', 0.0),
                'train_acc': metrics.get('accuracy', 0.0)
            })

    def on_validation_end(
        self,
        trainer: Any,
        epoch: int,
        metrics: Optional[Dict[str, float]] = None,
    ) -> None:
        """Log validation metrics."""
        if metrics:
            if len(self.history) <= epoch:
                self.history.append({'epoch': epoch + 1})
            self.history[epoch].update({
                'val_loss': metrics.get('loss', 0.0),
                'val_acc': metrics.get('accuracy', 0.0)
            })
            
            # Write the current epoch to CSV since validation is the end of the epoch step
            entry = self.history[epoch]
            with open(self.csv_path, mode='a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    entry.get('epoch', epoch + 1),
                    entry.get('train_loss', 0.0),
                    entry.get('train_acc', 0.0),
                    entry.get('val_loss', 0.0),
                    entry.get('val_acc', 0.0)
                ])

    def on_train_end(self, trainer: Any) -> None:
        """Log training completion summary."""
        pass

