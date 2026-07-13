"""
TensorBoard experiment tracker implementation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional


from src.core.registry import TRACKER_REGISTRY
from src.tracking.base_tracker import BaseExperimentTracker


@TRACKER_REGISTRY.register("tensorboard")
class TensorBoardTracker(BaseExperimentTracker):
    """TensorBoard-based experiment tracker.

    Attributes:
        log_dir: Directory for TensorBoard log files.
        writer: SummaryWriter instance (lazily initialized).
    """

    def __init__(self, log_dir: str | Path) -> None:
        """Initialize TensorBoard tracker.

        Args:
            log_dir: Directory for TensorBoard log files.
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._writer = None

    @property
    def writer(self) -> Any:
        """Lazily initialize the SummaryWriter."""
        if self._writer is None:
            from torch.utils.tensorboard import SummaryWriter
            self._writer = SummaryWriter(log_dir=str(self.log_dir))
        return self._writer

    def log_metrics(
        self,
        metrics: Dict[str, float],
        step: int,
        prefix: str = "",
    ) -> None:
        """Log scalar metrics to TensorBoard."""
        for name, value in metrics.items():
            tag = f"{prefix}{name}" if prefix else name
            self.writer.add_scalar(tag, value, step)

    def log_hyperparameters(self, params: Dict[str, Any]) -> None:
        """Log hyperparameters to TensorBoard."""
        self.writer.add_hparams(params, {})

    def log_artifact(self, path: str, name: Optional[str] = None) -> None:
        """Log artifact (text reference) to TensorBoard."""
        self.writer.add_text(
            name or "artifact",
            f"Artifact saved at: {path}",
        )

    def close(self) -> None:
        """Flush and close the TensorBoard writer."""
        if self._writer is not None:
            self._writer.flush()
            self._writer.close()
