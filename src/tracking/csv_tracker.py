"""
CSV-based experiment tracker for simple metric logging.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.core.registry import TRACKER_REGISTRY
from src.tracking.base_tracker import BaseExperimentTracker


@TRACKER_REGISTRY.register("csv")
class CSVTracker(BaseExperimentTracker):
    """CSV file-based experiment tracker.

    Writes metrics to a CSV file, one row per logging step.
    Simple and portable — no external dependencies required.

    Attributes:
        log_path: Path to the CSV log file.
    """

    def __init__(self, log_dir: str | Path, filename: str = "metrics.csv") -> None:
        """Initialize CSV tracker.

        Args:
            log_dir: Directory for log files.
            filename: CSV file name.
        """
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = log_dir / filename
        self._header_written = self.log_path.exists()
        self._columns: List[str] = []

    def log_metrics(
        self,
        metrics: Dict[str, float],
        step: int,
        prefix: str = "",
    ) -> None:
        """Append metrics as a row to the CSV file."""
        row = {"step": step}
        for name, value in metrics.items():
            key = f"{prefix}{name}" if prefix else name
            row[key] = value

        # Write header if needed
        if not self._header_written:
            self._columns = list(row.keys())
            with open(self.log_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=self._columns)
                writer.writeheader()
            self._header_written = True

        # Update columns if new metrics appear
        new_cols = [k for k in row if k not in self._columns]
        if new_cols:
            self._columns.extend(new_cols)

        with open(self.log_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f, fieldnames=self._columns, extrasaction="ignore"
            )
            writer.writerow(row)

    def log_hyperparameters(self, params: Dict[str, Any]) -> None:
        """Log hyperparameters to a separate JSON file."""
        import json

        hp_path = self.log_path.parent / "hyperparameters.json"
        with open(hp_path, "w", encoding="utf-8") as f:
            json.dump(params, f, indent=2, default=str)

    def log_artifact(self, path: str, name: Optional[str] = None) -> None:
        """Log artifact reference (no-op for CSV tracker)."""
        pass

    def close(self) -> None:
        """Close the tracker (no-op for CSV)."""
        pass
