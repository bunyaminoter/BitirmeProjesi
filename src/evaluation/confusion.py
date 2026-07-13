"""
Confusion matrix computation and visualization utilities.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import numpy.typing as npt


class ConfusionMatrix:
    """Compute and store a confusion matrix.

    Attributes:
        num_classes: Number of classes.
        matrix: The confusion matrix array (num_classes, num_classes).
    """

    def __init__(self, num_classes: int) -> None:
        """Initialize the confusion matrix.

        Args:
            num_classes: Number of output classes.
        """
        self.num_classes = num_classes
        self.matrix = np.zeros((num_classes, num_classes), dtype=np.int64)

    def update(
        self,
        predictions: npt.NDArray[np.int64],
        labels: npt.NDArray[np.int64],
    ) -> None:
        """Update the confusion matrix with a batch of predictions.

        Args:
            predictions: (B,) predicted class indices.
            labels: (B,) ground truth class indices.
        """
        for pred, label in zip(predictions, labels):
            self.matrix[label, pred] += 1

    def reset(self) -> None:
        """Reset the confusion matrix to zeros."""
        self.matrix = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)

    def get_accuracy_per_class(self) -> npt.NDArray[np.float64]:
        """Compute per-class accuracy from the confusion matrix.

        Returns:
            (num_classes,) array of per-class accuracies.
        """
        row_sums = self.matrix.sum(axis=1)
        # Avoid division by zero
        row_sums = np.maximum(row_sums, 1)
        return np.diag(self.matrix).astype(np.float64) / row_sums

    def save(self, path: str, class_names: Optional[List[str]] = None) -> None:
        """Save the confusion matrix to a CSV file.

        Args:
            path: Output file path.
            class_names: Optional list of class names for headers.
        """
        # TODO: Save as CSV with optional class name headers
        np.savetxt(path, self.matrix, delimiter=",", fmt="%d")
