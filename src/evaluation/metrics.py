"""
Metric computation for sign language recognition.

Computes: Accuracy, Top-5 Accuracy, Precision, Recall, F1 Score,
and per-class breakdowns. All metrics are computed from accumulated
predictions over a full evaluation pass.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np


@dataclass
class MetricResult:
    """Container for computed metric results.

    Attributes:
        accuracy: Overall accuracy (Top-1).
        top5_accuracy: Top-5 accuracy.
        precision_macro: Macro-averaged precision.
        recall_macro: Macro-averaged recall.
        f1_macro: Macro-averaged F1 score.
        precision_weighted: Weighted-averaged precision.
        recall_weighted: Weighted-averaged recall.
        f1_weighted: Weighted-averaged F1 score.
        per_class_accuracy: Per-class accuracy dict.
    """

    accuracy: float = 0.0
    top5_accuracy: float = 0.0
    precision_macro: float = 0.0
    recall_macro: float = 0.0
    f1_macro: float = 0.0
    precision_weighted: float = 0.0
    recall_weighted: float = 0.0
    f1_weighted: float = 0.0
    per_class_accuracy: Dict[int, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, float]:
        """Convert to a flat dictionary for logging.

        Returns:
            Dictionary of metric name → value.
        """
        return {
            "accuracy": self.accuracy,
            "top5_accuracy": self.top5_accuracy,
            "precision_macro": self.precision_macro,
            "recall_macro": self.recall_macro,
            "f1_macro": self.f1_macro,
            "precision_weighted": self.precision_weighted,
            "recall_weighted": self.recall_weighted,
            "f1_weighted": self.f1_weighted,
        }


class MetricCalculator:
    """Accumulates predictions and computes evaluation metrics.

    Designed for batch-by-batch accumulation during evaluation,
    then computing all metrics at once.

    Attributes:
        num_classes: Number of output classes.
    """

    def __init__(self, num_classes: int) -> None:
        """Initialize the metric calculator.

        Args:
            num_classes: Number of classes.
        """
        self.num_classes = num_classes
        self._all_predictions: List[int] = []
        self._all_labels: List[int] = []
        self._all_top5_predictions: List[List[int]] = []

    def reset(self) -> None:
        """Reset accumulated predictions."""
        self._all_predictions = []
        self._all_labels = []
        self._all_top5_predictions = []

    def update(
        self,
        predictions: np.ndarray,
        labels: np.ndarray,
        logits: Optional[np.ndarray] = None,
    ) -> None:
        """Accumulate a batch of predictions.

        Args:
            predictions: (B,) predicted class indices.
            labels: (B,) ground truth class indices.
            logits: (B, C) optional logits for top-5 computation.
        """
        self._all_predictions.extend(predictions.tolist())
        self._all_labels.extend(labels.tolist())

        if logits is not None:
            top5 = np.argsort(logits, axis=-1)[:, -5:]
            self._all_top5_predictions.extend(top5.tolist())

    def compute(self) -> MetricResult:
        """Compute all metrics from accumulated predictions.

        Returns:
            MetricResult with all computed metrics.
        """
        if not self._all_predictions:
            return MetricResult()

        preds = np.array(self._all_predictions, dtype=np.int64)
        labels = np.array(self._all_labels, dtype=np.int64)

        # Top-1 accuracy
        accuracy = float(np.mean(preds == labels))

        # Top-5 accuracy
        top5_accuracy = 0.0
        if self._all_top5_predictions:
            correct = sum(
                label in top5
                for label, top5 in zip(
                    self._all_labels, self._all_top5_predictions
                )
            )
            top5_accuracy = correct / len(self._all_labels)

        # Per-class metrics
        per_class_precision: Dict[int, float] = {}
        per_class_recall: Dict[int, float] = {}
        per_class_accuracy: Dict[int, float] = {}

        for cls in range(self.num_classes):
            cls_mask = labels == cls
            if cls_mask.sum() == 0:
                continue

            tp = int(((preds == cls) & (labels == cls)).sum())
            fp = int(((preds == cls) & (labels != cls)).sum())
            fn = int(((preds != cls) & (labels == cls)).sum())

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

            per_class_precision[cls] = precision
            per_class_recall[cls] = recall
            per_class_accuracy[cls] = recall  # Per-class accuracy == recall

        # Macro averages
        active_classes = len(per_class_precision)
        precision_macro = (
            sum(per_class_precision.values()) / active_classes
            if active_classes > 0
            else 0.0
        )
        recall_macro = (
            sum(per_class_recall.values()) / active_classes
            if active_classes > 0
            else 0.0
        )
        f1_macro = (
            2 * precision_macro * recall_macro / (precision_macro + recall_macro)
            if (precision_macro + recall_macro) > 0
            else 0.0
        )

        # Weighted averages
        class_counts = np.bincount(labels, minlength=self.num_classes)
        total_samples = len(labels)

        precision_weighted = sum(
            per_class_precision.get(cls, 0.0) * class_counts[cls]
            for cls in range(self.num_classes)
        ) / total_samples if total_samples > 0 else 0.0

        recall_weighted = sum(
            per_class_recall.get(cls, 0.0) * class_counts[cls]
            for cls in range(self.num_classes)
        ) / total_samples if total_samples > 0 else 0.0

        f1_weighted = (
            2 * precision_weighted * recall_weighted
            / (precision_weighted + recall_weighted)
            if (precision_weighted + recall_weighted) > 0
            else 0.0
        )

        return MetricResult(
            accuracy=accuracy,
            top5_accuracy=top5_accuracy,
            precision_macro=precision_macro,
            recall_macro=recall_macro,
            f1_macro=f1_macro,
            precision_weighted=precision_weighted,
            recall_weighted=recall_weighted,
            f1_weighted=f1_weighted,
            per_class_accuracy=per_class_accuracy,
        )
