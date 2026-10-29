"""Tests for evaluation metrics."""

from __future__ import annotations

import numpy as np
import pytest

from src.evaluation.metrics import MetricCalculator


class TestMetricCalculator:
    """Test the MetricCalculator."""

    def test_perfect_accuracy(self):
        """Test 100% accuracy case."""
        calc = MetricCalculator(num_classes=3)
        preds = np.array([0, 1, 2, 0, 1])
        labels = np.array([0, 1, 2, 0, 1])
        calc.update(preds, labels)

        result = calc.compute()
        assert result.accuracy == 1.0

    def test_zero_accuracy(self):
        """Test 0% accuracy case."""
        calc = MetricCalculator(num_classes=3)
        preds = np.array([1, 2, 0])
        labels = np.array([0, 1, 2])
        calc.update(preds, labels)

        result = calc.compute()
        assert result.accuracy == 0.0

    def test_partial_accuracy(self):
        """Test partial accuracy."""
        calc = MetricCalculator(num_classes=2)
        preds = np.array([0, 0, 1, 1])
        labels = np.array([0, 1, 1, 0])
        calc.update(preds, labels)

        result = calc.compute()
        assert result.accuracy == 0.5

    def test_reset(self):
        """Test that reset clears accumulated data."""
        calc = MetricCalculator(num_classes=2)
        calc.update(np.array([0]), np.array([0]))
        calc.reset()

        result = calc.compute()
        # After reset, with no data, accuracy should handle gracefully
        assert isinstance(result.accuracy, float)

    def test_to_dict(self):
        """Test metric result serialization."""
        calc = MetricCalculator(num_classes=2)
        preds = np.array([0, 1])
        labels = np.array([0, 1])
        calc.update(preds, labels)

        result = calc.compute()
        d = result.to_dict()
        assert "accuracy" in d
        assert "f1_macro" in d
        assert "top5_accuracy" in d

    def test_macro_and_weighted_f1_average_per_class_scores(self):
        calc = MetricCalculator(num_classes=2)
        calc.update(
            predictions=np.array([0, 0, 0, 1]),
            labels=np.array([0, 0, 1, 1]),
        )

        result = calc.compute()
        # Class 0: precision 2/3, recall 1, F1 4/5.
        # Class 1: precision 1, recall 1/2, F1 2/3.
        expected = (4 / 5 + 2 / 3) / 2
        assert result.f1_macro == pytest.approx(expected)
        assert result.f1_weighted == pytest.approx(expected)
        assert result.f1_macro != pytest.approx(
            2 * result.precision_macro * result.recall_macro
            / (result.precision_macro + result.recall_macro)
        )
