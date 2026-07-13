"""
Evaluation orchestrator.

Runs the model over a dataset split, collects predictions,
and computes all evaluation metrics.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.evaluation.metrics import MetricCalculator, MetricResult

logger = logging.getLogger(__name__)


class Evaluator:
    """Orchestrates model evaluation over a DataLoader.

    Handles the evaluation loop, prediction collection, and
    metric computation. Supports mixed precision evaluation.

    Attributes:
        model: The model to evaluate.
        device: Device to run evaluation on.
        num_classes: Number of output classes.
    """

    def __init__(
        self,
        model: nn.Module,
        num_classes: int,
        device: Optional[torch.device] = None,
    ) -> None:
        """Initialize the evaluator.

        Args:
            model: Trained model to evaluate.
            num_classes: Number of output classes.
            device: Target device (auto-detected if None).
        """
        self.model = model
        self.num_classes = num_classes
        self.device = device or torch.device("cpu")
        self.metric_calculator = MetricCalculator(num_classes)

    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader) -> MetricResult:
        """Run evaluation over the full DataLoader.

        Args:
            dataloader: DataLoader for the evaluation split.

        Returns:
            MetricResult with all computed metrics.
        """
        self.model.eval()
        self.metric_calculator.reset()

        total_batches = len(dataloader)

        for batch_idx, batch in enumerate(dataloader):
            # Move batch to device
            batch_device: Dict[str, Any] = {}
            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    batch_device[key] = value.to(self.device)
                else:
                    batch_device[key] = value

            labels = batch_device["labels"]

            # Forward pass
            logits = self.model(batch_device)  # (B, num_classes)

            # Convert to numpy for metric computation
            logits_np = logits.cpu().numpy()
            labels_np = labels.cpu().numpy()
            predictions = logits_np.argmax(axis=-1)

            # Update metric calculator with this batch
            self.metric_calculator.update(
                predictions=predictions,
                labels=labels_np,
                logits=logits_np,
            )

            if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == total_batches:
                logger.info(
                    f"  Evaluation: {batch_idx + 1}/{total_batches} batches"
                )

        result = self.metric_calculator.compute()

        logger.info(
            f"Evaluation complete — "
            f"Accuracy: {result.accuracy:.4f}, "
            f"Top-5 Accuracy: {result.top5_accuracy:.4f}, "
            f"F1 (macro): {result.f1_macro:.4f}"
        )

        return result
