"""
Loss functions for sign language recognition.

Provides standard and custom loss functions, all accessible
through the global loss registry.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.core.registry import LOSS_REGISTRY


@LOSS_REGISTRY.register("cross_entropy")
class CrossEntropyLoss(nn.Module):
    """Standard cross-entropy loss with optional label smoothing.

    Attributes:
        label_smoothing: Label smoothing factor in [0, 1).
    """

    def __init__(self, label_smoothing: float = 0.0) -> None:
        """Initialize cross-entropy loss.

        Args:
            label_smoothing: Label smoothing factor.
        """
        super().__init__()
        self.label_smoothing = label_smoothing

    def forward(
        self, logits: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        """Compute cross-entropy loss.

        Args:
            logits: (B, C) predicted logits.
            targets: (B,) integer class labels.

        Returns:
            Scalar loss tensor.
        """
        return F.cross_entropy(
            logits, targets, label_smoothing=self.label_smoothing
        )


@LOSS_REGISTRY.register("focal")
class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance.

    Focuses training on hard-to-classify examples by down-weighting
    the loss contribution from well-classified examples.

    Attributes:
        alpha: Weighting factor for the minority class.
        gamma: Focusing parameter (higher = more focus on hard examples).
    """

    def __init__(self, alpha: float = 1.0, gamma: float = 2.0) -> None:
        """Initialize focal loss.

        Args:
            alpha: Class weight factor.
            gamma: Focusing parameter.
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(
        self, logits: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        """Compute focal loss.

        Args:
            logits: (B, C) predicted logits.
            targets: (B,) integer class labels.

        Returns:
            Scalar loss tensor.
        """
        ce_loss = F.cross_entropy(logits, targets, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()
