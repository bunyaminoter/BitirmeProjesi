"""
Mixup data augmentation for training.

Mixes two samples' features and labels to create soft training targets,
which improves generalization and reduces overfitting.

Reference: Zhang et al. "mixup: Beyond Empirical Risk Minimization" (ICLR 2018)

Usage:
    mixed_batch, targets_a, targets_b, lam = mixup_batch(batch, alpha=0.2)
    logits = model(mixed_batch)
    loss = mixup_criterion(criterion, logits, targets_a, targets_b, lam)
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn


def mixup_batch(
    batch: Dict[str, Any],
    alpha: float = 0.2,
) -> Tuple[Dict[str, Any], torch.Tensor, torch.Tensor, float]:
    """Apply Mixup augmentation to a batch.

    Samples a mixing coefficient lambda from Beta(alpha, alpha),
    then mixes all tensor fields of two randomly permuted samples.

    Args:
        batch: Dictionary containing batch tensors.
        alpha: Mixup interpolation strength. Higher = more mixing.
              When alpha=0, no mixing is applied.

    Returns:
        Tuple of:
            - mixed_batch: Batch with mixed features
            - targets_a: Original labels
            - targets_b: Shuffled labels
            - lam: Mixing coefficient (lambda)
    """
    if alpha <= 0:
        return batch, batch["labels"], batch["labels"], 1.0

    # Sample lambda from Beta distribution
    lam = np.random.beta(alpha, alpha)
    # Ensure lambda >= 0.5 so the "primary" sample dominates
    lam = max(lam, 1.0 - lam)

    labels = batch["labels"]
    batch_size = labels.size(0)

    # Random permutation for mixing partners
    index = torch.randperm(batch_size, device=labels.device)

    # Mix all tensor fields except labels and non-tensor fields
    mixed_batch: Dict[str, Any] = {}
    for key, value in batch.items():
        if key in ("labels", "mask"):
            mixed_batch[key] = value  # Keep labels & mask unchanged
        elif isinstance(value, torch.Tensor):
            mixed_batch[key] = lam * value + (1.0 - lam) * value[index]
        else:
            mixed_batch[key] = value  # Keep non-tensor fields as-is

    targets_a = labels
    targets_b = labels[index]

    return mixed_batch, targets_a, targets_b, lam


def mixup_criterion(
    criterion: nn.Module,
    logits: torch.Tensor,
    targets_a: torch.Tensor,
    targets_b: torch.Tensor,
    lam: float,
) -> torch.Tensor:
    """Compute mixed loss for Mixup training.

    The loss is a weighted combination of losses computed against
    both the original and shuffled targets:
        loss = lam * L(logits, targets_a) + (1 - lam) * L(logits, targets_b)

    Args:
        criterion: Loss function.
        logits: Model predictions (B, C).
        targets_a: Original labels (B,).
        targets_b: Shuffled labels (B,).
        lam: Mixing coefficient.

    Returns:
        Scalar loss tensor.
    """
    return lam * criterion(logits, targets_a) + (1.0 - lam) * criterion(logits, targets_b)
