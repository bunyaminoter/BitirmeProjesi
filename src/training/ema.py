"""
Exponential Moving Average (EMA) for model weights.

Maintains a shadow copy of model parameters, updated with exponential
moving average at each training step. Produces smoother, better-
generalizing models when used for inference.

Usage:
    ema = ModelEMA(model, decay=0.999)
    # In training loop:
    ema.update(model)
    # For validation:
    ema.apply(model)      # load EMA weights
    validate(model)
    ema.restore(model)    # restore training weights
"""

from __future__ import annotations

import copy
import logging
from typing import Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class ModelEMA:
    """Exponential Moving Average of model parameters.

    Keeps a shadow copy of model weights that is updated as:
        shadow = decay * shadow + (1 - decay) * current

    This produces a "smoother" model that typically generalizes
    better than the raw training model.

    Attributes:
        decay: EMA decay factor (higher = slower updates).
        shadow: Shadow copy of model state_dict.
        backup: Backup of original weights during apply/restore cycle.
        num_updates: Number of EMA updates performed.
    """

    def __init__(self, model: nn.Module, decay: float = 0.999) -> None:
        """Initialize EMA with a copy of the model's current weights.

        Args:
            model: The model whose weights to track.
            decay: EMA decay factor. Values close to 1.0 mean slower
                  updates (more smoothing).
        """
        self.decay = decay
        self.num_updates = 0
        self.shadow: dict = {}
        self.backup: dict = {}

        # Deep copy initial weights
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

        logger.info(
            f"ModelEMA initialized: decay={decay}, "
            f"tracking {len(self.shadow)} parameters"
        )

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        """Update shadow weights with current model weights.

        Uses a dynamic decay that increases with the number of updates,
        which helps in the early stages of training:
            decay_t = min(decay, (1 + num_updates) / (10 + num_updates))

        Args:
            model: Current model with updated weights.
        """
        self.num_updates += 1
        # Dynamic decay: starts lower, ramps up to self.decay
        decay = min(
            self.decay,
            (1 + self.num_updates) / (10 + self.num_updates),
        )

        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.shadow[name].mul_(decay).add_(
                    param.data, alpha=1.0 - decay
                )

    def apply(self, model: nn.Module) -> None:
        """Replace model weights with EMA shadow weights.

        Saves the current weights as backup so they can be restored later.

        Args:
            model: Model to apply EMA weights to.
        """
        self.backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])

    def restore(self, model: nn.Module) -> None:
        """Restore original model weights from backup.

        Should be called after apply() + validation to resume training
        with the original (non-EMA) weights.

        Args:
            model: Model to restore weights to.
        """
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.backup:
                param.data.copy_(self.backup[name])
        self.backup = {}

    def state_dict(self) -> dict:
        """Return EMA state for checkpointing.

        Returns:
            Dictionary containing shadow weights and metadata.
        """
        return {
            "shadow": self.shadow,
            "num_updates": self.num_updates,
            "decay": self.decay,
        }

    def load_state_dict(self, state: dict) -> None:
        """Load EMA state from a checkpoint.

        Args:
            state: Dictionary from a previous state_dict() call.
        """
        self.shadow = state["shadow"]
        self.num_updates = state["num_updates"]
        self.decay = state["decay"]
