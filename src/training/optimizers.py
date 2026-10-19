"""
Optimizer and learning rate scheduler factory.

Creates optimizers and schedulers from configuration, registered
through the global registry for extensibility.
"""

from __future__ import annotations

from typing import Any, Dict, Iterator

import torch
import torch.nn as nn
import torch.optim as optim


def build_optimizer(
    name: str,
    parameters: Iterator[nn.Parameter],
    lr: float,
    weight_decay: float = 1e-4,
    **kwargs: Any,
) -> optim.Optimizer:
    """Build an optimizer from a name string.

    Args:
        name: Optimizer name ('adam', 'adamw', 'sgd').
        parameters: Model parameters to optimize.
        lr: Learning rate.
        weight_decay: Weight decay (L2 regularization).
        **kwargs: Additional optimizer-specific arguments.

    Returns:
        Configured optimizer instance.

    Raises:
        ValueError: If the optimizer name is not recognized.
    """
    optimizers: Dict[str, type] = {
        "adam": optim.Adam,
        "adamw": optim.AdamW,
        "sgd": optim.SGD,
        "rmsprop": optim.RMSprop,
    }

    if name.lower() not in optimizers:
        available = ", ".join(sorted(optimizers.keys()))
        raise ValueError(f"Unknown optimizer '{name}'. Available: [{available}]")

    optimizer_cls = optimizers[name.lower()]
    return optimizer_cls(parameters, lr=lr, weight_decay=weight_decay, **kwargs)


def build_scheduler(
    name: str,
    optimizer: optim.Optimizer,
    epochs: int = 100,
    **kwargs: Any,
) -> Any:
    """Build a learning rate scheduler from a name string.

    Args:
        name: Scheduler name ('cosine', 'step', 'plateau',
              'cosine_warmup_restart', 'none').
        optimizer: Optimizer to schedule.
        epochs: Total number of training epochs.
        **kwargs: Additional scheduler-specific arguments.

    Returns:
        Configured scheduler instance, or None if 'none'.

    Raises:
        ValueError: If the scheduler name is not recognized.
    """
    if name.lower() == "none":
        return None

    if name.lower() == "cosine":
        return optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=kwargs.get("T_max", epochs),
            eta_min=kwargs.get("eta_min", 1e-7),
        )
    elif name.lower() == "cosine_warmup_restart":
        return optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=kwargs.get("T_0", 20),
            T_mult=kwargs.get("T_mult", 2),
            eta_min=kwargs.get("eta_min", 1e-7),
        )
    elif name.lower() == "step":
        return optim.lr_scheduler.StepLR(
            optimizer,
            step_size=kwargs.get("step_size", 30),
            gamma=kwargs.get("gamma", 0.1),
        )
    elif name.lower() == "plateau":
        return optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=kwargs.get("mode", "max"),
            patience=kwargs.get("patience", 10),
            factor=kwargs.get("factor", 0.1),
        )
    else:
        available = "cosine, cosine_warmup_restart, step, plateau, none"
        raise ValueError(f"Unknown scheduler '{name}'. Available: [{available}]")


class LinearWarmupScheduler:
    """Linear warmup wrapper for any LR scheduler.

    During the first `warmup_epochs` epochs, the learning rate is linearly
    ramped from a small fraction to the base LR. After warmup, the wrapped
    scheduler takes over.

    Attributes:
        optimizer: The optimizer whose LR is being scheduled.
        warmup_epochs: Number of warmup epochs.
        base_lrs: Target learning rates after warmup completes.
        wrapped_scheduler: The scheduler to delegate to after warmup.
    """

    def __init__(
        self,
        optimizer: optim.Optimizer,
        warmup_epochs: int,
        wrapped_scheduler: Any = None,
    ) -> None:
        """Initialize the warmup scheduler.

        Args:
            optimizer: Optimizer to schedule.
            warmup_epochs: Number of epochs for linear warmup.
            wrapped_scheduler: Optional scheduler to use after warmup.
        """
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.wrapped_scheduler = wrapped_scheduler
        self.base_lrs = [pg["lr"] for pg in optimizer.param_groups]
        self.current_epoch = 0

        # Start with very small LR
        if warmup_epochs > 0:
            for pg in optimizer.param_groups:
                pg["lr"] = pg["lr"] * 0.01

    def step(self, metric: Any = None) -> None:
        """Perform one scheduler step.

        Args:
            metric: Optional metric value (for ReduceLROnPlateau compatibility).
        """
        self.current_epoch += 1

        if self.current_epoch <= self.warmup_epochs:
            # Linear warmup: scale from 1% to 100% of base LR
            scale = self.current_epoch / self.warmup_epochs
            for pg, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
                pg["lr"] = base_lr * scale
        elif self.wrapped_scheduler is not None:
            if isinstance(
                self.wrapped_scheduler,
                optim.lr_scheduler.ReduceLROnPlateau,
            ):
                self.wrapped_scheduler.step(metric)
            else:
                self.wrapped_scheduler.step()

    def state_dict(self) -> Dict[str, Any]:
        """Return scheduler state for checkpointing."""
        state = {
            "current_epoch": self.current_epoch,
            "warmup_epochs": self.warmup_epochs,
            "base_lrs": self.base_lrs,
        }
        if self.wrapped_scheduler is not None:
            state["wrapped_scheduler"] = self.wrapped_scheduler.state_dict()
        return state

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        """Load scheduler state from checkpoint."""
        self.current_epoch = state["current_epoch"]
        self.warmup_epochs = state["warmup_epochs"]
        self.base_lrs = state["base_lrs"]
        if (
            self.wrapped_scheduler is not None
            and "wrapped_scheduler" in state
        ):
            self.wrapped_scheduler.load_state_dict(state["wrapped_scheduler"])

