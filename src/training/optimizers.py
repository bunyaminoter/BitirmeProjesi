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
        name: Scheduler name ('cosine', 'step', 'plateau', 'none').
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
        available = "cosine, step, plateau, none"
        raise ValueError(f"Unknown scheduler '{name}'. Available: [{available}]")
