"""
Abstract base class for feature encoders.

All encoders (hand CNN, landmark MLP, etc.) must implement this interface.
This ensures uniform feature vector output regardless of the input modality.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class BaseEncoder(nn.Module, ABC):
    """Abstract base class for all feature encoders.

    An encoder takes a raw input (image, landmarks, etc.) and produces
    a fixed-dimensional feature vector. All encoders must declare their
    output dimensionality via the output_dim property.

    This enables the fusion module to dynamically determine input
    dimensions without hardcoded assumptions.
    """

    @property
    @abstractmethod
    def output_dim(self) -> int:
        """Return the dimensionality of the output feature vector.

        Returns:
            Integer output dimension.
        """
        ...

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to a feature vector.

        Args:
            x: Input tensor (shape depends on encoder type).

        Returns:
            Feature tensor of shape (..., output_dim).
        """
        ...
