"""
Abstract base class for feature fusion modules.

Defines the interface for combining feature vectors from multiple
modality branches (landmark encoder, left hand CNN, right hand CNN)
into a single fused representation.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

import torch
import torch.nn as nn


class BaseFusion(nn.Module, ABC):
    """Abstract base class for multi-modal feature fusion.

    Takes a list of feature vectors from different branches and
    produces a single fused feature vector. All fusion strategies
    must declare their output dimensionality.

    Subclasses implement different fusion strategies:
        - Concatenation
        - Gated fusion
        - Attention-based fusion
        - Cross-modal attention
    """

    @property
    @abstractmethod
    def output_dim(self) -> int:
        """Return the dimensionality of the fused output vector.

        Returns:
            Integer output dimension.
        """
        ...

    @abstractmethod
    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        """Fuse feature vectors from multiple branches.

        Args:
            features: List of tensors, each of shape (B, D_i),
                     where D_i is the dimension of branch i.

        Returns:
            Fused feature tensor of shape (B, output_dim).
        """
        ...
