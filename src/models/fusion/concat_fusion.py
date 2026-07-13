"""
Concatenation-based feature fusion.

The simplest fusion strategy: concatenate all branch outputs and
optionally project to a desired output dimension.
"""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn

from src.core.config import FusionConfig
from src.core.registry import FUSION_REGISTRY
from src.models.fusion.base_fusion import BaseFusion


@FUSION_REGISTRY.register("concat")
class ConcatFusion(BaseFusion):
    """Concatenation-based feature fusion.

    Concatenates all input feature vectors along the feature dimension,
    then optionally projects through a linear layer to reduce dimensionality.

    Input: List of (B, D_i) tensors from each branch.
    Output: (B, output_dim) fused feature tensor.

    Attributes:
        config: Fusion configuration.
        projection: Optional linear projection after concatenation.
    """

    def __init__(self, config: FusionConfig, input_dims: List[int]) -> None:
        """Initialize concatenation fusion.

        Args:
            config: Fusion configuration.
            input_dims: List of input dimensions from each branch.
        """
        super().__init__()
        self._config = config
        self._input_dims = input_dims
        self._total_input_dim = sum(input_dims)
        self._output_dim = config.output_dim

        # Projection from concatenated dim to output dim
        self.projection = nn.Sequential(
            nn.Linear(self._total_input_dim, config.output_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(config.dropout),
        )

    @property
    def output_dim(self) -> int:
        """Return the fused output dimension."""
        return self._output_dim

    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        """Fuse features by concatenation and projection.

        Args:
            features: List of (B, D_i) tensors.

        Returns:
            (B, output_dim) fused tensor.
        """
        concatenated = torch.cat(features, dim=-1)
        return self.projection(concatenated)
