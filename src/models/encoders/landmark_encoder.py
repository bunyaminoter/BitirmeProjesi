"""
Landmark feature encoder for pose and face landmarks.

Encodes flattened landmark coordinate vectors into a fixed-dimensional
feature representation using an MLP with batch normalization and dropout.
"""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn

from src.core.config import LandmarkEncoderConfig
from src.core.registry import ENCODER_REGISTRY
from src.models.encoders.base_encoder import BaseEncoder


@ENCODER_REGISTRY.register("landmark_mlp")
class LandmarkEncoder(BaseEncoder):
    """MLP encoder for landmark coordinate vectors.

    Takes flattened landmark coordinates (pose + optional face) and
    produces a fixed-dimensional feature vector through a multi-layer
    perceptron with batch normalization and dropout.

    Input: (B, num_landmarks * 3) where each landmark has (x, y, z).
    Output: (B, output_dim) feature vector.

    Attributes:
        config: Landmark encoder configuration.
        mlp: Multi-layer perceptron.
    """

    def __init__(self, config: LandmarkEncoderConfig) -> None:
        """Initialize the landmark encoder.

        Args:
            config: Landmark encoder configuration.
        """
        super().__init__()
        self._config = config
        self._output_dim = config.output_dim

        # Build MLP layers
        layers: List[nn.Module] = []
        in_dim = config.input_dim

        for hidden_dim in config.hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
            if config.use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(config.dropout))
            in_dim = hidden_dim

        # Final projection
        layers.append(nn.Linear(in_dim, config.output_dim))
        layers.append(nn.ReLU(inplace=True))

        self.mlp = nn.Sequential(*layers)

    @property
    def output_dim(self) -> int:
        """Return the output feature dimension."""
        return self._output_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode landmark coordinates to a feature vector.

        Args:
            x: (B, input_dim) float tensor of flattened landmarks.

        Returns:
            (B, output_dim) feature tensor.
        """
        return self.mlp(x)
