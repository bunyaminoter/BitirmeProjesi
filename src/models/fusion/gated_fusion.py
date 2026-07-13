"""
Gated Multimodal Fusion.

Learns per-modality importance weights through gating mechanisms,
allowing the model to dynamically weight different branches based
on the input. More expressive than simple concatenation.
"""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn

from src.core.config import FusionConfig
from src.core.registry import FUSION_REGISTRY
from src.models.fusion.base_fusion import BaseFusion


@FUSION_REGISTRY.register("gated")
class GatedFusion(BaseFusion):
    """Gated Multimodal Fusion module.

    Each modality branch is projected to a common dimension, then
    a learned gating mechanism produces per-modality importance weights.
    The final output is a weighted sum of projected features.

    This is more powerful than concatenation because it can learn
    to suppress noisy modalities (e.g., missing hands).

    Attributes:
        config: Fusion configuration.
        projections: Per-branch linear projections to common dim.
        gates: Per-branch gating networks.
        output_projection: Final projection after gated sum.
    """

    def __init__(self, config: FusionConfig, input_dims: List[int]) -> None:
        """Initialize gated fusion.

        Args:
            config: Fusion configuration.
            input_dims: List of input dimensions from each branch.
        """
        super().__init__()
        self._config = config
        self._output_dim = config.output_dim
        hidden_dim = config.output_dim

        # Project each branch to a common hidden dimension
        self.projections = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, hidden_dim),
                nn.ReLU(inplace=True),
            )
            for dim in input_dims
        ])

        # Gate for each branch: sigmoid output in [0, 1]
        self.gates = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, hidden_dim),
                nn.Sigmoid(),
            )
            for dim in input_dims
        ])

        # Final projection with dropout
        self.output_projection = nn.Sequential(
            nn.Linear(hidden_dim, config.output_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(config.dropout),
        )

    @property
    def output_dim(self) -> int:
        """Return the fused output dimension."""
        return self._output_dim

    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        """Fuse features using learned gating weights.

        Args:
            features: List of (B, D_i) tensors, one per branch.

        Returns:
            (B, output_dim) gated fused tensor.
        """
        gated_sum = torch.zeros(
            features[0].shape[0],
            self._output_dim,
            device=features[0].device,
            dtype=features[0].dtype,
        )

        for feat, proj, gate in zip(features, self.projections, self.gates):
            projected = proj(feat)       # (B, hidden_dim)
            gate_weight = gate(feat)     # (B, hidden_dim) in [0, 1]
            gated_sum = gated_sum + projected * gate_weight

        return self.output_projection(gated_sum)
