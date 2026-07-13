"""
Bidirectional LSTM temporal model.

Processes the sequence of fused features with a multi-layer BiLSTM.
The final hidden states from both directions are concatenated and
projected to produce a video-level representation.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from src.core.config import TemporalConfig
from src.core.registry import TEMPORAL_REGISTRY
from src.models.temporal.base_temporal import BaseTemporal


@TEMPORAL_REGISTRY.register("bilstm")
class BiLSTMTemporal(BaseTemporal):
    """Bidirectional LSTM temporal sequence model.

    Uses a multi-layer BiLSTM to capture temporal dependencies in
    both forward and backward directions. The final hidden states
    are concatenated and projected to output_dim.

    Attributes:
        config: Temporal model configuration.
        lstm: Multi-layer BiLSTM module.
        projection: Linear projection from LSTM output to output_dim.
    """

    def __init__(self, config: TemporalConfig, input_dim: int) -> None:
        """Initialize the BiLSTM temporal model.

        Args:
            config: Temporal model configuration.
            input_dim: Dimension of input features per time step.
        """
        super().__init__()
        self._config = config
        self._output_dim = config.hidden_dim

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=config.hidden_dim,
            num_layers=config.num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=config.dropout if config.num_layers > 1 else 0.0,
        )

        # BiLSTM outputs 2 * hidden_dim, project to hidden_dim
        self.projection = nn.Sequential(
            nn.Linear(config.hidden_dim * 2, config.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(config.dropout),
        )

    @property
    def output_dim(self) -> int:
        """Return the output feature dimension."""
        return self._output_dim

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Process sequence with BiLSTM.

        Args:
            x: (B, T, D) input sequence.
            mask: (B, T) optional boolean mask (unused for now).

        Returns:
            (B, output_dim) video-level representation.
        """
        # LSTM forward pass
        output, (h_n, _) = self.lstm(x)

        # h_n shape: (num_layers * 2, B, hidden_dim) for bidirectional
        # Take the last layer's forward and backward hidden states
        forward_h = h_n[-2]   # (B, hidden_dim)
        backward_h = h_n[-1]  # (B, hidden_dim)
        combined = torch.cat([forward_h, backward_h], dim=-1)  # (B, 2*hidden_dim)

        return self.projection(combined)
