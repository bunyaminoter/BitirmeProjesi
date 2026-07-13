"""
Transformer Encoder temporal model.

Uses a standard Transformer Encoder with positional encoding and
a CLS token for sequence-level pooling. Well-suited for capturing
long-range temporal dependencies in sign language sequences.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn

from src.core.config import TemporalConfig
from src.core.registry import TEMPORAL_REGISTRY
from src.models.temporal.base_temporal import BaseTemporal


@TEMPORAL_REGISTRY.register("transformer")
class TransformerTemporal(BaseTemporal):
    """Transformer Encoder temporal sequence model.

    Uses sinusoidal positional encoding and a learnable CLS token
    for sequence-level pooling. The CLS token output is used as
    the video-level representation.

    Attributes:
        config: Temporal model configuration.
        input_projection: Projects input features to model dimension.
        pos_encoding: Sinusoidal positional encoding.
        cls_token: Learnable CLS token for pooling.
        encoder: Stack of Transformer encoder layers.
        output_projection: Projects CLS output to final dimension.
    """

    def __init__(self, config: TemporalConfig, input_dim: int) -> None:
        """Initialize the Transformer temporal model.

        Args:
            config: Temporal model configuration.
            input_dim: Dimension of input features per time step.
        """
        super().__init__()
        self._config = config
        self._output_dim = config.hidden_dim
        d_model = config.hidden_dim

        # Input projection
        self.input_projection = nn.Linear(input_dim, d_model)

        # Positional encoding (sinusoidal)
        self.pos_encoding = self._create_positional_encoding(
            config.max_seq_len + 1,  # +1 for CLS token
            d_model,
        )

        # Learnable CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=config.num_heads,
            dim_feedforward=d_model * 4,
            dropout=config.dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.num_layers,
        )

        # Layer norm and output projection
        self.layer_norm = nn.LayerNorm(d_model)
        self.output_projection = nn.Sequential(
            nn.Linear(d_model, config.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(config.dropout),
        )

    @staticmethod
    def _create_positional_encoding(
        max_len: int, d_model: int
    ) -> nn.Parameter:
        """Create sinusoidal positional encoding.

        Args:
            max_len: Maximum sequence length.
            d_model: Model dimension.

        Returns:
            Positional encoding parameter of shape (1, max_len, d_model).
        """
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return nn.Parameter(pe.unsqueeze(0), requires_grad=False)

    @property
    def output_dim(self) -> int:
        """Return the output feature dimension."""
        return self._output_dim

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Process sequence with Transformer encoder.

        Args:
            x: (B, T, D) input sequence.
            mask: (B, T) optional boolean mask (True = valid).

        Returns:
            (B, output_dim) video-level representation from CLS token.
        """
        B, T, _ = x.shape

        # Project input features
        x = self.input_projection(x)  # (B, T, d_model)

        # Prepend CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, d_model)
        x = torch.cat([cls_tokens, x], dim=1)  # (B, T+1, d_model)

        # Add positional encoding
        x = x + self.pos_encoding[:, : T + 1, :]

        # Create attention mask if needed
        src_key_padding_mask = None
        if mask is not None:
            # Add True for CLS token (always valid)
            cls_mask = torch.ones(B, 1, dtype=torch.bool, device=mask.device)
            src_key_padding_mask = ~torch.cat([cls_mask, mask], dim=1)

        # Transformer encoding
        encoded = self.encoder(x, src_key_padding_mask=src_key_padding_mask)
        encoded = self.layer_norm(encoded)

        # Extract CLS token output
        cls_output = encoded[:, 0, :]  # (B, d_model)

        return self.output_projection(cls_output)
