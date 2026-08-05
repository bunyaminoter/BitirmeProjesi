"""
TCN-Transformer Hybrid Temporal Model.

Combines the local receptive field of Temporal Convolutional Networks (TCN)
with the global context modeling of Transformers. The TCN extracts local motion
trajectories, which are then passed to the Transformer for global sequence modeling.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn

from src.core.config import TemporalConfig
from src.core.registry import TEMPORAL_REGISTRY
from src.models.temporal.base_temporal import BaseTemporal


@TEMPORAL_REGISTRY.register("tcn_transformer")
class TCNTransformerTemporal(BaseTemporal):
    """TCN + Transformer temporal sequence model.

    Applies a 1D Convolution (TCN) to capture local temporal context (e.g., motion direction),
    followed by a standard Transformer Encoder for global context.
    
    Attributes:
        config: Temporal configuration.
        tcn: 1D Convolutional layer(s).
        pos_encoding: Positional encoding for the transformer.
        transformer: Transformer encoder.
        cls_token: Learnable CLS token for pooling.
        output_projection: Projects output to final dimension.
    """

    def __init__(self, config: TemporalConfig, input_dim: int) -> None:
        super().__init__()
        self._config = config
        self._output_dim = config.hidden_dim
        d_model = config.hidden_dim
        
        # We need to project input_dim to hidden_dim if they don't match,
        # but we'll let the TCN do this projection while doing the convolution.
        
        # TCN Layer (1D Conv over time)
        # Input shape to Conv1d: (B, C, T)
        # Output shape: (B, hidden_dim, T)
        self.tcn = nn.Sequential(
            nn.Conv1d(
                in_channels=input_dim, 
                out_channels=d_model, 
                kernel_size=3, 
                padding=1, # Keep sequence length same
                bias=False
            ),
            nn.BatchNorm1d(d_model),
            nn.ReLU(inplace=True),
            nn.Dropout(config.dropout)
        )

        # Positional encoding (sinusoidal)
        self.pos_encoding = self._create_positional_encoding(
            config.max_seq_len + 1,  # +1 for CLS token
            d_model,
        )

        # Learnable CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))

        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=config.num_heads,
            dim_feedforward=d_model * 4,
            dropout=config.dropout,
            batch_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layers,
            num_layers=config.num_layers,
        )

        self.layer_norm = nn.LayerNorm(d_model)
        self.output_projection = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(inplace=True),
            nn.Dropout(config.dropout),
        )

    @staticmethod
    def _create_positional_encoding(
        max_len: int, d_model: int
    ) -> nn.Parameter:
        """Create sinusoidal positional encoding."""
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
        return self._output_dim

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: (B, T, input_dim)
            mask: (B, T) boolean mask where True means valid, False means padded
        
        Returns:
            (B, output_dim)
        """
        B, T, _ = x.shape
        
        # --- 1. TCN Pass ---
        # x: (B, T, C) -> (B, C, T) for Conv1d
        x = x.permute(0, 2, 1)
        x = self.tcn(x)
        # Back to (B, T, C)
        x = x.permute(0, 2, 1)
        
        # --- 2. Transformer Pass ---
        # Prepend CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # (B, T+1, d_model)
        
        # Add positional encoding
        x = x + self.pos_encoding[:, : T + 1, :]
        
        # Transformer expects mask where True = ignore
        src_key_padding_mask = None
        if mask is not None:
            # Add True for CLS token (always valid)
            cls_mask = torch.ones(B, 1, dtype=torch.bool, device=mask.device)
            src_key_padding_mask = ~torch.cat([cls_mask, mask], dim=1)
            
        out = self.transformer(
            x,
            src_key_padding_mask=src_key_padding_mask
        )
        out = self.layer_norm(out)
        
        # --- 3. Pooling ---
        cls_output = out[:, 0, :]  # (B, d_model)
        return self.output_projection(cls_output)
