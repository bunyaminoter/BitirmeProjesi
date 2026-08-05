"""
Cross-Attention Multimodal Fusion.

Uses multi-head attention to model interactions between modalities.
Specifically, one modality can attend to another to dynamically
select relevant features.
"""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn

from src.core.config import FusionConfig
from src.core.registry import FUSION_REGISTRY
from src.models.fusion.base_fusion import BaseFusion


@FUSION_REGISTRY.register("cross_attention")
class CrossAttentionFusion(BaseFusion):
    """Cross-Attention Multimodal Fusion module.

    Projects each branch to a common hidden dimension, then uses
    multi-head attention to let branches interact. Finally concatenates
    the attended features and projects to the output dimension.
    
    Attributes:
        config: Fusion configuration.
        projections: Per-branch linear projections to common dim.
        attention: Multi-head attention layer.
        output_projection: Final projection after attention.
    """

    def __init__(self, config: FusionConfig, input_dims: List[int]) -> None:
        """Initialize cross attention fusion.

        Args:
            config: Fusion configuration.
            input_dims: List of input dimensions from each branch.
        """
        super().__init__()
        self._config = config
        self._output_dim = config.output_dim
        
        # We will project each modality to a common embed_dim
        # Make embed_dim a multiple of num_heads
        self.embed_dim = 512
        if self.embed_dim % config.num_heads != 0:
            self.embed_dim = config.num_heads * (self.embed_dim // config.num_heads)

        # Project each branch to a common embedding dimension
        self.projections = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, self.embed_dim),
                nn.ReLU(inplace=True),
            )
            for dim in input_dims
        ])
        
        # Learnable modality tokens to help the model distinguish branches
        self.modality_embeddings = nn.Parameter(torch.randn(len(input_dims), self.embed_dim))

        # Multi-head attention (Self-attention across modalities)
        self.attention = nn.MultiheadAttention(
            embed_dim=self.embed_dim,
            num_heads=config.num_heads,
            dropout=config.dropout,
            batch_first=True
        )
        
        self.layer_norm = nn.LayerNorm(self.embed_dim)

        # Final projection: we flatten the attended features and project
        flatten_dim = len(input_dims) * self.embed_dim
        self.output_projection = nn.Sequential(
            nn.Linear(flatten_dim, config.output_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(config.dropout),
        )

    @property
    def output_dim(self) -> int:
        """Return the fused output dimension."""
        return self._output_dim

    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        """Fuse features using cross-attention.

        Args:
            features: List of (B, D_i) tensors, one per branch.

        Returns:
            (B, output_dim) fused tensor.
        """
        # Project each feature to embed_dim and stack
        projected = []
        for i, (feat, proj) in enumerate(zip(features, self.projections)):
            p_feat = proj(feat) # (B, embed_dim)
            # Add modality embedding
            p_feat = p_feat + self.modality_embeddings[i].unsqueeze(0)
            projected.append(p_feat)
            
        # Stack to form a sequence of length = num_modalities
        # x shape: (B, num_modalities, embed_dim)
        x = torch.stack(projected, dim=1)
        
        # Apply self-attention across the modalities
        attn_out, _ = self.attention(x, x, x)
        
        # Add & Norm
        out = self.layer_norm(x + attn_out)
        
        # Flatten and project
        # out shape: (B, num_modalities * embed_dim)
        B = out.shape[0]
        out_flat = out.view(B, -1)
        
        return self.output_projection(out_flat)
