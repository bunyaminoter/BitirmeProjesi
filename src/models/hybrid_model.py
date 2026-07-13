"""
Main Hybrid ASL Model — the top-level model that assembles all branches.

Architecture:
    1. Hand CNN Encoder(s) — extract features from left/right hand RGB crops
    2. Landmark Encoder — extract features from pose/face landmarks
    3. Feature Fusion — combine all branch outputs into a single vector
    4. Temporal Model — model the sequence of fused features over time
    5. Classification Head — produce class logits

All components are injected via configuration, making the model
fully modular and every component independently replaceable.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import torch
import torch.nn as nn

from src.core.config import ModelConfig
from src.core.registry import ENCODER_REGISTRY, FUSION_REGISTRY, TEMPORAL_REGISTRY

logger = logging.getLogger(__name__)

# Force-import encoder, fusion, and temporal modules to trigger registration
import src.models.encoders.hand_cnn_encoder  # noqa: F401
import src.models.encoders.landmark_encoder  # noqa: F401
import src.models.fusion.concat_fusion  # noqa: F401
import src.models.fusion.gated_fusion  # noqa: F401
import src.models.temporal.bilstm_temporal  # noqa: F401
import src.models.temporal.transformer_temporal  # noqa: F401


class HybridASLModel(nn.Module):
    """Hybrid ASL recognition model combining RGB hand features
    with body landmark features.

    This is the top-level model that orchestrates:
        - Branch 1: Body landmark encoding (pose + optional face)
        - Branch 2: Left hand RGB encoding
        - Branch 3: Right hand RGB encoding
        - Feature fusion across all branches
        - Temporal modeling over the fused sequence
        - Final word-level classification

    Attributes:
        config: Model architecture configuration.
        landmark_encoder: Encoder for body landmarks.
        hand_encoder_left: CNN encoder for left hand crops.
        hand_encoder_right: CNN encoder for right hand crops (may share weights).
        fusion: Feature fusion module.
        temporal: Temporal sequence model.
        classifier: Final classification head.
    """

    def __init__(self, config: ModelConfig) -> None:
        """Initialize the hybrid model from configuration.

        Args:
            config: ModelConfig specifying all component configurations.
        """
        super().__init__()
        self.config = config

        # --- Branch 1: Landmark Encoder ---
        self.landmark_encoder: nn.Module = ENCODER_REGISTRY.build(
            "landmark_mlp",
            config=config.landmark_encoder,
        )
        landmark_out_dim = config.landmark_encoder.output_dim
        logger.info(
            f"Landmark encoder: input_dim={config.landmark_encoder.input_dim}, "
            f"output_dim={landmark_out_dim}"
        )

        # --- Branch 2 & 3: Hand CNN Encoders ---
        self.hand_encoder_left: nn.Module = ENCODER_REGISTRY.build(
            "hand_cnn",
            config=config.hand_encoder,
        )
        hand_out_dim = config.hand_encoder.output_dim

        # If shared_weights is True, right encoder points to left
        if config.hand_encoder.shared_weights:
            self.hand_encoder_right = self.hand_encoder_left
            logger.info(
                f"Hand CNN encoder (shared): backbone={config.hand_encoder.backbone}, "
                f"output_dim={hand_out_dim}"
            )
        else:
            self.hand_encoder_right: nn.Module = ENCODER_REGISTRY.build(
                "hand_cnn",
                config=config.hand_encoder,
            )
            logger.info(
                f"Hand CNN encoder (independent L/R): backbone={config.hand_encoder.backbone}, "
                f"output_dim={hand_out_dim}"
            )

        # --- Feature Fusion ---
        # Input dims: [landmark_out, left_hand_out, right_hand_out]
        fusion_input_dims: List[int] = [landmark_out_dim, hand_out_dim, hand_out_dim]
        self.fusion: nn.Module = FUSION_REGISTRY.build(
            config.fusion.method,
            config=config.fusion,
            input_dims=fusion_input_dims,
        )
        fusion_out_dim = config.fusion.output_dim
        logger.info(
            f"Fusion: method={config.fusion.method}, "
            f"input_dims={fusion_input_dims}, output_dim={fusion_out_dim}"
        )

        # --- Temporal Model ---
        self.temporal: nn.Module = TEMPORAL_REGISTRY.build(
            config.temporal.method,
            config=config.temporal,
            input_dim=fusion_out_dim,
        )
        temporal_out_dim = config.temporal.hidden_dim
        logger.info(
            f"Temporal: method={config.temporal.method}, "
            f"input_dim={fusion_out_dim}, output_dim={temporal_out_dim}"
        )

        # --- Classification Head ---
        self.classifier = nn.Sequential(
            nn.Dropout(config.classifier_dropout),
            nn.Linear(temporal_out_dim, config.num_classes),
        )
        logger.info(
            f"Classifier: input_dim={temporal_out_dim}, "
            f"num_classes={config.num_classes}"
        )

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Forward pass through the hybrid model.

        Args:
            batch: Dictionary containing:
                - 'pose_landmarks': (B, T, landmark_dim) float tensor
                - 'left_hand_images': (B, T, C, H, W) float tensor
                - 'right_hand_images': (B, T, C, H, W) float tensor
                - 'mask': (B, T) bool tensor (optional)

        Returns:
            (B, num_classes) logit tensor.
        """
        pose_landmarks = batch["pose_landmarks"]
        left_hand_images = batch["left_hand_images"]
        right_hand_images = batch["right_hand_images"]
        mask = batch.get("mask", None)

        B, T = pose_landmarks.shape[:2]

        # --- Encode landmarks per frame ---
        # pose_landmarks: (B, T, landmark_dim) → reshape to (B*T, landmark_dim)
        landmark_flat = pose_landmarks.reshape(B * T, -1)
        landmark_features = self.landmark_encoder(landmark_flat)  # (B*T, lm_out)
        landmark_features = landmark_features.reshape(B, T, -1)  # (B, T, lm_out)

        # --- Encode hand crops per frame ---
        # left_hand_images: (B, T, C, H, W) → (B*T, C, H, W)
        left_flat = left_hand_images.reshape(B * T, *left_hand_images.shape[2:])
        left_features = self.hand_encoder_left(left_flat)  # (B*T, hand_out)
        left_features = left_features.reshape(B, T, -1)  # (B, T, hand_out)

        # right_hand_images: (B, T, C, H, W) → (B*T, C, H, W)
        right_flat = right_hand_images.reshape(B * T, *right_hand_images.shape[2:])
        right_features = self.hand_encoder_right(right_flat)  # (B*T, hand_out)
        right_features = right_features.reshape(B, T, -1)  # (B, T, hand_out)

        # --- Fuse features per frame ---
        # Each frame: [landmark_feat, left_feat, right_feat] → fused
        # Process all frames at once by reshaping
        lm_fuse = landmark_features.reshape(B * T, -1)
        left_fuse = left_features.reshape(B * T, -1)
        right_fuse = right_features.reshape(B * T, -1)

        fused = self.fusion([lm_fuse, left_fuse, right_fuse])  # (B*T, fused_dim)
        fused = fused.reshape(B, T, -1)  # (B, T, fused_dim)

        # --- Temporal modeling ---
        # fused: (B, T, fused_dim) → sequence_repr: (B, temporal_out_dim)
        sequence_repr = self.temporal(fused, mask=mask)

        # --- Classification ---
        # sequence_repr: (B, temporal_out_dim) → logits: (B, num_classes)
        logits = self.classifier(sequence_repr)

        return logits

    def get_num_parameters(self) -> Dict[str, int]:
        """Count parameters per component for analysis.

        Returns:
            Dictionary mapping component names to parameter counts.
        """
        counts = {}
        for name, module in [
            ("landmark_encoder", self.landmark_encoder),
            ("hand_encoder_left", self.hand_encoder_left),
            ("hand_encoder_right", self.hand_encoder_right),
            ("fusion", self.fusion),
            ("temporal", self.temporal),
            ("classifier", self.classifier),
        ]:
            if module is not None:
                counts[name] = sum(p.numel() for p in module.parameters())
        counts["total"] = sum(counts.values())
        return counts
