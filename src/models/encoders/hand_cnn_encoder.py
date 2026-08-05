"""
CNN encoder for RGB hand crop images.

Uses a configurable torchvision backbone (ResNet18, EfficientNet-B0,
MobileNetV3, ConvNeXt-Tiny) as the feature extractor. The classification
head is replaced with a linear projection to the desired output dimension.

Supports both shared and independent weights for left/right hands.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional

import torch
import torch.nn as nn
import torchvision.models as tv_models

from src.core.config import HandEncoderConfig
from src.core.registry import ENCODER_REGISTRY
from src.models.encoders.base_encoder import BaseEncoder
from src.models.attention.cbam import CBAMBlock

logger = logging.getLogger(__name__)

# Mapping from config backbone names to torchvision constructor info.
# Each entry: (torchvision_model_func_name, pretrained_weights_name, output_feature_dim)
_BACKBONE_REGISTRY: Dict[str, tuple] = {
    "resnet18": ("resnet18", "IMAGENET1K_V1", 512),
    "resnet34": ("resnet34", "IMAGENET1K_V1", 512),
    "resnet50": ("resnet50", "IMAGENET1K_V1", 2048),
    "mobilenet_v3_small": ("mobilenet_v3_small", "IMAGENET1K_V1", 576),
    "mobilenet_v3_large": ("mobilenet_v3_large", "IMAGENET1K_V1", 960),
    "efficientnet_b0": ("efficientnet_b0", "IMAGENET1K_V1", 1280),
    "efficientnet_b2": ("efficientnet_b2", "IMAGENET1K_V1", 1408),
    "convnext_tiny": ("convnext_tiny", "IMAGENET1K_V1", 768),
    "convnext_small": ("convnext_small", "IMAGENET1K_V1", 768),
}


def _build_backbone(
    name: str, pretrained: bool, use_cbam: bool = False
) -> tuple[nn.Module, int]:
    """Build a torchvision backbone and return (feature_extractor, out_dim).

    The classification head is removed; output is a spatial feature map
    followed by AdaptiveAvgPool2d → Flatten.

    Args:
        name: Backbone name (must be in _BACKBONE_REGISTRY).
        pretrained: Whether to load ImageNet-pretrained weights.

    Returns:
        Tuple of (backbone_module, output_feature_dim).

    Raises:
        ValueError: If backbone name is not supported.
    """
    if name not in _BACKBONE_REGISTRY:
        available = ", ".join(sorted(_BACKBONE_REGISTRY.keys()))
        raise ValueError(
            f"Unknown backbone '{name}'. Available: [{available}]"
        )

    model_func_name, weights_name, out_dim = _BACKBONE_REGISTRY[name]
    model_fn = getattr(tv_models, model_func_name)

    # Load with or without pretrained weights
    if pretrained:
        weights = getattr(tv_models, f"{weights_name.split('_V')[0]}_Weights", None)
        if weights is None:
            # Fallback: use string-based weights
            model = model_fn(weights=weights_name)
        else:
            model = model_fn(weights=weights.DEFAULT)
        logger.info(f"Loaded pretrained backbone: {name}")
    else:
        model = model_fn(weights=None)
        logger.info(f"Loaded backbone (random init): {name}")

    # Remove the classification head and build a feature extractor
    if name.startswith("resnet"):
        # ResNet: remove fc layer, keep everything up to avgpool
        features = nn.Sequential(
            model.conv1,
            model.bn1,
            model.relu,
            model.maxpool,
            model.layer1,
            model.layer2,
            model.layer3,
            model.layer4,
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
    elif name.startswith("mobilenet"):
        # MobileNetV3: use .features only
        features = nn.Sequential(
            model.features,
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
    elif name.startswith("efficientnet"):
        # EfficientNet: use .features only
        features = nn.Sequential(
            model.features,
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
    elif name.startswith("convnext"):
        # ConvNeXt: use .features only
        features = nn.Sequential(
            model.features,
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
    else:
        raise ValueError(f"No feature extraction logic for backbone '{name}'")
        
    if use_cbam:
        # Insert CBAM right before the AdaptiveAvgPool2d
        # Find index of AdaptiveAvgPool2d
        pool_idx = -1
        for i, layer in enumerate(features):
            if isinstance(layer, nn.AdaptiveAvgPool2d):
                pool_idx = i
                break
                
        if pool_idx != -1:
            # We recreate the Sequential block with CBAM inserted
            new_layers = list(features[:pool_idx])
            new_layers.append(CBAMBlock(in_planes=out_dim))
            new_layers.extend(list(features[pool_idx:]))
            features = nn.Sequential(*new_layers)
            logger.info(f"Injected CBAM block before pooling layer in {name}")

    return features, out_dim


@ENCODER_REGISTRY.register("hand_cnn")
class HandCNNEncoder(BaseEncoder):
    """CNN-based encoder for RGB hand crop images.

    Extracts spatial features from cropped hand images using a
    pretrained torchvision backbone. The final classification layer
    is replaced with a projection to the configured output_dim.

    Supports multiple backbone architectures:
        - resnet18, resnet34, resnet50
        - efficientnet_b0, efficientnet_b2
        - mobilenet_v3_small, mobilenet_v3_large
        - convnext_tiny, convnext_small

    Attributes:
        config: Hand encoder configuration.
        backbone: The CNN backbone (without classification head).
        projector: Linear projection to output_dim.
    """

    def __init__(self, config: HandEncoderConfig) -> None:
        """Initialize the Hand CNN encoder.

        Args:
            config: Hand encoder configuration specifying backbone,
                   pretrained weights, output_dim, etc.
        """
        super().__init__()
        self._config = config
        self._output_dim = config.output_dim

        # Build real torchvision backbone
        self.backbone, backbone_out_dim = _build_backbone(
            config.backbone, config.pretrained, getattr(config, "use_cbam", False)
        )

        # Projection head: maps backbone features to desired output_dim
        self.projector = nn.Sequential(
            nn.Dropout(config.dropout),
            nn.Linear(backbone_out_dim, config.output_dim),
            nn.ReLU(inplace=True),
        )

    @property
    def output_dim(self) -> int:
        """Return the output feature dimension."""
        return self._output_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features from hand crop images.

        Args:
            x: (B, C, H, W) float tensor of hand crop images.

        Returns:
            (B, output_dim) feature tensor.
        """
        features = self.backbone(x)
        return self.projector(features)
