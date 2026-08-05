"""
Augmentations applied to cached feature tensors during training.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch

from src.core.config import AugmentationConfig


class CacheAugmentation:
    """Lightweight augmentations for pre-cached pose + hand crop tensors."""

    def __init__(self, config: AugmentationConfig) -> None:
        self.config = config

    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        if not self.config.enabled:
            return sample

        pose = sample["pose_landmarks"]
        left = sample["left_hand_images"]
        right = sample["right_hand_images"]

        if self.config.random_horizontal_flip and torch.rand(1).item() < 0.5:
            left, right = right.clone(), left.clone()
            pose = self._flip_pose(pose)

        if self.config.random_brightness > 0:
            left = self._jitter_brightness(left)
            right = self._jitter_brightness(right)

        if getattr(self.config, "pose_jitter_std", 0.0) > 0:
            noise = torch.randn_like(pose) * self.config.pose_jitter_std
            pose = pose + noise

        sample = dict(sample)
        sample["pose_landmarks"] = pose
        sample["left_hand_images"] = left.clamp(0.0, 1.0)
        sample["right_hand_images"] = right.clamp(0.0, 1.0)
        return sample

    def _flip_pose(self, pose: torch.Tensor) -> torch.Tensor:
        """Mirror normalized x coordinates and swap left/right semantics."""
        flipped = pose.clone()
        num_landmarks = flipped.shape[-1] // 3
        for i in range(num_landmarks):
            x_idx = i * 3
            flipped[..., x_idx] = 1.0 - flipped[..., x_idx]
        return flipped

    def _jitter_brightness(self, images: torch.Tensor) -> torch.Tensor:
        delta = (torch.rand(1).item() * 2 - 1) * self.config.random_brightness
        return (images + delta).clamp(0.0, 1.0)


def build_cache_augmentation(
    config: AugmentationConfig,
    split: str,
) -> Optional[CacheAugmentation]:
    """Build augmentation only for the training split."""
    if split != "train" or not config.enabled:
        return None
    return CacheAugmentation(config)
