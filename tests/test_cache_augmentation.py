"""Temporal augmentation must keep the model attention mask aligned with features."""

import torch

from src.core.config import AugmentationConfig
from src.data.transforms.cache_augmentation import CacheAugmentation


def test_speed_resampling_updates_mask_and_preserves_missing_joints():
    config = AugmentationConfig(
        random_horizontal_flip=False,
        random_rotation=0.0,
        hand_random_translation=0,
        random_brightness=0.0,
        random_speed_range=[2.0, 2.0],
        pose_jitter_std=0.01,
    )
    pose = torch.zeros(16, 225)
    pose[:8, 0] = 0.25
    sample = {
        "pose_landmarks": pose,
        "left_hand_images": torch.zeros(16, 3, 2, 2),
        "right_hand_images": torch.zeros(16, 3, 2, 2),
        "mask": torch.arange(16) < 8,
    }

    result = CacheAugmentation(config)(sample)

    assert result["mask"].sum().item() == 4
    assert not result["mask"][4:].any()
    assert torch.count_nonzero(result["pose_landmarks"][:, 3:]) == 0
    assert torch.count_nonzero(result["pose_landmarks"][4:]) == 0
