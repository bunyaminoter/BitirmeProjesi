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
        mask = sample["mask"]

        # Temporal Resampling (Speed Augmentation)
        speed_range = getattr(self.config, "random_speed_range", [1.0, 1.0])
        if speed_range[0] != 1.0 or speed_range[1] != 1.0:
            speed = speed_range[0] + torch.rand(1).item() * (speed_range[1] - speed_range[0])
            if speed != 1.0:
                pose, left, right, mask = self._resample_temporal(pose, left, right, mask, speed)

        # Spatial Augmentations
        rotation = getattr(self.config, "random_rotation", 0.0)
        translation = getattr(self.config, "hand_random_translation", 0)
        if rotation > 0 or translation > 0:
            left, right = self._spatial_transform(left, right, rotation, translation)

        if self.config.random_horizontal_flip and torch.rand(1).item() < 0.5:
            left, right = right.clone(), left.clone()
            pose = self._flip_pose(pose)

        if self.config.random_brightness > 0:
            left = self._jitter_brightness(left)
            right = self._jitter_brightness(right)

        if getattr(self.config, "pose_jitter_std", 0.0) > 0:
            noise = torch.randn_like(pose) * self.config.pose_jitter_std
            valid_joints = pose.reshape(pose.shape[0], -1, 3).abs().sum(dim=-1, keepdim=True) > 1e-4
            pose = pose + (noise.reshape(pose.shape[0], -1, 3) * valid_joints).reshape_as(pose)

        # Random Erasing (Cutout)
        erasing_prob = getattr(self.config, "random_erasing_prob", 0.0)
        if erasing_prob > 0:
            left = self._random_erase(left, erasing_prob)
            right = self._random_erase(right, erasing_prob)

        sample = dict(sample)
        sample["pose_landmarks"] = pose
        sample["mask"] = mask
        sample["left_hand_images"] = left.clamp(0.0, 1.0)
        sample["right_hand_images"] = right.clamp(0.0, 1.0)
        return sample

    def _spatial_transform(self, left: torch.Tensor, right: torch.Tensor, rot_range: float, trans_range: float) -> tuple[torch.Tensor, torch.Tensor]:
        import torchvision.transforms.functional as TF
        angle = (torch.rand(1).item() * 2 - 1) * rot_range
        tx = int((torch.rand(1).item() * 2 - 1) * trans_range)
        ty = int((torch.rand(1).item() * 2 - 1) * trans_range)
        left = TF.affine(left, angle=angle, translate=[tx, ty], scale=1.0, shear=0.0)
        right = TF.affine(right, angle=angle, translate=[tx, ty], scale=1.0, shear=0.0)
        return left, right

    def _resample_temporal(self, pose: torch.Tensor, left: torch.Tensor, right: torch.Tensor, mask: torch.Tensor, speed: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        import torch.nn.functional as F
        T = pose.shape[0]
        valid_T = max(1, int(mask.sum().item()))
        new_T = max(1, int(round(valid_T / speed)))
        pose, left, right = pose[:valid_T], left[:valid_T], right[:valid_T]
        
        pose_D = pose.shape[1]
        pose_resampled = F.interpolate(pose.t().unsqueeze(0), size=new_T, mode='linear', align_corners=False).squeeze(0).t()
        
        C, H, W = left.shape[1:]
        def resample_images(imgs):
            imgs_res = F.interpolate(imgs.reshape(valid_T, C*H*W).t().unsqueeze(0), size=new_T, mode='linear', align_corners=False)
            return imgs_res.squeeze(0).t().reshape(new_T, C, H, W)
            
        left_resampled = resample_images(left)
        right_resampled = resample_images(right)
        
        def pad_or_trunc(tensor, target_len):
            if tensor.shape[0] >= target_len:
                return tensor[:target_len]
            pad_shape = list(tensor.shape)
            pad_shape[0] = target_len - tensor.shape[0]
            padding = torch.zeros(pad_shape, dtype=tensor.dtype, device=tensor.device)
            return torch.cat([tensor, padding], dim=0)
            
        new_mask = torch.arange(T, device=mask.device) < min(new_T, T)
        return pad_or_trunc(pose_resampled, T), pad_or_trunc(left_resampled, T), pad_or_trunc(right_resampled, T), new_mask

    def _random_erase(self, images: torch.Tensor, prob: float) -> torch.Tensor:
        from torchvision.transforms import RandomErasing
        eraser = RandomErasing(p=1.0, scale=(0.02, 0.1), ratio=(0.3, 3.3), value=0)
        # Apply independently per frame for better robustness
        erased = images.clone()
        for i in range(images.shape[0]):
            if torch.rand(1).item() < prob:
                erased[i] = eraser(images[i])
        return erased

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
