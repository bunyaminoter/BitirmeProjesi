"""
WLASL DataModule — manages cached feature loading and DataLoaders.

Provides cache-aware dataset that loads preprocessed features from
.npz files instead of running MediaPipe during training.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from src.core.config import AugmentationConfig, DatasetConfig, TrainingConfig
from src.data.base_datamodule import BaseDataModule
from src.data.datasets.wlasl_dataset import WLASLDataset
from src.data.transforms.cache_augmentation import build_cache_augmentation

logger = logging.getLogger(__name__)


class CachedWLASLDataset(Dataset):
    """Dataset that loads preprocessed features from cache directory."""

    def __init__(
        self,
        base_dataset: WLASLDataset,
        cache_dir: str | Path,
        num_frames: int = 16,
        landmark_dim: int = 99,
        hand_image_size: tuple[int, int] = (224, 224),
        augment: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
    ) -> None:
        self.base_dataset = base_dataset
        self.cache_dir = Path(cache_dir)
        self.num_frames = num_frames
        self.landmark_dim = landmark_dim
        self.hand_image_size = hand_image_size
        self.augment = augment

        self.valid_samples: list[int] = []
        for i in range(len(base_dataset)):
            metadata = base_dataset.get_metadata(i)
            cache_path = self.cache_dir / f"{metadata.video_id}.npz"
            if cache_path.exists():
                self.valid_samples.append(i)

        logger.info(
            f"CachedWLASLDataset ({base_dataset.split}): "
            f"{len(self.valid_samples)}/{len(base_dataset)} samples have cache"
        )

    def __len__(self) -> int:
        return len(self.valid_samples)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        orig_index = self.valid_samples[index]
        metadata = self.base_dataset.get_metadata(orig_index)
        cache_path = self.cache_dir / f"{metadata.video_id}.npz"

        data = np.load(str(cache_path), allow_pickle=False)

        pose = data.get(
            "pose_landmarks",
            np.zeros((self.num_frames, 33, 3), dtype=np.float32),
        )
        T = pose.shape[0]
        pose_flat = pose.reshape(T, -1)
        pose_flat = self._pad_or_truncate(pose_flat, self.num_frames, self.landmark_dim)

        left_crops = data.get(
            "left_hand_crops",
            np.zeros((self.num_frames, *self.hand_image_size, 3), dtype=np.uint8),
        )
        right_crops = data.get(
            "right_hand_crops",
            np.zeros((self.num_frames, *self.hand_image_size, 3), dtype=np.uint8),
        )

        left_tensor = self._images_to_tensor(left_crops, self.num_frames)
        right_tensor = self._images_to_tensor(right_crops, self.num_frames)

        actual_frames = min(T, self.num_frames)
        mask = np.zeros(self.num_frames, dtype=np.bool_)
        mask[:actual_frames] = True

        sample = {
            "pose_landmarks": torch.from_numpy(pose_flat).float(),
            "left_hand_images": left_tensor,
            "right_hand_images": right_tensor,
            "labels": torch.tensor(metadata.label, dtype=torch.long),
            "video_ids": metadata.video_id,
            "mask": torch.from_numpy(mask),
        }

        if self.augment is not None:
            sample = self.augment(sample)

        return sample

    def _pad_or_truncate(
        self, arr: np.ndarray, target_len: int, feat_dim: int
    ) -> np.ndarray:
        T = arr.shape[0]
        if T >= target_len:
            return arr[:target_len]
        padded = np.zeros((target_len, feat_dim), dtype=arr.dtype)
        padded[:T] = arr
        return padded

    def _images_to_tensor(self, images: np.ndarray, target_len: int) -> torch.Tensor:
        imgs = images.astype(np.float32) / 255.0
        imgs = imgs.transpose(0, 3, 1, 2)

        T = imgs.shape[0]
        if T >= target_len:
            imgs = imgs[:target_len]
        else:
            padded = np.zeros((target_len, 3, *self.hand_image_size), dtype=np.float32)
            padded[:T] = imgs
            imgs = padded

        return torch.from_numpy(imgs)


class WLASLDataModule(BaseDataModule):
    """DataModule for WLASL with cached feature loading."""

    def __init__(
        self,
        dataset_config: DatasetConfig,
        training_config: TrainingConfig,
        cache_dir: str = "cache/features/",
        augmentation_config: Optional[AugmentationConfig] = None,
        hand_image_size: tuple[int, int] = (224, 224),
        landmark_dim: int = 99,
    ) -> None:
        super().__init__(dataset_config, training_config, collate_fn=collate_fn)
        self.cache_dir = cache_dir
        self.augmentation_config = augmentation_config
        self.hand_image_size = hand_image_size
        self.landmark_dim = landmark_dim

    def setup(self, stage: Optional[str] = None) -> None:
        num_frames = self.dataset_config.num_frames

        if stage in (None, "fit"):
            train_base = WLASLDataset(
                config=self.dataset_config,
                split=self.dataset_config.train_split,
            )
            train_aug = None
            if self.augmentation_config is not None:
                train_aug = build_cache_augmentation(
                    self.augmentation_config,
                    split=self.dataset_config.train_split,
                )
            self._train_dataset = CachedWLASLDataset(
                base_dataset=train_base,
                cache_dir=self.cache_dir,
                num_frames=num_frames,
                landmark_dim=self.landmark_dim,
                hand_image_size=self.hand_image_size,
                augment=train_aug,
            )

            val_base = WLASLDataset(
                config=self.dataset_config,
                split=self.dataset_config.val_split,
            )
            self._val_dataset = CachedWLASLDataset(
                base_dataset=val_base,
                cache_dir=self.cache_dir,
                num_frames=num_frames,
                landmark_dim=self.landmark_dim,
                hand_image_size=self.hand_image_size,
            )

        if stage in (None, "test", "fit"):
            test_base = WLASLDataset(
                config=self.dataset_config,
                split=self.dataset_config.test_split,
            )
            self._test_dataset = CachedWLASLDataset(
                base_dataset=test_base,
                cache_dir=self.cache_dir,
                num_frames=num_frames,
                landmark_dim=self.landmark_dim,
                hand_image_size=self.hand_image_size,
            )

        logger.info(
            f"WLASLDataModule setup complete — "
            f"Train: {len(self._train_dataset) if self._train_dataset else 0}, "
            f"Val: {len(self._val_dataset) if self._val_dataset else 0}, "
            f"Test: {len(self._test_dataset) if self._test_dataset else 0}"
        )


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Custom collate function for the WLASL cached dataset."""
    collated: Dict[str, Any] = {}
    for key in batch[0].keys():
        values = [sample[key] for sample in batch]
        if isinstance(values[0], torch.Tensor):
            collated[key] = torch.stack(values, dim=0)
        else:
            collated[key] = values
    return collated
