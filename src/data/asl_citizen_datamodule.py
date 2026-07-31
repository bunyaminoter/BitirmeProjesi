"""
ASL Citizen DataModule.
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
from src.data.datasets.asl_citizen_dataset import ASLCitizenDataset
from src.data.transforms.cache_augmentation import build_cache_augmentation

logger = logging.getLogger(__name__)


class CachedASLCitizenDataset(Dataset):
    def __init__(
        self,
        base_dataset: ASLCitizenDataset,
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

        # Kaggle environment fallback for cache
        if not self.cache_dir.exists():
            kaggle_cache = Path("/kaggle/working/cache/asl_citizen_features")
            if kaggle_cache.exists() or Path("/kaggle/working").exists():
                self.cache_dir = kaggle_cache

        self.valid_samples: list[int] = []
        for i in range(len(base_dataset)):
            metadata = base_dataset.get_metadata(i)
            cache_path = self.cache_dir / f"{metadata.video_id}.npz"
            if cache_path.exists():
                self.valid_samples.append(i)

        logger.info(
            f"CachedASLCitizenDataset ({base_dataset.split}): "
            f"{len(self.valid_samples)}/{len(base_dataset)} samples have cache in {self.cache_dir}"
        )

    def __len__(self) -> int:
        return len(self.valid_samples)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        orig_index = self.valid_samples[index]
        metadata = self.base_dataset.get_metadata(orig_index)
        cache_path = self.cache_dir / f"{metadata.video_id}.npz"

        data = np.load(str(cache_path), allow_pickle=False)

        pose = data.get("pose_landmarks", np.zeros((self.num_frames, 33, 3), dtype=np.float32))
        T = pose.shape[0]
        pose_flat = pose.reshape(T, -1)
        pose_flat = self._pad_or_truncate(pose_flat, self.num_frames, self.landmark_dim)

        left_crops = data.get("left_hand_crops", np.zeros((self.num_frames, *self.hand_image_size, 3), dtype=np.uint8))
        right_crops = data.get("right_hand_crops", np.zeros((self.num_frames, *self.hand_image_size, 3), dtype=np.uint8))

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

    def _pad_or_truncate(self, arr: np.ndarray, target_len: int, feat_dim: int) -> np.ndarray:
        T = arr.shape[0]
        if T >= target_len: return arr[:target_len]
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


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Custom collate function for the cached dataset."""
    collated: Dict[str, Any] = {}
    for key in batch[0].keys():
        values = [sample[key] for sample in batch]
        if isinstance(values[0], torch.Tensor):
            collated[key] = torch.stack(values, dim=0)
        else:
            collated[key] = values
    return collated

class ASLCitizenDataModule(BaseDataModule):
    def __init__(
        self,
        dataset_config: DatasetConfig,
        training_config: TrainingConfig,
        cache_dir: str = "cache/asl_citizen_features/",
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
        num_classes = self.dataset_config.num_classes

        # Train veri seti: preprocessor ile aynı top_n_classes kullanarak
        # aynı 100 sınıfı seçmesini sağlıyoruz
        if stage in (None, "fit"):
            train_base = ASLCitizenDataset(
                self.dataset_config, self.dataset_config.train_split,
                top_n_classes=num_classes,
            )

            # Sınıf listesini dosyaya kaydet (val/test aynı listeyi kullanacak)
            class_list_path = Path(self.dataset_config.class_list_file)
            if train_base.class_to_idx and not class_list_path.exists():
                class_list_path.parent.mkdir(parents=True, exist_ok=True)
                with open(class_list_path, "w", encoding="utf-8") as f:
                    for idx in sorted(train_base.idx_to_class.keys()):
                        f.write(f"{idx} {train_base.idx_to_class[idx]}\n")
                logger.info(f"Class list saved: {class_list_path} ({len(train_base.class_to_idx)} classes)")

            train_aug = build_cache_augmentation(self.augmentation_config, self.dataset_config.train_split) if self.augmentation_config else None
            self._train_dataset = CachedASLCitizenDataset(
                train_base, self.cache_dir, num_frames, self.landmark_dim, self.hand_image_size, train_aug
            )

            # Val veri seti: kaydedilen class_list.txt otomatik yüklenecek
            val_base = ASLCitizenDataset(self.dataset_config, self.dataset_config.val_split)
            self._val_dataset = CachedASLCitizenDataset(
                val_base, self.cache_dir, num_frames, self.landmark_dim, self.hand_image_size
            )

        if stage in (None, "test", "fit"):
            test_base = ASLCitizenDataset(self.dataset_config, self.dataset_config.test_split)
            self._test_dataset = CachedASLCitizenDataset(
                test_base, self.cache_dir, num_frames, self.landmark_dim, self.hand_image_size
            )
