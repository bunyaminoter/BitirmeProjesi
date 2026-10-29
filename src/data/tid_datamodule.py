"""
TİD (Türk İşaret Dili / AUTSL) DataModule & Cache Loader.

Bu modül:
1. MediaPipe 75 eklem koordinatları ve el kırpıntılarını (160x160) önceden çıkarılmış
   .npz cache dosyalarından okur (CachedTIDDataset).
2. Veri artırma (Augmentation), normalizasyon ve dinamik padding/truncation uygular.
3. Train, Validation ve Kilitli Test splitleri için DataLoader'ları yönetir (TIDDataModule).
4. K-Fold Cross-Validation için `setup_fold(train_indices, val_indices)` desteği sunar.
"""

from __future__ import annotations

import logging
import json
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from src.core.config import AugmentationConfig, DatasetConfig, TrainingConfig
from src.data.base_datamodule import BaseDataModule
from src.data.datasets.tid_dataset import TIDDataset
from src.data.transforms.cache_augmentation import build_cache_augmentation

logger = logging.getLogger(__name__)


class CachedTIDDataset(Dataset):
    """TİD (AUTSL) için önceden çıkarılmış özellikleri (.npz) diskten okuyan veri seti.
    
    Her video için:
      - pose_landmarks: (num_frames, 225) -> 33 pose + 21 sol el + 21 sağ el (75 eklem * 3)
      - left_hand_images: (num_frames, 3, 160, 160) -> ImageNet normalize edilmiş el kırpıntıları
      - right_hand_images: (num_frames, 3, 160, 160)
      - labels: Sınıf indeksi (0-9)
      - mask: Gerçek kareleri belirten boolean maske
    """

    def __init__(
        self,
        base_dataset: TIDDataset,
        cache_dir: str | Path,
        num_frames: int = 16,
        landmark_dim: int = 225,
        hand_image_size: tuple[int, int] = (160, 160),
        augment: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
        indices: Optional[List[int]] = None,
        load_rgb: bool = True,
    ) -> None:
        self.base_dataset = base_dataset
        self.cache_dir = Path(cache_dir)
        self.num_frames = num_frames
        self.landmark_dim = landmark_dim
        self.hand_image_size = hand_image_size
        self.augment = augment
        self.load_rgb = load_rgb

        if base_dataset.config.name == "autsl":
            manifest_path = self.cache_dir / "manifest.json"
            if not manifest_path.is_file():
                raise FileNotFoundError(f"AUTSL cache manifest missing: {manifest_path}")
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if (
                manifest.get("dataset") != "autsl"
                or manifest.get("pipeline_version") != 2
                or manifest.get("num_frames") != num_frames
                or manifest.get("hand_image_size") != list(hand_image_size)
                or manifest.get("num_classes") != base_dataset.config.num_classes
                or manifest.get("class_ids") != base_dataset.config.class_ids
                or (manifest.get("derivation") or {}).get("method") != base_dataset.config.cache_derivation
                or (manifest.get("derivation") or {}).get("max_gap") != base_dataset.config.max_repair_gap
                or (base_dataset.config.cache_derivation_version is not None
                    and (manifest.get("derivation") or {}).get("version") != base_dataset.config.cache_derivation_version)
                or (manifest.get("derivation") or {}).get("box_scale", 1.0) != base_dataset.config.fallback_box_scale
                or (manifest.get("derivation") or {}).get("center_repair") != base_dataset.config.fallback_center_repair
            ):
                raise ValueError(f"AUTSL cache settings differ from {manifest_path}")

        # Kaggle fallback
        if not self.cache_dir.exists():
            kaggle_cache = Path("/kaggle/working/cache/tid_10_features")
            if kaggle_cache.exists():
                self.cache_dir = kaggle_cache

        self.valid_samples: list[int] = []
        self._cache_paths: dict[int, Path] = {}
        self._landmark_cache: dict[int, tuple[np.ndarray, int]] = {}
        alt_drive = Path(f"/content/drive/MyDrive/BitirmeProjesi/cache/{self.cache_dir.name}")
        sample_indices = indices if indices is not None else range(len(base_dataset))

        candidate_dirs: list[Path] = [
            self.cache_dir / base_dataset.split,
            self.cache_dir,
        ]
        if alt_drive.exists():
            candidate_dirs.extend([
                alt_drive / base_dataset.split,
                alt_drive,
            ])

        dir_listings: list[tuple[Path, set[str]]] = []
        for d in candidate_dirs:
            try:
                files = set(os.listdir(d)) if d.is_dir() else set()
            except OSError:
                files = set()
            dir_listings.append((d, files))

        for i in sample_indices:
            metadata = base_dataset.get_metadata(i)
            fname = f"{metadata.video_id}.npz"
            resolved: Optional[Path] = None
            for d, files in dir_listings:
                if fname in files:
                    resolved = d / fname
                    break
            if resolved is not None:
                self.valid_samples.append(i)
                self._cache_paths[i] = resolved

        if base_dataset.config.name == "autsl" and len(self.valid_samples) != len(sample_indices):
            found = set(self.valid_samples)
            missing = [base_dataset.get_metadata(i).video_id for i in sample_indices if i not in found]
            raise FileNotFoundError(
                f"AUTSL {base_dataset.split}: {len(missing)} of {len(sample_indices)} "
                f"feature caches are missing from {self.cache_dir}; first IDs: {missing[:5]}"
            )

        subset_tag = f" (subset of {len(indices)})" if indices is not None else ""
        logger.info(
            f"CachedTIDDataset ({base_dataset.split}{subset_tag}): "
            f"{len(self.valid_samples)}/{len(sample_indices)} samples have cache in {self.cache_dir} (or {alt_drive})"
        )

    def __len__(self) -> int:
        return len(self.valid_samples)

    def _resolve_cache_path(self, orig_index: int) -> Path:
        cached = self._cache_paths.get(orig_index)
        if cached is not None:
            return cached
        metadata = self.base_dataset.get_metadata(orig_index)
        cache_path = self.cache_dir / self.base_dataset.split / f"{metadata.video_id}.npz"
        if not cache_path.exists():
            cache_path = self.cache_dir / f"{metadata.video_id}.npz"
        if not cache_path.exists():
            alt_drive = Path(f"/content/drive/MyDrive/BitirmeProjesi/cache/{self.cache_dir.name}")
            if (alt_drive / self.base_dataset.split / f"{metadata.video_id}.npz").exists():
                cache_path = alt_drive / self.base_dataset.split / f"{metadata.video_id}.npz"
            elif (alt_drive / f"{metadata.video_id}.npz").exists():
                cache_path = alt_drive / f"{metadata.video_id}.npz"
        self._cache_paths[orig_index] = cache_path
        return cache_path

    def _load_landmarks_for_index(self, orig_index: int) -> tuple[np.ndarray, int]:
        if orig_index in self._landmark_cache:
            pose_flat, T = self._landmark_cache[orig_index]
            return pose_flat.copy(), T

        metadata = self.base_dataset.get_metadata(orig_index)
        cache_path = self._resolve_cache_path(orig_index)
        with np.load(str(cache_path), allow_pickle=False) as data:
            if self.base_dataset.config.name == "autsl":
                if int(data["label"]) != metadata.label or int(data["num_frames"]) > self.num_frames:
                    raise ValueError(f"AUTSL cache metadata mismatch: {cache_path}")
            files = set(data.files)
            pose = data["pose_landmarks"] if "pose_landmarks" in files else np.zeros((self.num_frames, 33, 3), dtype=np.float32)
            left_hand = data["left_hand_landmarks"] if "left_hand_landmarks" in files else np.zeros((self.num_frames, 21, 3), dtype=np.float32)
            right_hand = data["right_hand_landmarks"] if "right_hand_landmarks" in files else np.zeros((self.num_frames, 21, 3), dtype=np.float32)

        T = int(pose.shape[0])
        combined_landmarks = np.concatenate([pose, left_hand, right_hand], axis=1)
        pose_flat = combined_landmarks.reshape(T, -1)
        pose_flat = self._pad_or_truncate(pose_flat, self.num_frames, self.landmark_dim).astype(np.float32, copy=False)
        self._landmark_cache[orig_index] = (pose_flat, T)
        return pose_flat.copy(), T

    def preload_landmarks(self, max_workers: int = 8) -> None:
        """Pose-only modunda tüm iskelet özelliklerini çok iş parçacıklı olarak RAM'e yükler."""
        if self.load_rgb or len(self._landmark_cache) == len(self.valid_samples):
            return
        from concurrent.futures import ThreadPoolExecutor
        from tqdm import tqdm

        missing_indices = [idx for idx in self.valid_samples if idx not in self._landmark_cache]
        if not missing_indices:
            return
        desc = f"RAM Cache ({self.base_dataset.split})"
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            list(
                tqdm(
                    pool.map(self._load_landmarks_for_index, missing_indices),
                    total=len(missing_indices),
                    desc=desc,
                    leave=False,
                )
            )
        logger.info(
            f"Preloaded {len(self._landmark_cache)}/{len(self.valid_samples)} "
            f"pose landmark sequences into RAM for '{self.base_dataset.split}' split."
        )

    def __getitem__(self, index: int) -> Dict[str, Any]:
        orig_index = self.valid_samples[index]
        metadata = self.base_dataset.get_metadata(orig_index)

        if not self.load_rgb:
            pose_flat, T = self._load_landmarks_for_index(orig_index)
            actual_frames = min(T, self.num_frames)
            mask = np.zeros(self.num_frames, dtype=np.bool_)
            mask[:actual_frames] = True

            sample: Dict[str, Any] = {
                "pose_landmarks": torch.from_numpy(pose_flat).float(),
                "labels": torch.tensor(metadata.label, dtype=torch.long),
                "video_ids": metadata.video_id,
                "mask": torch.from_numpy(mask),
            }
            if self.augment is not None:
                sample = self.augment(sample)
            return sample

        cache_path = self._resolve_cache_path(orig_index)
        with np.load(str(cache_path), allow_pickle=False) as data:
            if self.base_dataset.config.name == "autsl":
                if int(data["label"]) != metadata.label or int(data["num_frames"]) > self.num_frames:
                    raise ValueError(f"AUTSL cache metadata mismatch: {cache_path}")

            files = set(data.files)
            if self.base_dataset.config.name == "autsl" and (
                "left_hand_crops" not in files or "right_hand_crops" not in files
            ):
                raise ValueError(f"AUTSL RGB cache is missing hand crops: {cache_path}")
            pose = data["pose_landmarks"] if "pose_landmarks" in files else np.zeros((self.num_frames, 33, 3), dtype=np.float32)
            left_hand = data["left_hand_landmarks"] if "left_hand_landmarks" in files else np.zeros((self.num_frames, 21, 3), dtype=np.float32)
            right_hand = data["right_hand_landmarks"] if "right_hand_landmarks" in files else np.zeros((self.num_frames, 21, 3), dtype=np.float32)
            left_crops = data["left_hand_crops"] if "left_hand_crops" in files else np.zeros((self.num_frames, *self.hand_image_size, 3), dtype=np.uint8)
            right_crops = data["right_hand_crops"] if "right_hand_crops" in files else np.zeros((self.num_frames, *self.hand_image_size, 3), dtype=np.uint8)

        T = pose.shape[0]
        # Concat along the joints (V) dimension -> (T, 33+21+21, 3) = (T, 75, 3)
        combined_landmarks = np.concatenate([pose, left_hand, right_hand], axis=1)

        # Flatten to (T, 75*3=225)
        pose_flat = combined_landmarks.reshape(T, -1)
        pose_flat = self._pad_or_truncate(pose_flat, self.num_frames, self.landmark_dim)

        # Çözünürlük farklıysa yeniden boyutlandır
        left_crops = self._resize_crops(left_crops)
        right_crops = self._resize_crops(right_crops)

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

        # ImageNet normalizasyonu (RGB el kırpıntıları için)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        sample["left_hand_images"] = (sample["left_hand_images"] - mean) / std
        sample["right_hand_images"] = (sample["right_hand_images"] - mean) / std

        return sample

    def _resize_crops(self, crops: np.ndarray) -> np.ndarray:
        """Hand crop boyutlarını hand_image_size ile eşleştirir."""
        h, w = self.hand_image_size
        if crops.ndim == 4 and (crops.shape[1] != h or crops.shape[2] != w):
            resized = np.zeros((crops.shape[0], h, w, 3), dtype=crops.dtype)
            for i in range(crops.shape[0]):
                resized[i] = cv2.resize(crops[i], (w, h), interpolation=cv2.INTER_LINEAR)
            return resized
        return crops

    def _pad_or_truncate(self, arr: np.ndarray, target_len: int, feat_dim: int) -> np.ndarray:
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


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """TİD batch birleştirici."""
    collated: Dict[str, Any] = {}
    for key in batch[0].keys():
        values = [sample[key] for sample in batch]
        if isinstance(values[0], torch.Tensor):
            collated[key] = torch.stack(values, dim=0)
        else:
            collated[key] = values
    return collated


class TIDDataModule(BaseDataModule):
    """Türk İşaret Dili (TİD / AUTSL) DataModule."""

    def __init__(
        self,
        dataset_config: DatasetConfig,
        training_config: TrainingConfig,
        cache_dir: str = "cache/tid_10_features/",
        augmentation_config: Optional[AugmentationConfig] = None,
        hand_image_size: tuple[int, int] = (160, 160),
        landmark_dim: int = 225,
        load_rgb: bool = True,
    ) -> None:
        super().__init__(dataset_config, training_config, collate_fn=collate_fn)
        self.cache_dir = cache_dir
        self.augmentation_config = augmentation_config
        self.hand_image_size = hand_image_size
        self.landmark_dim = landmark_dim
        self.load_rgb = load_rgb

    def preload_landmarks(self, max_workers: int = 8) -> None:
        """Pose-only modunda aktif veri kümelerinin iskelet özelliklerini RAM'e önyükler."""
        if self.load_rgb:
            return
        for ds in (self._train_dataset, self._val_dataset, self._test_dataset):
            if ds is not None:
                ds.preload_landmarks(max_workers=max_workers)

    def setup(self, stage: Optional[str] = None) -> None:
        num_frames = self.dataset_config.num_frames
        num_classes = self.dataset_config.num_classes

        if stage in (None, "fit"):
            train_base = TIDDataset(
                self.dataset_config, self.dataset_config.train_split,
                top_n_classes=num_classes,
            )

            # Sınıf listesini dosyaya kaydet
            class_list_path = Path(self.dataset_config.class_list_file)
            if train_base.class_to_idx and not class_list_path.exists():
                class_list_path.parent.mkdir(parents=True, exist_ok=True)
                with open(class_list_path, "w", encoding="utf-8") as f:
                    for idx in sorted(train_base.idx_to_class.keys()):
                        f.write(f"{idx} {train_base.idx_to_class[idx]}\n")
                logger.info(f"Class list saved: {class_list_path} ({len(train_base.class_to_idx)} classes)")

            train_aug = build_cache_augmentation(self.augmentation_config, self.dataset_config.train_split) if self.augmentation_config else None
            self._train_dataset = CachedTIDDataset(
                train_base, self.cache_dir, num_frames, self.landmark_dim, self.hand_image_size, train_aug,
                load_rgb=self.load_rgb,
            )

        if stage in (None, "fit", "validate"):
            val_base = TIDDataset(self.dataset_config, self.dataset_config.val_split)
            self._val_dataset = CachedTIDDataset(
                val_base, self.cache_dir, num_frames, self.landmark_dim, self.hand_image_size,
                load_rgb=self.load_rgb,
            )

        if stage in (None, "test") or (stage == "fit" and self.dataset_config.name != "autsl"):
            test_base = TIDDataset(self.dataset_config, self.dataset_config.test_split)
            self._test_dataset = CachedTIDDataset(
                test_base, self.cache_dir, num_frames, self.landmark_dim, self.hand_image_size,
                load_rgb=self.load_rgb,
            )

    def setup_fold(self, train_indices: List[int], val_indices: List[int]) -> None:
        """K-Fold cross validation için train ve val veri kümelerini indislerden hazırlar."""
        num_frames = self.dataset_config.num_frames
        num_classes = self.dataset_config.num_classes

        train_base = TIDDataset(
            self.dataset_config, self.dataset_config.train_split,
            top_n_classes=num_classes,
        )

        class_list_path = Path(self.dataset_config.class_list_file)
        if train_base.class_to_idx and not class_list_path.exists():
            class_list_path.parent.mkdir(parents=True, exist_ok=True)
            with open(class_list_path, "w", encoding="utf-8") as f:
                for idx in sorted(train_base.idx_to_class.keys()):
                    f.write(f"{idx} {train_base.idx_to_class[idx]}\n")

        # Train katı: augmentasyon aktif
        train_aug = build_cache_augmentation(self.augmentation_config, self.dataset_config.train_split) if self.augmentation_config else None
        self._train_dataset = CachedTIDDataset(
            train_base, self.cache_dir, num_frames, self.landmark_dim, self.hand_image_size,
            augment=train_aug, indices=train_indices, load_rgb=self.load_rgb,
        )

        # Val katı: augmentasyonsuz temiz değerlendirme
        self._val_dataset = CachedTIDDataset(
            train_base, self.cache_dir, num_frames, self.landmark_dim, self.hand_image_size,
            augment=None, indices=val_indices, load_rgb=self.load_rgb,
        )

        # Kilitli Test seti: her zaman bağımsız test splitini kullanır
        test_base = TIDDataset(self.dataset_config, self.dataset_config.test_split)
        self._test_dataset = CachedTIDDataset(
            test_base, self.cache_dir, num_frames, self.landmark_dim, self.hand_image_size,
            load_rgb=self.load_rgb,
        )


