"""Official AUTSL format and cache completeness checks."""

from __future__ import annotations

import pytest
import json

from src.core.config import DatasetConfig
from src.data.datasets.tid_dataset import TIDDataset
from src.data.tid_datamodule import CachedTIDDataset


@pytest.fixture
def official_root(tmp_path, monkeypatch):
    (tmp_path / "SignList_ClassId_TR_EN.csv").write_text(
        "ClassId,TR,EN\n0,anne,mother\n1,baba,father\n2,ev,house\n", encoding="utf-8"
    )
    for split, signer in (("train", 0), ("val", 1), ("test", 2)):
        video_id = f"signer{signer}_sample1"
        (tmp_path / f"{split}_labels.csv").write_text(
            f"{video_id},{signer}\n", encoding="utf-8"
        )
        (tmp_path / split).mkdir()
        (tmp_path / split / f"{video_id}_color.mp4").touch()
    monkeypatch.setenv("AUTSL_DATA_DIR", str(tmp_path))
    return tmp_path


def test_official_labels_keep_class_ids_and_split_paths(official_root):
    config = DatasetConfig(name="autsl", num_classes=3)
    for split, label in (("train", 0), ("val", 1), ("test", 2)):
        dataset = TIDDataset(config, split)
        assert len(dataset) == 1
        assert dataset.get_label(0) == label
        assert dataset.get_metadata(0).signer_id == f"signer{label}"
        assert dataset.get_video_path(0) == official_root / split / f"signer{label}_sample1_color.mp4"
        assert dataset.idx_to_class == {0: "anne", 1: "baba", 2: "ev"}


def test_missing_official_cache_is_an_error(official_root, tmp_path):
    dataset = TIDDataset(DatasetConfig(name="autsl", num_classes=3), "train")
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    (cache_dir / "manifest.json").write_text(json.dumps({
        "dataset": "autsl", "pipeline_version": 2, "num_frames": 16,
        "hand_image_size": [160, 160], "num_classes": 3, "class_ids": None,
    }), encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="feature caches are missing"):
        CachedTIDDataset(dataset, cache_dir)
    with pytest.raises(ValueError, match="cache settings differ"):
        CachedTIDDataset(dataset, cache_dir, num_frames=32)


def test_derived_cache_cannot_be_mistaken_for_baseline(official_root, tmp_path):
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    manifest = {
        "dataset": "autsl", "pipeline_version": 2, "num_frames": 16,
        "hand_image_size": [160, 160], "num_classes": 3, "class_ids": None,
        "derivation": {"method": "short_hand_gap_linear", "max_gap": 2},
    }
    (cache_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    base = TIDDataset(DatasetConfig(name="autsl", num_classes=3), "train")
    with pytest.raises(ValueError, match="cache settings differ"):
        CachedTIDDataset(base, cache_dir)
    repaired = TIDDataset(DatasetConfig(
        name="autsl", num_classes=3,
        cache_derivation="short_hand_gap_linear", max_repair_gap=2,
    ), "train")
    with pytest.raises(FileNotFoundError, match="feature caches are missing"):
        CachedTIDDataset(repaired, cache_dir)


def test_duplicate_video_id_is_rejected(official_root):
    labels = official_root / "train_labels.csv"
    labels.write_text("signer0_sample1,0\nsigner0_sample1,1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate AUTSL labels"):
        TIDDataset(DatasetConfig(name="autsl", num_classes=3), "train")


def test_derived_cache_version_must_match_config(official_root, tmp_path):
    cache_dir = tmp_path / "versioned_cache"
    cache_dir.mkdir()
    (cache_dir / "manifest.json").write_text(json.dumps({
        "dataset": "autsl", "pipeline_version": 2, "num_frames": 16,
        "hand_image_size": [160, 160], "num_classes": 3, "class_ids": None,
        "derivation": {"method": "short_hand_gap_linear", "max_gap": 2, "version": 1},
    }), encoding="utf-8")
    dataset = TIDDataset(DatasetConfig(name="autsl", num_classes=3,
        cache_derivation="short_hand_gap_linear", max_repair_gap=2,
        cache_derivation_version=2), "train")
    with pytest.raises(ValueError, match="cache settings differ"):
        CachedTIDDataset(dataset, cache_dir)


def test_scaled_fallback_cache_scale_must_match_config(official_root, tmp_path):
    cache_dir = tmp_path / "scaled_cache"
    cache_dir.mkdir()
    (cache_dir / "manifest.json").write_text(json.dumps({
        "dataset": "autsl", "pipeline_version": 2, "num_frames": 16,
        "hand_image_size": [160, 160], "num_classes": 3, "class_ids": None,
        "derivation": {"method": "pose_wrist_fallback_crop", "max_gap": None,
                       "version": 2, "box_scale": 1.75},
    }), encoding="utf-8")
    dataset = TIDDataset(DatasetConfig(name="autsl", num_classes=3,
        cache_derivation="pose_wrist_fallback_crop", cache_derivation_version=2,
        fallback_box_scale=1.5), "train")
    with pytest.raises(ValueError, match="cache settings differ"):
        CachedTIDDataset(dataset, cache_dir)


def test_center_repair_policy_must_match_manifest(official_root, tmp_path):
    from src.preprocessing.wrist_fallback_crop import DEFAULT_CENTER_REPAIR
    cache_dir = tmp_path/'center_cache'
    cache_dir.mkdir()
    policy = dict(DEFAULT_CENTER_REPAIR)
    manifest_policy = dict(policy, min_disagreement_ratio=.5)
    (cache_dir/'manifest.json').write_text(json.dumps({
        'dataset':'autsl', 'pipeline_version':2, 'num_frames':16,
        'hand_image_size':[160,160], 'num_classes':3, 'class_ids':None,
        'derivation':{'method':'pose_wrist_fallback_crop','version':3,'max_gap':None,
                      'box_scale':1.5,'center_repair':manifest_policy},
    }),encoding='utf-8')
    dataset=TIDDataset(DatasetConfig(name='autsl',num_classes=3,
        cache_derivation='pose_wrist_fallback_crop',cache_derivation_version=3,
        fallback_box_scale=1.5,fallback_center_repair=policy),'train')
    with pytest.raises(ValueError,match='cache settings differ'):
        CachedTIDDataset(dataset,cache_dir)


def test_autsl_subset_remaps_original_ids_to_contiguous_model_labels(official_root):
    config = DatasetConfig(name="autsl", num_classes=2, class_ids=[2, 0])
    train = TIDDataset(config, "train")
    val = TIDDataset(config, "val")
    test = TIDDataset(config, "test")
    assert train.get_label(0) == 1  # source class 0 is the second selected class
    assert len(val) == 0            # source class 1 is intentionally excluded
    assert test.get_label(0) == 0   # source class 2 is the first selected class
    assert train.idx_to_class == {0: "ev", 1: "anne"}


def test_cached_dataset_load_rgb_toggle(official_root, tmp_path):
    import numpy as np
    import torch
    cache_dir = tmp_path / "cache_toggle"
    cache_dir.mkdir()
    (cache_dir / "manifest.json").write_text(json.dumps({
        "dataset": "autsl", "pipeline_version": 2, "num_frames": 16,
        "hand_image_size": [160, 160], "num_classes": 3, "class_ids": None,
    }), encoding="utf-8")
    train_dir = cache_dir / "train"
    train_dir.mkdir()
    np.savez(
        train_dir / "signer0_sample1_color.npz",
        pose_landmarks=np.ones((16, 33, 3), dtype=np.float32),
        left_hand_landmarks=np.ones((16, 21, 3), dtype=np.float32),
        right_hand_landmarks=np.ones((16, 21, 3), dtype=np.float32),
        left_hand_crops=np.full((16, 160, 160, 3), 128, dtype=np.uint8),
        right_hand_crops=np.full((16, 160, 160, 3), 128, dtype=np.uint8),
        label=0,
        num_frames=16,
    )
    dataset = TIDDataset(DatasetConfig(name="autsl", num_classes=3), "train")
    
    # 1. With load_rgb=True (default)
    ds_rgb = CachedTIDDataset(dataset, cache_dir, load_rgb=True)
    sample_rgb = ds_rgb[0]
    assert sample_rgb["left_hand_images"].shape == (16, 3, 160, 160)
    assert sample_rgb["pose_landmarks"].shape == (16, 225)

    # 2. With load_rgb=False (pose-only optimization)
    ds_pose = CachedTIDDataset(dataset, cache_dir, load_rgb=False)
    sample_pose = ds_pose[0]
    assert "left_hand_images" not in sample_pose
    assert "right_hand_images" not in sample_pose
    assert sample_pose["pose_landmarks"].shape == (16, 225)
    assert torch.allclose(sample_pose["pose_landmarks"], sample_rgb["pose_landmarks"])

    # Verify CacheAugmentation and HybridASLModel work without hand crop keys
    from src.core.config import AugmentationConfig, LandmarkEncoderConfig, ModelConfig
    from src.data.transforms.cache_augmentation import CacheAugmentation
    from src.models.hybrid_model import HybridASLModel
    from src.data.tid_datamodule import collate_fn

    aug = CacheAugmentation(AugmentationConfig(enabled=True, pose_jitter_std=0.01))
    aug_sample = aug(sample_pose)
    assert "left_hand_images" not in aug_sample
    assert aug_sample["pose_landmarks"].shape == (16, 225)

    batch = collate_fn([sample_pose])
    model = HybridASLModel(
        ModelConfig(
            use_rgb=False,
            num_classes=3,
            landmark_encoder=LandmarkEncoderConfig(input_dim=225, num_pose_landmarks=75),
        )
    )
    model.eval()
    with torch.no_grad():
        logits = model(batch)
    assert logits.shape == (1, 3)


def test_checkpoint_callback_resume_preserves_best_value(tmp_path):
    from src.training.callbacks.checkpoint import CheckpointCallback

    class DummyTrainer:
        def __init__(self):
            self.current_epoch = 5
            self.best_metric = 0.90
            self.saved = []

        def save_checkpoint(self, path):
            self.saved.append(path.name)

    trainer = DummyTrainer()
    cb = CheckpointCallback(save_dir=tmp_path, monitor="accuracy", mode="max")
    cb.on_train_start(trainer)
    assert cb.best_value == 0.90

    # Lower accuracy after resume must NOT overwrite best_model.pt
    cb.on_validation_end(trainer, epoch=5, metrics={"accuracy": 0.75})
    assert "best_model.pt" not in trainer.saved
    assert "latest_model.pt" in trainer.saved

    # Higher accuracy after resume MUST update best_model.pt
    trainer.saved.clear()
    cb.on_validation_end(trainer, epoch=6, metrics={"accuracy": 0.92})
    assert "best_model.pt" in trainer.saved
    assert cb.best_value == 0.92


