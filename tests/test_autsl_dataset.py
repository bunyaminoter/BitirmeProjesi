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


def test_autsl_subset_remaps_original_ids_to_contiguous_model_labels(official_root):
    config = DatasetConfig(name="autsl", num_classes=2, class_ids=[2, 0])
    train = TIDDataset(config, "train")
    val = TIDDataset(config, "val")
    test = TIDDataset(config, "test")
    assert train.get_label(0) == 1  # source class 0 is the second selected class
    assert len(val) == 0            # source class 1 is intentionally excluded
    assert test.get_label(0) == 0   # source class 2 is the first selected class
    assert train.idx_to_class == {0: "ev", 1: "anne"}
