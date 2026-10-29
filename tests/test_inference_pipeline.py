from types import SimpleNamespace

import numpy as np
import pytest
import torch

from src.core.config import ExperimentConfig
from src.core.types import LandmarkResult
from src.data.tid_datamodule import CachedTIDDataset
from src.preprocessing.inference_pipeline import extract_features, features_to_batch, sample_video_frames
from src.preprocessing.wrist_fallback_crop import DEFAULT_CENTER_REPAIR


@pytest.mark.parametrize("method,version,scale", [(None, None, 1.), ("pose_wrist_fallback_crop", 1, 1.),
    ("pose_wrist_fallback_crop", 2, 1.5), ("pose_wrist_fallback_crop", 2, 1.75),
    ("pose_wrist_fallback_crop", 3, 1.5),
    ("short_hand_gap_linear", 2, 1.)])
def test_inference_matches_cached_validation_joint_order_pixels_and_padding(tmp_path, method, version, scale):
    config = ExperimentConfig()
    config.dataset.num_frames = 4
    config.dataset.cache_derivation = method
    config.dataset.cache_derivation_version = version
    config.dataset.fallback_box_scale = scale
    config.dataset.fallback_center_repair = dict(DEFAULT_CENTER_REPAIR) if version==3 else None
    config.dataset.max_repair_gap = 2 if method == "short_hand_gap_linear" else None
    config.model.landmark_encoder.input_dim = 225
    config.model.hand_encoder.input_size = [8, 8]
    frames = [np.full((100, 100, 3), 30 + t*40, np.uint8) for t in range(3)]
    pose = np.full((33, 3), .5, np.float32)
    pose[[15, 17, 19, 21], 0] = .2
    pose[[16, 18, 20, 22], 0] = .8
    left = np.zeros((21, 3), np.float32)
    left[:, 0] = np.linspace(.15, .25, 21)
    left[:, 1] = np.linspace(.45, .55, 21)
    left[0, :2] = [.2, .5]
    right = left.copy()
    right[:, 0] += .6
    detections = [LandmarkResult(pose=pose.copy(), left_hand=left if t != 1 else None,
                                right_hand=right) for t in range(3)]
    if version==3:
        detections[1].pose[[15,17,19,21],0] = .9
    results = iter(detections)
    extractor = SimpleNamespace(ensure_required_models=lambda: None, extract=lambda frame: next(results))
    features = extract_features(frames, config, extractor=extractor)
    features["label"] = np.array(0)
    if method == "pose_wrist_fallback_crop":
        assert features["left_hand_crops"][1, 4, 4, 0] == 70
        if version==3:
            assert features['left_crop_source'][1] == 3
            assert features['left_crop_center_imputed'][1]
    elif method is None:
        assert np.array_equal(features["left_hand_crops"][1], features["left_hand_crops"][0])
    else:
        assert features["left_hand_imputed"].tolist() == [False, True, False]
    (tmp_path / "val").mkdir()
    np.savez(tmp_path / "val/video.npz", **features)
    base = SimpleNamespace(config=SimpleNamespace(name="tid"), split="val",
                           get_metadata=lambda index: SimpleNamespace(video_id="video", label=0))
    cached = CachedTIDDataset(base, tmp_path, num_frames=4, landmark_dim=225,
                              hand_image_size=(8, 8), indices=[0])[0]
    batch = features_to_batch(features, config)
    for key in ("pose_landmarks", "left_hand_images", "right_hand_images", "mask"):
        assert torch.equal(batch[key][0], cached[key]), key
    assert batch["pose_landmarks"].shape == (1, 4, 225)
    assert batch["mask"].tolist() == [[True, True, True, False]]


def test_failed_video_decode_is_not_silently_shortened(monkeypatch):
    capture = SimpleNamespace(isOpened=lambda: True, get=lambda prop: 4,
                              set=lambda *args: None, read=lambda: (False, None),
                              release=lambda: None)
    monkeypatch.setattr("src.preprocessing.inference_pipeline.cv2.VideoCapture", lambda path: capture)
    with pytest.raises(ValueError, match="could not be decoded"):
        sample_video_frames("bad.mp4", 4)
