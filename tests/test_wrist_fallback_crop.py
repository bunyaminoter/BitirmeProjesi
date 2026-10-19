import numpy as np

from src.preprocessing.wrist_fallback_crop import (
    SOURCE_DETECTED, SOURCE_POSE_FALLBACK, SOURCE_UNCHANGED, fallback_crops, square_crop,
)


def _frames(count, width=200, height=100):
    frames = []
    for t in range(count):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[:, :, 0] = t * 20  # frame identity in red channel
        frames.append(frame)
    return frames


def test_only_missing_frames_with_pose_get_current_frame_crop():
    count = 4
    frames = _frames(count)
    crops = np.full((count, 32, 32, 3), 7, dtype=np.uint8)  # "stale" carry-forward value
    hands = np.zeros((count, 21, 3), dtype=np.float32)
    hands[:, :, 0] = np.linspace(0.45, 0.55, 21)
    hands[:, :, 1] = np.linspace(0.4, 0.6, 21)
    observed = np.array([True, False, False, True])
    pose = np.zeros((count, 33, 3), dtype=np.float32)
    pose[:, [15, 17, 19, 21], 0] = 0.5
    pose[:, [15, 17, 19, 21], 1] = 0.5
    pose_ok = np.array([True, True, False, True])
    new, source = fallback_crops(frames, crops, hands, observed, pose, pose_ok, "left")
    assert source.tolist() == [SOURCE_DETECTED, SOURCE_POSE_FALLBACK, SOURCE_UNCHANGED, SOURCE_DETECTED]
    assert np.array_equal(new[[0, 2, 3]], crops[[0, 2, 3]])
    assert new[1, 16, 16, 0] == 20  # pixels taken from frame 1, not the stale crop


def test_square_crop_keeps_scale_and_pads_outside_black():
    frame = np.full((50, 50, 3), 255, dtype=np.uint8)
    crop = square_crop(frame, np.array([0.0, 25.0]), 20.0, (20, 20))
    assert crop.shape == (20, 20, 3)
    assert crop[:, :9].max() == 0 and crop[:, 11:].min() == 255
