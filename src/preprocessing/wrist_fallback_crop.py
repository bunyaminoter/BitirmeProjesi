"""Pose-wrist fallback hand crops for frames where MediaPipe lost the hand.

The landmark arrays are never changed.  Only the crop image of a missing frame
is replaced: instead of repeating the last detected crop, a square box is cut
from the *current* frame around the pose hand points (wrist, pinky, index,
thumb).  The box side is the median detected-hand box of the same video, so the
scale matches real crops.  This is a separate, labelled experiment variant.
"""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

# MediaPipe Pose indices: wrist, pinky, index, thumb; elbow.
POSE_HAND_POINTS = {"left": (15, 17, 19, 21), "right": (16, 18, 20, 22)}
POSE_ELBOW = {"left": 13, "right": 14}

SOURCE_DETECTED, SOURCE_POSE_FALLBACK, SOURCE_UNCHANGED = 0, 1, 2


def hand_box_side(hand: np.ndarray, width: int, height: int, padding_ratio: float = 0.2) -> float:
    """Square side (pixels) HandCropper would use for detected 21-point landmarks."""
    xs, ys = hand[:, 0] * width, hand[:, 1] * height
    return float(max(xs.max() - xs.min(), ys.max() - ys.min()) * (1 + 2 * padding_ratio))


def video_box_side(hands: np.ndarray, observed: np.ndarray, pose: np.ndarray, pose_ok: np.ndarray,
                   side: str, width: int, height: int) -> Optional[float]:
    """Median detected box side; else ~0.9 forearm length; None if nothing usable."""
    if observed.any():
        return float(np.median([hand_box_side(hands[i], width, height) for i in np.flatnonzero(observed)]))
    if pose_ok.any():
        scale = np.array([width, height], dtype=np.float32)
        wrist = pose[pose_ok, POSE_HAND_POINTS[side][0], :2] * scale
        elbow = pose[pose_ok, POSE_ELBOW[side], :2] * scale
        forearm = float(np.median(np.linalg.norm(wrist - elbow, axis=1)))
        return 0.9 * forearm if forearm > 1 else None
    return None


def pose_hand_center(pose_frame: np.ndarray, side: str, width: int, height: int) -> np.ndarray:
    points = pose_frame[list(POSE_HAND_POINTS[side]), :2]
    return points.mean(axis=0) * np.array([width, height], dtype=np.float32)


def square_crop(frame: np.ndarray, center: np.ndarray, box_side: float, output_size: tuple[int, int]) -> np.ndarray:
    """Fixed-scale square crop; parts outside the image stay black (no rescaling)."""
    height, width = frame.shape[:2]
    half = max(box_side, 8.0) / 2
    x0, y0 = int(round(center[0] - half)), int(round(center[1] - half))
    side = int(round(2 * half))
    canvas = np.zeros((side, side, 3), dtype=np.uint8)
    sx0, sy0 = max(0, x0), max(0, y0)
    sx1, sy1 = min(width, x0 + side), min(height, y0 + side)
    if sx1 > sx0 and sy1 > sy0:
        canvas[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = frame[sy0:sy1, sx0:sx1]
    return cv2.resize(canvas, (output_size[1], output_size[0]), interpolation=cv2.INTER_LINEAR)


def fallback_crops(frames: list[np.ndarray], crops: np.ndarray, hands: np.ndarray, observed: np.ndarray,
                   pose: np.ndarray, pose_ok: np.ndarray, side: str) -> tuple[np.ndarray, np.ndarray]:
    """Return (new crops, per-frame source code) for one hand of one video."""
    count = len(frames)
    height, width = frames[0].shape[:2]
    new = crops[:count].copy()
    source = np.where(observed[:count], SOURCE_DETECTED, SOURCE_UNCHANGED).astype(np.uint8)
    box = video_box_side(hands[:count], observed[:count], pose[:count], pose_ok[:count], side, width, height)
    if box is None:
        return new, source
    for index in np.flatnonzero(~observed[:count] & pose_ok[:count]):
        center = pose_hand_center(pose[index], side, width, height)
        new[index] = square_crop(frames[index], center, box, crops.shape[1:3])
        source[index] = SOURCE_POSE_FALLBACK
    return new, source
