"""Pose-wrist fallback hand crops for frames where MediaPipe lost the hand.

The landmark arrays are never changed.  Only the crop image of a missing frame
is replaced: instead of repeating the last detected crop, a square box is cut
from the *current* frame around the pose hand points (wrist, pinky, index,
thumb). The base side is the median detected-hand box of the same video.
An explicit multiplier can enlarge only missing-hand boxes; the output image
size stays fixed. This is a separate, labelled experiment variant.
"""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

# MediaPipe Pose indices: wrist, pinky, index, thumb; elbow.
POSE_HAND_POINTS = {"left": (15, 17, 19, 21), "right": (16, 18, 20, 22)}
POSE_ELBOW = {"left": 13, "right": 14}

SOURCE_DETECTED, SOURCE_POSE_FALLBACK, SOURCE_UNCHANGED = 0, 1, 2
SOURCE_CENTER_REPAIRED = 3

DEFAULT_CENTER_REPAIR = {
    "method": "short_gap_bbox_linear_if_disagree",
    "max_gap": 2,
    "max_wrist_distance": 0.12,
    "max_anchor_step_ratio": 1.0,
    "min_disagreement_ratio": 1.0,
}


def center_repair_policy(options: dict | None) -> dict | None:
    """Validate all policy settings explicitly; never silently ignore a key."""
    if options is None:
        return None
    if set(options) != set(DEFAULT_CENTER_REPAIR) or options["method"] != DEFAULT_CENTER_REPAIR["method"]:
        raise ValueError("Unsupported center repair policy or keys")
    policy = dict(options)
    if type(policy["max_gap"]) is not int or not 1 <= policy["max_gap"] <= 2:
        raise ValueError("Center repair max_gap must be 1 or 2 sampled frames")
    for key in ("max_wrist_distance", "max_anchor_step_ratio", "min_disagreement_ratio"):
        value = float(policy[key])
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"{key} must be finite and positive")
        policy[key] = value
    return policy


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


def crop_bounds(center: np.ndarray, box_side: float) -> tuple[int, int, int]:
    """Integer x/y origin and side actually used by the padded crop."""
    half = max(box_side, 8.0) / 2
    x0, y0 = int(round(center[0] - half)), int(round(center[1] - half))
    side = int(round(2 * half))
    return x0, y0, side


def square_crop(frame: np.ndarray, center: np.ndarray, box_side: float, output_size: tuple[int, int]) -> np.ndarray:
    """Fixed-scale square crop; parts outside the image stay black (no rescaling)."""
    return crop_from_bounds(frame, crop_bounds(center, box_side), output_size)


def crop_from_bounds(frame: np.ndarray, bounds, output_size: tuple[int, int]) -> np.ndarray:
    """Render the exact integer box recorded in the cache metadata."""
    height, width = frame.shape[:2]
    x0, y0, side = map(int, bounds)
    canvas = np.zeros((side, side, 3), dtype=np.uint8)
    sx0, sy0 = max(0, x0), max(0, y0)
    sx1, sy1 = min(width, x0 + side), min(height, y0 + side)
    if sx1 > sx0 and sy1 > sy0:
        canvas[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = frame[sy0:sy1, sx0:sx1]
    return cv2.resize(canvas, (output_size[1], output_size[0]), interpolation=cv2.INTER_LINEAR)


def fallback_geometry(hands: np.ndarray, observed: np.ndarray, pose: np.ndarray, pose_ok: np.ndarray,
                      side: str, width: int, height: int, *, box_scale: float = 1.0,
                      center_repair: dict | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return source, pixel center and integer box; non-fallback metadata is zero.

    Only short interior missing runs bracketed by reliable same-hand detections
    can change center. Stable pose centers stay unchanged. No extrapolation or
    landmark imputation is performed. The whole clip is available at inference.
    """
    if not np.isfinite(box_scale) or box_scale <= 0:
        raise ValueError("box_scale must be finite and positive")
    policy = center_repair_policy(center_repair)
    count = len(observed)
    observed, pose_ok = observed.astype(bool), pose_ok.astype(bool)
    source = np.where(observed, SOURCE_DETECTED, SOURCE_UNCHANGED).astype(np.uint8)
    centers = np.zeros((count, 2), dtype=np.float64)
    boxes = np.zeros((count, 3), dtype=np.int32)
    box = video_box_side(hands, observed, pose, pose_ok, side, width, height)
    if box is None:
        return source, centers, boxes
    base_side = max(box, 8.0)
    for index in np.flatnonzero(~observed & pose_ok):
        centers[index] = pose_hand_center(pose[index], side, width, height)
        source[index] = SOURCE_POSE_FALLBACK

    if policy is not None:
        own = POSE_HAND_POINTS[side][0]
        other = POSE_HAND_POINTS["right" if side == "left" else "left"][0]
        units = np.array([width, height], dtype=np.float64)

        def anchor(index):
            if not (observed[index] and pose_ok[index]):
                return None
            xy = hands[index, :, :2]
            if not (np.isfinite(hands[index]).all() and np.isfinite(pose[index]).all()
                    and ((xy >= 0) & (xy <= 1)).all() and hand_box_side(hands[index], width, height) >= 8):
                return None
            wrist = xy[0]
            own_dist = np.linalg.norm(wrist-pose[index, own, :2])
            other_dist = np.linalg.norm(wrist-pose[index, other, :2])
            if not (own_dist < other_dist and own_dist < policy["max_wrist_distance"]):
                return None
            return (xy.min(axis=0)+xy.max(axis=0))/2 * units

        start = 0
        while start < count:
            if observed[start]:
                start += 1
                continue
            end = start
            while end < count and not observed[end]:
                end += 1
            if start > 0 and end < count and end-start <= policy["max_gap"]:
                left, right = anchor(start-1), anchor(end)
                if (left is not None and right is not None
                        and np.linalg.norm(right-left) <= base_side*policy["max_anchor_step_ratio"]):
                    for index in range(start, end):
                        alpha = (index-start+1)/(end-start+1)
                        center = (1-alpha)*left+alpha*right
                        if (not pose_ok[index] or np.linalg.norm(center-centers[index])
                                >= base_side*policy["min_disagreement_ratio"]):
                            centers[index] = center
                            source[index] = SOURCE_CENTER_REPAIRED
            start = end
    for index in np.flatnonzero(np.isin(source, [SOURCE_POSE_FALLBACK, SOURCE_CENTER_REPAIRED])):
        boxes[index] = crop_bounds(centers[index], base_side*box_scale)
    return source, centers, boxes


def fallback_crops(frames: list[np.ndarray], crops: np.ndarray, hands: np.ndarray, observed: np.ndarray,
                   pose: np.ndarray, pose_ok: np.ndarray, side: str, *, box_scale: float = 1.0,
                   center_repair: dict | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Return current-frame crops and sources (0 detected, 1 pose, 2 unchanged, 3 repaired center)."""
    if not np.isfinite(box_scale) or box_scale <= 0:
        raise ValueError("box_scale must be finite and positive")
    count = len(frames)
    height, width = frames[0].shape[:2]
    source, _, boxes = fallback_geometry(hands[:count], observed[:count], pose[:count], pose_ok[:count],
        side, width, height, box_scale=box_scale, center_repair=center_repair)
    new = crops[:count].copy()
    for index in np.flatnonzero(np.isin(source, [SOURCE_POSE_FALLBACK, SOURCE_CENTER_REPAIRED])):
        new[index] = crop_from_bounds(frames[index], boxes[index], crops.shape[1:3])
    return new, source
