"""Conservative interpolation for short, interior gaps in hand landmarks."""

from __future__ import annotations

import numpy as np


def repair_hand_gaps(
    hand: np.ndarray,
    observed: np.ndarray,
    pose: np.ndarray,
    pose_observed: np.ndarray,
    *,
    wrist_index: int,
    other_wrist_index: int,
    max_gap: int = 2,
    max_wrist_distance: float = 0.12,
) -> tuple[np.ndarray, np.ndarray]:
    """Fill only short gaps bracketed by reliable observations of the same hand.

    Returns repaired landmarks and a separate imputation mask. The original
    observed mask remains untouched so downstream QA can distinguish estimates.
    Long, leading, trailing, and pose-missing gaps are left as zeros.
    """
    if max_gap < 1:
        raise ValueError("max_gap must be positive")
    if hand.ndim != 3 or hand.shape[1:] != (21, 3):
        raise ValueError("hand must have shape (T, 21, 3)")
    count = hand.shape[0]
    if observed.shape != (count,) or pose_observed.shape != (count,) or pose.shape != (count, 33, 3):
        raise ValueError("hand and pose masks must match the frame count")
    repaired = hand.copy()
    imputed = np.zeros(count, dtype=np.bool_)

    def reliable(frame: int) -> bool:
        if not observed[frame] or not pose_observed[frame]:
            return False
        wrist = hand[frame, 0, :2]
        own = np.linalg.norm(wrist - pose[frame, wrist_index, :2])
        other = np.linalg.norm(wrist - pose[frame, other_wrist_index, :2])
        return bool(np.isfinite(hand[frame]).all() and own < other and own < max_wrist_distance)

    start = 0
    while start < count:
        if observed[start]:
            start += 1
            continue
        end = start
        while end < count and not observed[end]:
            end += 1
        if (start > 0 and end < count and end - start <= max_gap
                and pose_observed[start:end].all()
                and reliable(start - 1) and reliable(end)):
            for frame in range(start, end):
                alpha = (frame - start + 1) / (end - start + 1)
                repaired[frame] = ((1 - alpha) * hand[start - 1] + alpha * hand[end]).astype(hand.dtype)
                imputed[frame] = True
        start = end
    return repaired, imputed
