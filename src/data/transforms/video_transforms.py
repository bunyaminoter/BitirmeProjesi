"""
Video-level augmentation transforms.

These transforms operate on sequences of frames (temporal augmentation).
Each transform follows a consistent interface: takes a list of frames
and returns a transformed list of frames.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

import numpy as np
import numpy.typing as npt


class BaseVideoTransform(ABC):
    """Abstract base class for video-level transforms.

    Video transforms operate on a sequence of frames and may alter
    the temporal structure (frame dropping, speed changes) or apply
    consistent spatial transforms across all frames.
    """

    @abstractmethod
    def __call__(
        self, frames: List[npt.NDArray[np.uint8]]
    ) -> List[npt.NDArray[np.uint8]]:
        """Apply the transform to a list of frames.

        Args:
            frames: List of (H, W, 3) uint8 frames.

        Returns:
            Transformed list of frames.
        """
        ...


class RandomFrameDrop(BaseVideoTransform):
    """Randomly drop frames from the sequence with a given probability.

    Dropped frames are removed entirely, and the remaining frames
    are returned. At least one frame is always kept.

    Attributes:
        drop_prob: Probability of dropping each individual frame.
    """

    def __init__(self, drop_prob: float = 0.1) -> None:
        """Initialize RandomFrameDrop.

        Args:
            drop_prob: Per-frame drop probability.
        """
        self.drop_prob = drop_prob

    def __call__(
        self, frames: List[npt.NDArray[np.uint8]]
    ) -> List[npt.NDArray[np.uint8]]:
        """Apply random frame dropping.

        Args:
            frames: List of frames.

        Returns:
            List with some frames randomly removed.
        """
        if len(frames) <= 1:
            return frames

        mask = np.random.random(len(frames)) > self.drop_prob
        # Ensure at least one frame survives
        if not mask.any():
            mask[0] = True

        return [f for f, keep in zip(frames, mask) if keep]


class RandomTemporalSpeed(BaseVideoTransform):
    """Simulate speed variation by resampling frame indices.

    Attributes:
        speed_range: Tuple (min_speed, max_speed), e.g., (0.8, 1.2).
    """

    def __init__(self, speed_range: tuple[float, float] = (0.8, 1.2)) -> None:
        """Initialize RandomTemporalSpeed.

        Args:
            speed_range: (min_speed, max_speed) multiplier range.
        """
        self.speed_range = speed_range

    def __call__(
        self, frames: List[npt.NDArray[np.uint8]]
    ) -> List[npt.NDArray[np.uint8]]:
        """Apply random temporal speed change.

        Args:
            frames: List of frames.

        Returns:
            Resampled list of frames.
        """
        if len(frames) <= 1:
            return frames

        speed = np.random.uniform(*self.speed_range)
        num_frames = len(frames)
        new_length = max(1, int(num_frames / speed))

        indices = np.linspace(0, num_frames - 1, new_length, dtype=int)
        return [frames[i] for i in indices]
