"""
Abstract base class for frame sampling strategies.

Frame samplers decide which frames from a video to select for model input.
Different strategies (uniform, random, temporal-aware) can significantly
impact training quality.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List


class BaseFrameSampler(ABC):
    """Abstract base class for frame sampling strategies.

    A frame sampler selects a fixed number of frame indices from a video
    of arbitrary length. This decouples the sampling strategy from
    dataset and model implementations.

    Attributes:
        num_frames: Number of frames to sample.
    """

    def __init__(self, num_frames: int) -> None:
        """Initialize the sampler.

        Args:
            num_frames: Number of frames to sample from each video.

        Raises:
            ValueError: If num_frames is not positive.
        """
        if num_frames <= 0:
            raise ValueError(f"num_frames must be positive, got {num_frames}")
        self.num_frames = num_frames

    @abstractmethod
    def sample(self, total_frames: int) -> List[int]:
        """Select frame indices from a video.

        Args:
            total_frames: Total number of frames in the video.

        Returns:
            Sorted list of frame indices (0-indexed) of length self.num_frames.

        Raises:
            ValueError: If total_frames is not positive.
        """
        ...

    def __repr__(self) -> str:
        """Return string representation."""
        return f"{self.__class__.__name__}(num_frames={self.num_frames})"
