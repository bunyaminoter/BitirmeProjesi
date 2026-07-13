"""
Uniform frame sampling strategy.

Samples frames at evenly spaced intervals across the video duration.
If the video has fewer frames than requested, frames are repeated
with nearest-neighbor interpolation.
"""

from __future__ import annotations

import numpy as np
from typing import List

from src.core.registry import SAMPLER_REGISTRY
from src.data.samplers.base_sampler import BaseFrameSampler


@SAMPLER_REGISTRY.register("uniform")
class UniformFrameSampler(BaseFrameSampler):
    """Uniformly sample frames across the video duration.

    Divides the video into `num_frames` equal segments and picks
    the center frame from each segment. Handles edge cases where
    the video is shorter than the requested number of frames.
    """

    def sample(self, total_frames: int) -> List[int]:
        """Sample uniformly spaced frame indices.

        Args:
            total_frames: Total number of frames in the video.

        Returns:
            Sorted list of frame indices of length self.num_frames.

        Raises:
            ValueError: If total_frames is not positive.
        """
        if total_frames <= 0:
            raise ValueError(f"total_frames must be positive, got {total_frames}")

        if total_frames >= self.num_frames:
            # Evenly spaced indices across the video
            indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)
        else:
            # Video is shorter than requested: repeat frames
            indices = np.arange(total_frames)
            # Pad by repeating the last frame
            pad = np.full(self.num_frames - total_frames, total_frames - 1, dtype=int)
            indices = np.concatenate([indices, pad])

        return sorted(indices.tolist())
