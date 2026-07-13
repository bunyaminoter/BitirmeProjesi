"""
Abstract base class for landmark extraction backends.

Defines the interface that all landmark extractors (MediaPipe, custom, etc.)
must implement. This allows swapping extraction backends without changing
the rest of the pipeline.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

import numpy as np
import numpy.typing as npt

from src.core.types import LandmarkResult


class BaseLandmarkExtractor(ABC):
    """Abstract base class for landmark extraction from video frames.

    Implementations should handle:
        - Pose landmark detection
        - Hand landmark detection (left + right)
        - Face landmark detection (optional)
        - Graceful handling of detection failures

    Attributes:
        enable_face: Whether to extract face landmarks.
    """

    def __init__(self, enable_face: bool = False) -> None:
        """Initialize the extractor.

        Args:
            enable_face: Whether to enable face landmark extraction.
        """
        self.enable_face = enable_face

    @abstractmethod
    def extract(self, frame: npt.NDArray[np.uint8]) -> LandmarkResult:
        """Extract landmarks from a single RGB frame.

        Args:
            frame: (H, W, 3) uint8 RGB image.

        Returns:
            LandmarkResult containing detected landmarks.
            Fields are None for undetected components.
        """
        ...

    @abstractmethod
    def extract_batch(
        self, frames: List[npt.NDArray[np.uint8]]
    ) -> List[LandmarkResult]:
        """Extract landmarks from a batch of frames.

        Default implementation calls extract() per frame.
        Subclasses may override for batched/parallel extraction.

        Args:
            frames: List of (H, W, 3) uint8 RGB images.

        Returns:
            List of LandmarkResult, one per frame.
        """
        ...

    @abstractmethod
    def close(self) -> None:
        """Release any resources held by the extractor.

        Should be called when the extractor is no longer needed.
        """
        ...

    def __enter__(self) -> BaseLandmarkExtractor:
        """Context manager entry."""
        return self

    def __exit__(self, *args) -> None:
        """Context manager exit — releases resources."""
        self.close()
