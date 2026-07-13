"""
Spatial augmentation transforms for individual frames and hand crops.

These transforms operate on single images and are suitable for
both full frames and hand crop regions. Built on top of standard
NumPy/OpenCV operations for maximum portability.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import numpy.typing as npt


class BaseSpatialTransform(ABC):
    """Abstract base class for spatial (per-frame) transforms."""

    @abstractmethod
    def __call__(self, image: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint8]:
        """Apply the spatial transform to an image.

        Args:
            image: (H, W, 3) uint8 image.

        Returns:
            Transformed image.
        """
        ...


class RandomBrightness(BaseSpatialTransform):
    """Randomly adjust image brightness.

    Attributes:
        factor: Maximum brightness change factor (e.g., 0.2 for ±20%).
    """

    def __init__(self, factor: float = 0.2) -> None:
        """Initialize RandomBrightness.

        Args:
            factor: Maximum brightness adjustment factor.
        """
        self.factor = factor

    def __call__(self, image: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint8]:
        """Apply random brightness adjustment.

        Args:
            image: (H, W, 3) uint8 image.

        Returns:
            Brightness-adjusted image.
        """
        delta = np.random.uniform(-self.factor, self.factor)
        adjusted = image.astype(np.float32) * (1.0 + delta)
        return np.clip(adjusted, 0, 255).astype(np.uint8)


class RandomGaussianNoise(BaseSpatialTransform):
    """Add random Gaussian noise to the image.

    Attributes:
        std: Standard deviation of the noise.
    """

    def __init__(self, std: float = 0.01) -> None:
        """Initialize RandomGaussianNoise.

        Args:
            std: Standard deviation relative to [0, 1] pixel range.
        """
        self.std = std

    def __call__(self, image: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint8]:
        """Apply Gaussian noise.

        Args:
            image: (H, W, 3) uint8 image.

        Returns:
            Noisy image.
        """
        noise = np.random.randn(*image.shape).astype(np.float32) * self.std * 255
        noisy = image.astype(np.float32) + noise
        return np.clip(noisy, 0, 255).astype(np.uint8)


class RandomRotation(BaseSpatialTransform):
    """Randomly rotate the image.

    Attributes:
        max_angle: Maximum rotation angle in degrees.
    """

    def __init__(self, max_angle: float = 15.0) -> None:
        """Initialize RandomRotation.

        Args:
            max_angle: Maximum rotation angle in degrees.
        """
        self.max_angle = max_angle

    def __call__(self, image: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint8]:
        """Apply random rotation.

        Uses OpenCV for rotation. Import is deferred to avoid hard
        dependency at module load time.

        Args:
            image: (H, W, 3) uint8 image.

        Returns:
            Rotated image.
        """
        import cv2

        angle = np.random.uniform(-self.max_angle, self.max_angle)
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        return cv2.warpAffine(image, rotation_matrix, (w, h))
