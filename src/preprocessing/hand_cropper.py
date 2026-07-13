"""
Hand region cropping from video frames.

Crops hand regions using either MediaPipe hand bounding boxes or
landmark-derived bounding boxes. Handles padding, aspect ratio
preservation, and missing hand detections.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import cv2
import numpy as np
import numpy.typing as npt

from src.core.types import HandCrop, LandmarkResult


class HandCropper:
    """Crop hand regions from RGB frames using landmark information.

    Uses hand landmarks (from MediaPipe) to determine bounding boxes
    and crops the original RGB image. Supports configurable padding,
    output size, and fallback behavior for missing detections.

    Attributes:
        output_size: Target crop size (H, W).
        padding_ratio: Padding around the hand bounding box as a fraction.
        default_crop_value: Fill value for missing hands (0=black, 128=gray).
    """

    def __init__(
        self,
        output_size: Tuple[int, int] = (224, 224),
        padding_ratio: float = 0.2,
        default_crop_value: int = 0,
    ) -> None:
        """Initialize the HandCropper.

        Args:
            output_size: Target output size (H, W) for cropped hands.
            padding_ratio: Extra padding around hand bbox as fraction of size.
            default_crop_value: Pixel fill value for missing hand crops.
        """
        self.output_size = output_size
        self.padding_ratio = padding_ratio
        self.default_crop_value = default_crop_value

    def crop_hands(
        self,
        frame: npt.NDArray[np.uint8],
        landmarks: LandmarkResult,
    ) -> Tuple[HandCrop, HandCrop]:
        """Crop both hands from a frame using landmark information.

        Args:
            frame: (H, W, 3) uint8 RGB image.
            landmarks: Detected landmarks for this frame.

        Returns:
            Tuple of (left_hand_crop, right_hand_crop).
            Crops contain default images if the hand was not detected.
        """
        left_crop = self._crop_single_hand(
            frame, landmarks.left_hand, handedness="left"
        )
        right_crop = self._crop_single_hand(
            frame, landmarks.right_hand, handedness="right"
        )
        return left_crop, right_crop

    def _crop_single_hand(
        self,
        frame: npt.NDArray[np.uint8],
        hand_landmarks: Optional[npt.NDArray[np.float32]],
        handedness: str,
    ) -> HandCrop:
        """Crop a single hand from the frame.

        Args:
            frame: (H, W, 3) uint8 RGB image.
            hand_landmarks: (21, 3) normalized landmarks or None.
            handedness: 'left' or 'right'.

        Returns:
            HandCrop with the cropped image, or empty crop if not detected.
        """
        if hand_landmarks is None:
            return HandCrop(
                image=self._get_default_crop(),
                bbox=None,
                confidence=0.0,
                handedness=handedness,
            )

        h, w = frame.shape[:2]

        # Compute padded bounding box from landmarks
        bbox = self._compute_bbox_from_landmarks(hand_landmarks, h, w)
        x_min, y_min, x_max, y_max = bbox

        # Convert to integer pixel coordinates
        x_min_i = int(round(x_min))
        y_min_i = int(round(y_min))
        x_max_i = int(round(x_max))
        y_max_i = int(round(y_max))

        # Ensure valid crop region
        x_min_i = max(0, x_min_i)
        y_min_i = max(0, y_min_i)
        x_max_i = min(w, x_max_i)
        y_max_i = min(h, y_max_i)

        # Guard against zero-size or invalid crops
        if x_max_i <= x_min_i or y_max_i <= y_min_i:
            return HandCrop(
                image=self._get_default_crop(),
                bbox=bbox,
                confidence=0.5,
                handedness=handedness,
            )

        # Crop the hand region
        crop = frame[y_min_i:y_max_i, x_min_i:x_max_i].copy()

        # Make the crop square by padding the shorter side
        crop_h, crop_w = crop.shape[:2]
        if crop_h != crop_w:
            max_side = max(crop_h, crop_w)
            square_crop = np.full(
                (max_side, max_side, 3),
                self.default_crop_value,
                dtype=np.uint8,
            )
            # Center the crop in the square canvas
            y_offset = (max_side - crop_h) // 2
            x_offset = (max_side - crop_w) // 2
            square_crop[y_offset : y_offset + crop_h, x_offset : x_offset + crop_w] = crop
            crop = square_crop

        # Resize to target output_size
        resized = cv2.resize(
            crop,
            (self.output_size[1], self.output_size[0]),  # (width, height) for cv2
            interpolation=cv2.INTER_LINEAR,
        )

        return HandCrop(
            image=resized,
            bbox=bbox,
            confidence=1.0,
            handedness=handedness,
        )

    def _get_default_crop(self) -> npt.NDArray[np.uint8]:
        """Return a default (empty) hand crop.

        Returns:
            (H, W, 3) uint8 image filled with default_crop_value.
        """
        return np.full(
            (*self.output_size, 3),
            self.default_crop_value,
            dtype=np.uint8,
        )

    def _compute_bbox_from_landmarks(
        self,
        landmarks: npt.NDArray[np.float32],
        frame_height: int,
        frame_width: int,
    ) -> List[float]:
        """Compute padded bounding box from normalized landmarks.

        Args:
            landmarks: (21, 3) normalized hand landmarks.
            frame_height: Original frame height.
            frame_width: Original frame width.

        Returns:
            [x_min, y_min, x_max, y_max] in pixel coordinates.
        """
        # Convert normalized (0-1) to pixel coordinates
        x_coords = landmarks[:, 0] * frame_width
        y_coords = landmarks[:, 1] * frame_height

        x_min, x_max = float(x_coords.min()), float(x_coords.max())
        y_min, y_max = float(y_coords.min()), float(y_coords.max())

        # Add padding
        width = x_max - x_min
        height = y_max - y_min
        pad_x = width * self.padding_ratio
        pad_y = height * self.padding_ratio

        x_min = max(0, x_min - pad_x)
        y_min = max(0, y_min - pad_y)
        x_max = min(frame_width, x_max + pad_x)
        y_max = min(frame_height, y_max + pad_y)

        return [x_min, y_min, x_max, y_max]
