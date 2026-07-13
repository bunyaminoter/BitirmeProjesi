"""
Shared type definitions, protocols, and data containers.

Provides strongly-typed data structures used across the entire pipeline,
from preprocessing through model inference. Using dataclasses and TypedDicts
for zero-overhead type safety.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, TypedDict

import numpy as np
import numpy.typing as npt


# ============================================================================
# NumPy type aliases
# ============================================================================

# A single landmark array: (num_landmarks, 3) for x, y, z
LandmarkArray = npt.NDArray[np.float32]

# A single RGB image: (H, W, 3) in uint8
ImageArray = npt.NDArray[np.uint8]

# A batch of frames: (T, H, W, 3) in uint8
FrameSequence = npt.NDArray[np.uint8]


# ============================================================================
# Data containers
# ============================================================================


@dataclass
class LandmarkResult:
    """Container for landmarks extracted from a single frame.

    All landmark arrays have shape (num_landmarks, 3) with (x, y, z) coords
    normalized to [0, 1] relative to image dimensions.

    Attributes:
        pose: Pose landmarks (33, 3) or None if not detected.
        face: Face landmarks (468, 3) or None if not detected/disabled.
        left_hand: Left hand landmarks (21, 3) or None if not detected.
        right_hand: Right hand landmarks (21, 3) or None if not detected.
        pose_world: World-coordinate pose landmarks (33, 3) or None.
    """

    pose: Optional[LandmarkArray] = None
    face: Optional[LandmarkArray] = None
    left_hand: Optional[LandmarkArray] = None
    right_hand: Optional[LandmarkArray] = None
    pose_world: Optional[LandmarkArray] = None

    @property
    def has_pose(self) -> bool:
        """Whether pose landmarks were detected."""
        return self.pose is not None

    @property
    def has_face(self) -> bool:
        """Whether face landmarks were detected."""
        return self.face is not None

    @property
    def has_left_hand(self) -> bool:
        """Whether left hand landmarks were detected."""
        return self.left_hand is not None

    @property
    def has_right_hand(self) -> bool:
        """Whether right hand landmarks were detected."""
        return self.right_hand is not None


@dataclass
class HandCrop:
    """Container for a cropped hand region from an RGB frame.

    Attributes:
        image: The cropped hand image (H, W, 3) in uint8, or None if missing.
        bbox: Bounding box [x_min, y_min, x_max, y_max] in pixel coords.
        confidence: Detection confidence score.
        handedness: 'left' or 'right'.
    """

    image: Optional[ImageArray] = None
    bbox: Optional[List[float]] = None
    confidence: float = 0.0
    handedness: str = "unknown"

    @property
    def is_valid(self) -> bool:
        """Whether a valid hand crop exists."""
        return self.image is not None and self.image.size > 0


@dataclass
class FrameBatch:
    """A processed batch of frames for a single video sample.

    Contains all data needed by the model for a single sample:
    landmarks, hand crops, and metadata.

    Attributes:
        landmarks: List of LandmarkResult, one per sampled frame.
        left_hand_crops: List of HandCrop for left hand, one per frame.
        right_hand_crops: List of HandCrop for right hand, one per frame.
        label: Integer class label.
        video_id: Unique video identifier string.
        num_frames: Number of frames in this sample.
    """

    landmarks: List[LandmarkResult] = field(default_factory=list)
    left_hand_crops: List[HandCrop] = field(default_factory=list)
    right_hand_crops: List[HandCrop] = field(default_factory=list)
    label: int = -1
    video_id: str = ""
    num_frames: int = 0


class BatchDict(TypedDict, total=False):
    """Type-safe dictionary for collated batch tensors passed to the model.

    All tensors use PyTorch conventions (B = batch, T = time, ...).

    Keys:
        pose_landmarks: (B, T, num_pose_landmarks * 3) float tensor.
        face_landmarks: (B, T, num_face_landmarks * 3) float tensor.
        left_hand_images: (B, T, C, H, W) float tensor.
        right_hand_images: (B, T, C, H, W) float tensor.
        labels: (B,) long tensor of class labels.
        video_ids: List of video ID strings.
        mask: (B, T) bool tensor, True where frames are valid.
    """

    pose_landmarks: Any  # torch.Tensor at runtime
    face_landmarks: Any  # torch.Tensor at runtime
    left_hand_images: Any  # torch.Tensor at runtime
    right_hand_images: Any  # torch.Tensor at runtime
    labels: Any  # torch.Tensor at runtime
    video_ids: List[str]
    mask: Any  # torch.Tensor at runtime


@dataclass
class SampleMetadata:
    """Metadata for a single dataset sample.

    Attributes:
        video_id: Unique video identifier.
        label: Integer class label.
        gloss: Human-readable sign label (e.g., 'hello').
        split: Dataset split ('train', 'val', 'test').
        start_frame: Starting frame index (1-indexed for WLASL).
        end_frame: Ending frame index.
        signer_id: Optional signer identifier for signer-independent splits.
    """

    video_id: str = ""
    label: int = -1
    gloss: str = ""
    split: str = ""
    start_frame: int = 1
    end_frame: int = -1
    signer_id: Optional[str] = None
