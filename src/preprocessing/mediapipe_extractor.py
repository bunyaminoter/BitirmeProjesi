"""
MediaPipe Tasks API landmark extractor.

Uses the modern MediaPipe Tasks API (PoseLandmarker + HandLandmarker +
optional FaceLandmarker) instead of the deprecated MediaPipe Holistic.

This provides better accuracy and modular control over which components
to enable.
"""

from __future__ import annotations

import logging
from typing import List, Optional

import numpy as np
import numpy.typing as npt

from src.core.types import LandmarkResult
from src.preprocessing.base_extractor import BaseLandmarkExtractor
from src.utils.paths import get_mediapipe_models_dir

logger = logging.getLogger(__name__)


class MediaPipeLandmarkExtractor(BaseLandmarkExtractor):
    """Extract landmarks using the MediaPipe Tasks API.

    Uses separate PoseLandmarker, HandLandmarker, and (optionally)
    FaceLandmarker for modern, modular landmark extraction.

    Attributes:
        model_complexity: Model complexity (0=lite, 1=full, 2=heavy).
        min_detection_confidence: Minimum detection confidence threshold.
        min_tracking_confidence: Minimum tracking confidence threshold.
    """

    def __init__(
        self,
        enable_face: bool = False,
        model_complexity: int = 1,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        pose_model_path: Optional[str] = None,
        hand_model_path: Optional[str] = None,
        face_model_path: Optional[str] = None,
    ) -> None:
        """Initialize MediaPipe landmark extractors.

        Args:
            enable_face: Whether to enable face landmark extraction.
            model_complexity: Model complexity level (0, 1, or 2).
            min_detection_confidence: Minimum detection confidence.
            min_tracking_confidence: Minimum tracking confidence.
            pose_model_path: Path to custom pose landmarker model.
            hand_model_path: Path to custom hand landmarker model.
            face_model_path: Path to custom face landmarker model.
        """
        super().__init__(enable_face=enable_face)
        self.model_complexity = model_complexity
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence

        # Store model paths for lazy initialization
        self._pose_model_path = pose_model_path
        self._hand_model_path = hand_model_path
        self._face_model_path = face_model_path

        # Landmarker instances (lazily initialized)
        self._pose_landmarker = None
        self._hand_landmarker = None
        self._face_landmarker = None
        self._initialized = False

    def _initialize(self) -> None:
        """Lazily initialize MediaPipe landmarker instances.

        Called on first use to avoid import overhead when not needed.
        Uses MediaPipe Tasks API with PoseLandmarker, HandLandmarker,
        and optionally FaceLandmarker.
        """
        try:
            import mediapipe as mp
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision as mp_vision
        except ImportError:
            logger.warning(
                "mediapipe is not installed. Landmark extraction will "
                "return empty results. Install with: pip install mediapipe"
            )
            self._initialized = True
            return

        BaseOptions = mp_python.BaseOptions

        models_dir = get_mediapipe_models_dir()

        # --- PoseLandmarker ---
        try:
            if self._pose_model_path:
                pose_model_path = self._pose_model_path
            else:
                pose_model_name = (
                    "pose_landmarker_heavy.task"
                    if self.model_complexity == 2
                    else "pose_landmarker_full.task"
                    if self.model_complexity == 1
                    else "pose_landmarker_lite.task"
                )
                pose_model_path = str(models_dir / pose_model_name)

            pose_base = BaseOptions(model_asset_path=pose_model_path)

            pose_options = mp_vision.PoseLandmarkerOptions(
                base_options=pose_base,
                running_mode=mp_vision.RunningMode.IMAGE,
                min_pose_detection_confidence=self.min_detection_confidence,
                min_tracking_confidence=self.min_tracking_confidence,
            )
            self._pose_landmarker = mp_vision.PoseLandmarker.create_from_options(
                pose_options
            )
        except Exception as e:
            logger.warning(
                f"PoseLandmarker initialization failed: {e}. "
                "Pose landmarks will be None."
            )
            self._pose_landmarker = None

        # --- HandLandmarker ---
        try:
            if self._hand_model_path:
                hand_model_path = self._hand_model_path
            else:
                hand_model_path = str(models_dir / "hand_landmarker.task")

            hand_base = BaseOptions(model_asset_path=hand_model_path)

            hand_options = mp_vision.HandLandmarkerOptions(
                base_options=hand_base,
                running_mode=mp_vision.RunningMode.IMAGE,
                num_hands=2,
                min_hand_detection_confidence=self.min_detection_confidence,
                min_tracking_confidence=self.min_tracking_confidence,
            )
            self._hand_landmarker = mp_vision.HandLandmarker.create_from_options(
                hand_options
            )
        except Exception as e:
            logger.warning(
                f"HandLandmarker initialization failed: {e}. "
                "Hand landmarks will be None."
            )
            self._hand_landmarker = None

        # --- FaceLandmarker (optional) ---
        if self.enable_face:
            try:
                if self._face_model_path:
                    face_model_path = self._face_model_path
                else:
                    face_model_path = str(models_dir / "face_landmarker.task")

                face_base = BaseOptions(model_asset_path=face_model_path)

                face_options = mp_vision.FaceLandmarkerOptions(
                    base_options=face_base,
                    running_mode=mp_vision.RunningMode.IMAGE,
                    min_face_detection_confidence=self.min_detection_confidence,
                    min_tracking_confidence=self.min_tracking_confidence,
                )
                self._face_landmarker = mp_vision.FaceLandmarker.create_from_options(
                    face_options
                )
            except Exception as e:
                logger.warning(
                    f"FaceLandmarker initialization failed: {e}. "
                    "Face landmarks will be None."
                )
                self._face_landmarker = None

        self._initialized = True
        logger.info(
            f"MediaPipe initialized — Pose: {self._pose_landmarker is not None}, "
            f"Hand: {self._hand_landmarker is not None}, "
            f"Face: {self._face_landmarker is not None}"
        )

    def _landmarks_to_numpy(
        self, landmark_list, num_landmarks: int
    ) -> npt.NDArray[np.float32]:
        """Convert MediaPipe NormalizedLandmarkList to numpy array.

        Args:
            landmark_list: MediaPipe landmark list object.
            num_landmarks: Expected number of landmarks.

        Returns:
            (num_landmarks, 3) float32 array with (x, y, z) coordinates.
        """
        coords = np.zeros((num_landmarks, 3), dtype=np.float32)
        for i, lm in enumerate(landmark_list):
            if i >= num_landmarks:
                break
            coords[i] = [lm.x, lm.y, lm.z]
        return coords

    def extract(self, frame: npt.NDArray[np.uint8]) -> LandmarkResult:
        """Extract landmarks from a single RGB frame.

        Args:
            frame: (H, W, 3) uint8 RGB image.

        Returns:
            LandmarkResult with detected landmarks.
        """
        if not self._initialized:
            self._initialize()

        pose_landmarks = None
        face_landmarks = None
        left_hand_landmarks = None
        right_hand_landmarks = None

        try:
            import mediapipe as mp

            # Convert numpy frame to MediaPipe Image
            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=frame,
            )
        except ImportError:
            return LandmarkResult(
                pose=None, face=None,
                left_hand=None, right_hand=None,
            )

        # --- Pose detection ---
        if self._pose_landmarker is not None:
            try:
                pose_result = self._pose_landmarker.detect(mp_image)
                if pose_result.pose_landmarks:
                    pose_landmarks = self._landmarks_to_numpy(
                        pose_result.pose_landmarks[0], 33
                    )
            except Exception as e:
                logger.debug(f"Pose detection failed for frame: {e}")

        # --- Hand detection ---
        if self._hand_landmarker is not None:
            try:
                hand_result = self._hand_landmarker.detect(mp_image)
                if hand_result.hand_landmarks:
                    for i, handedness_list in enumerate(hand_result.handedness):
                        hand_label = handedness_list[0].category_name.lower()
                        hand_lm = self._landmarks_to_numpy(
                            hand_result.hand_landmarks[i], 21
                        )
                        # MediaPipe returns mirrored labels (camera view),
                        # "left" from MediaPipe = actual right hand
                        if hand_label == "left":
                            right_hand_landmarks = hand_lm
                        else:
                            left_hand_landmarks = hand_lm
            except Exception as e:
                logger.debug(f"Hand detection failed for frame: {e}")

        # --- Face detection (optional) ---
        if self._face_landmarker is not None:
            try:
                face_result = self._face_landmarker.detect(mp_image)
                if face_result.face_landmarks:
                    face_landmarks = self._landmarks_to_numpy(
                        face_result.face_landmarks[0], 478
                    )
            except Exception as e:
                logger.debug(f"Face detection failed for frame: {e}")

        return LandmarkResult(
            pose=pose_landmarks,
            face=face_landmarks,
            left_hand=left_hand_landmarks,
            right_hand=right_hand_landmarks,
        )

    def extract_batch(
        self, frames: List[npt.NDArray[np.uint8]]
    ) -> List[LandmarkResult]:
        """Extract landmarks from a batch of frames.

        Args:
            frames: List of (H, W, 3) uint8 RGB images.

        Returns:
            List of LandmarkResult, one per frame.
        """
        return [self.extract(frame) for frame in frames]

    def close(self) -> None:
        """Release MediaPipe resources."""
        if self._pose_landmarker is not None:
            self._pose_landmarker.close()
        if self._hand_landmarker is not None:
            self._hand_landmarker.close()
        if self._face_landmarker is not None:
            self._face_landmarker.close()

        self._pose_landmarker = None
        self._hand_landmarker = None
        self._face_landmarker = None
        self._initialized = False
