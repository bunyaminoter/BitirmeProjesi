"""
File I/O helper utilities.

Common file operations for video loading, path management,
and data serialization.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import numpy as np
import numpy.typing as npt


def load_video_frames(
    video_path: str | Path,
    max_frames: Optional[int] = None,
) -> List[npt.NDArray[np.uint8]]:
    """Load frames from a video file using OpenCV.

    Args:
        video_path: Path to the video file.
        max_frames: Maximum number of frames to load (None = all).

    Returns:
        List of (H, W, 3) RGB uint8 frames.

    Raises:
        FileNotFoundError: If the video file does not exist.
        RuntimeError: If the video cannot be opened.
    """
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    import cv2

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    frames: List[npt.NDArray[np.uint8]] = []
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # Convert BGR (OpenCV) to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)

            if max_frames and len(frames) >= max_frames:
                break
    finally:
        cap.release()

    return frames


def get_video_info(video_path: str | Path) -> dict:
    """Get basic video information.

    Args:
        video_path: Path to the video file.

    Returns:
        Dictionary with fps, frame_count, width, height.
    """
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    info = {
        "fps": cap.get(cv2.CAP_PROP_FPS),
        "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    }
    cap.release()
    return info


def ensure_dir(path: str | Path) -> Path:
    """Ensure a directory exists, creating it if necessary.

    Args:
        path: Directory path to ensure.

    Returns:
        The Path object.
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path
