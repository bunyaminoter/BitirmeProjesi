"""
Download MediaPipe Tasks API model files required for preprocessing.

Usage:
    python scripts/download_mediapipe_models.py
    python scripts/download_mediapipe_models.py --output_dir models/mediapipe
"""

from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.utils.paths import ensure_dir, get_mediapipe_models_dir

MODEL_URLS = {
    "pose_landmarker_lite.task": (
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
        "pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
    ),
    "pose_landmarker_full.task": (
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
        "pose_landmarker_full/float16/1/pose_landmarker_full.task"
    ),
    "pose_landmarker_heavy.task": (
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
        "pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task"
    ),
    "hand_landmarker.task": (
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
        "hand_landmarker/float16/1/hand_landmarker.task"
    ),
    "face_landmarker.task": (
        "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
        "face_landmarker/float16/1/face_landmarker.task"
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download MediaPipe .task models")
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory (default: models/mediapipe under project root)",
    )
    parser.add_argument(
        "--include_face",
        action="store_true",
        help="Also download the face landmarker model",
    )
    return parser.parse_args()


def download_file(url: str, dest: Path) -> None:
    if dest.exists():
        print(f"  skip (exists): {dest.name}")
        return
    print(f"  downloading: {dest.name}")
    urllib.request.urlretrieve(url, dest)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir) if args.output_dir else get_mediapipe_models_dir()
    ensure_dir(output_dir)

    models = [
        "pose_landmarker_lite.task",
        "pose_landmarker_full.task",
        "hand_landmarker.task",
    ]
    if args.include_face:
        models.append("face_landmarker.task")

    print(f"MediaPipe models -> {output_dir}")
    for name in models:
        download_file(MODEL_URLS[name], output_dir / name)

    print("Done.")


if __name__ == "__main__":
    main()
