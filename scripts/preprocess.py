"""
Batch preprocessing script.

Runs MediaPipe landmark extraction and hand cropping over the entire
dataset and caches results to disk. This should be run once before
training to avoid repeated MediaPipe calls.

Usage:
    python scripts/preprocess.py --config configs/experiment/wlasl100_baseline.yaml --cache_dir cache/features/
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.core.config import load_config
from src.preprocessing.feature_cache import FeatureCache
from src.preprocessing.hand_cropper import HandCropper
from src.preprocessing.mediapipe_extractor import MediaPipeLandmarkExtractor
from src.utils.logging import setup_logging, get_logger


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Preprocess dataset features")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--cache_dir", type=str, default="cache/features/")
    parser.add_argument("--num_workers", type=int, default=4)
    return parser.parse_args()


def sample_frames(
    video_path: str,
    num_frames: int,
    start_frame: int = 1,
    end_frame: int = -1,
) -> list[np.ndarray]:
    """Load and uniformly sample frames from a video file.

    Args:
        video_path: Path to the .mp4 video file.
        num_frames: Number of frames to sample.
        start_frame: 1-indexed start frame.
        end_frame: 1-indexed end frame (-1 = until end).

    Returns:
        List of (H, W, 3) uint8 RGB frames.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        return []

    # Adjust start/end (1-indexed to 0-indexed)
    start_idx = max(0, start_frame - 1)
    end_idx = total_frames if end_frame == -1 else min(end_frame, total_frames)

    available = end_idx - start_idx
    if available <= 0:
        cap.release()
        return []

    # Compute uniform sample indices
    if available <= num_frames:
        indices = list(range(start_idx, end_idx))
    else:
        indices = np.linspace(start_idx, end_idx - 1, num_frames, dtype=int).tolist()

    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            # Convert BGR to RGB
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        else:
            # If read fails, append a black frame as placeholder
            if frames:
                h, w = frames[-1].shape[:2]
            else:
                h, w = 480, 640
            frames.append(np.zeros((h, w, 3), dtype=np.uint8))

    cap.release()
    return frames


def main() -> None:
    """Main preprocessing entry point."""
    args = parse_args()
    config = load_config(args.config)

    logger = setup_logging(level="INFO")
    log = get_logger("preprocess")

    log.info(f"Preprocessing dataset: {config.dataset.name}")
    log.info(f"Cache directory: {args.cache_dir}")
    log.info(f"Num frames per video: {config.dataset.num_frames}")

    # --- Initialize components ---
    cache = FeatureCache(
        cache_dir=args.cache_dir,
        cache_format="npz",
        store_hand_crops=True,
    )

    extractor = MediaPipeLandmarkExtractor(
        enable_face=config.model.landmark_encoder.include_face,
        model_complexity=1,
    )

    cropper = HandCropper(
        output_size=tuple(config.model.hand_encoder.input_size),
        padding_ratio=0.2,
    )

    # --- Load annotations ---
    annotation_path = Path(config.dataset.annotation_file)
    if not annotation_path.exists():
        log.error(f"Annotation file not found: {annotation_path}")
        return

    with open(annotation_path, "r", encoding="utf-8") as f:
        annotations = json.load(f)

    # Filter by num_classes
    filtered = {}
    for video_id, info in annotations.items():
        class_idx = info.get("action", [0])[0]
        if class_idx < config.dataset.num_classes:
            filtered[video_id] = info

    total_videos = len(filtered)
    log.info(f"Total videos to process: {total_videos}")

    # --- Process each video ---
    processed = 0
    skipped = 0
    errors = 0

    try:
        for i, (video_id, info) in enumerate(filtered.items()):
            # Skip already cached
            if cache.exists(video_id):
                skipped += 1
                continue

            video_path = Path(config.dataset.video_dir) / f"{video_id}.mp4"

            if not video_path.exists():
                skipped += 1
                continue

            # Get start/end frames
            action = info.get("action", [0, 1, -1])
            start_frame = action[1] if len(action) > 1 else 1
            end_frame = action[2] if len(action) > 2 else -1

            # Load and sample frames
            frames = sample_frames(
                str(video_path),
                config.dataset.num_frames,
                start_frame=start_frame,
                end_frame=end_frame,
            )

            if not frames:
                log.warning(f"Failed to read frames from existing video: {video_path}")
                errors += 1
                continue

            # Extract landmarks and crop hands for each frame
            all_pose = []
            all_left_hand = []
            all_right_hand = []
            all_left_crop = []
            all_right_crop = []

            for frame in frames:
                # Extract landmarks
                result = extractor.extract(frame)

                # Store pose landmarks (33, 3) or zeros
                if result.pose is not None:
                    all_pose.append(result.pose)
                else:
                    all_pose.append(np.zeros((33, 3), dtype=np.float32))

                # Store hand landmarks (21, 3) or zeros
                if result.left_hand is not None:
                    all_left_hand.append(result.left_hand)
                else:
                    all_left_hand.append(np.zeros((21, 3), dtype=np.float32))

                if result.right_hand is not None:
                    all_right_hand.append(result.right_hand)
                else:
                    all_right_hand.append(np.zeros((21, 3), dtype=np.float32))

                # Crop hands
                left_crop, right_crop = cropper.crop_hands(frame, result)
                all_left_crop.append(left_crop.image)
                all_right_crop.append(right_crop.image)

            # Save to cache as .npz
            features = {
                "pose_landmarks": np.stack(all_pose, axis=0),        # (T, 33, 3)
                "left_hand_landmarks": np.stack(all_left_hand, axis=0),  # (T, 21, 3)
                "right_hand_landmarks": np.stack(all_right_hand, axis=0),  # (T, 21, 3)
                "left_hand_crops": np.stack(all_left_crop, axis=0),    # (T, H, W, 3)
                "right_hand_crops": np.stack(all_right_crop, axis=0),  # (T, H, W, 3)
                "label": np.array(action[0], dtype=np.int64),
                "num_frames": np.array(len(frames), dtype=np.int64),
            }

            cache.save(video_id, features)
            processed += 1

            if (processed + skipped) % 50 == 0 or (i + 1) == total_videos:
                log.info(
                    f"Progress: {i + 1}/{total_videos} "
                    f"(processed={processed}, skipped={skipped}, errors={errors})"
                )

    except KeyboardInterrupt:
        log.info("Preprocessing interrupted by user.")
    finally:
        extractor.close()

    log.info(
        f"Preprocessing complete! "
        f"Processed: {processed}, Skipped: {skipped}, Errors: {errors}, "
        f"Total cached: {cache.num_cached}"
    )


if __name__ == "__main__":
    main()
