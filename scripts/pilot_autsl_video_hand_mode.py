"""Diagnostic comparison of MediaPipe IMAGE cache versus VIDEO mode on selected clips.

Uses the same 16 decoded frames and hand model; does not modify caches.
Extra detections may be false or wrongly associated and require visual review.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from scripts.preprocess_tid import sample_frames  # noqa: E402
from src.utils.paths import get_mediapipe_models_dir  # noqa: E402


def timestamps(path: Path, count: int) -> list[int]:
    cap = cv2.VideoCapture(str(path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    cap.release()
    if total < count or fps <= 0:
        raise ValueError(f"Cannot construct source timestamps for {path}")
    indices = np.linspace(0, total - 1, count, dtype=int)
    times = [int(round(1000 * index / fps)) for index in indices]
    if len(times) != len(set(times)):
        raise ValueError(f"Duplicate timestamps for {path}")
    return times


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--data_dir", type=Path, required=True)
    parser.add_argument("--base_cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    selection = json.loads(args.selection.read_text(encoding="utf-8"))
    records = []
    for item in selection:
        split, video_id = item["split"], item["video_id"]
        path = args.data_dir / split / f"{video_id}.mp4"
        frames = sample_frames(str(path), 16)
        with np.load(args.base_cache / split / f"{video_id}.npz", allow_pickle=False) as data:
            baseline = (data["left_hand_detected"].astype(int) +
                        data["right_hand_detected"].astype(int)).tolist()
        if len(frames) != len(baseline):
            raise ValueError(f"Frame/cache length mismatch: {video_id}")
        options = mp.tasks.vision.HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(
                model_asset_path=str(get_mediapipe_models_dir() / "hand_landmarker.task")),
            running_mode=mp.tasks.vision.RunningMode.VIDEO, num_hands=2,
            min_hand_detection_confidence=0.5, min_tracking_confidence=0.5)
        results = []
        with mp.tasks.vision.HandLandmarker.create_from_options(options) as detector:
            for frame, timestamp in zip(frames, timestamps(path, len(frames))):
                result = detector.detect_for_video(
                    mp.Image(image_format=mp.ImageFormat.SRGB,
                             data=np.ascontiguousarray(frame)), timestamp)
                results.append({"count": len(result.hand_landmarks),
                                "wrists": [[round(float(hand[0].x), 4), round(float(hand[0].y), 4)]
                                           for hand in result.hand_landmarks]})
        counts = [x["count"] for x in results]
        records.append({"split": split, "video_id": video_id,
                        "image_mode_hand_frames": sum(baseline),
                        "video_mode_hand_frames": sum(counts),
                        "gain_frames": sum(v > b for v, b in zip(counts, baseline)),
                        "loss_frames": sum(v < b for v, b in zip(counts, baseline)),
                        "baseline_counts": baseline, "video_results": results})
        print(split, video_id, "IMAGE", sum(baseline), "VIDEO", sum(counts),
              "gain/loss frames", records[-1]["gain_frames"], records[-1]["loss_frames"], flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"records": records}, ensure_ascii=False, indent=2) + "\n",
                           encoding="utf-8")


if __name__ == "__main__":
    main()
