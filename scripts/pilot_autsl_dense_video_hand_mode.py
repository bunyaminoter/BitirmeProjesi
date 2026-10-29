"""Read-only pilot: HandLandmarker VIDEO mode on every source frame, evaluate 16.

Unlike the earlier sparse VIDEO pilot, tracking receives consecutive video
frames. Cache files are never changed; output is diagnostic JSON and panels.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.utils.paths import get_mediapipe_models_dir  # noqa: E402

DEFAULT_IDS = ["signer38_sample21_color", "signer17_sample52_color",
               "signer13_sample1389_color", "signer28_sample34_color",
               "signer0_sample1350_color", "signer41_sample239_color"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_dir", type=Path, required=True)
    parser.add_argument("--base_cache", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--video_ids", nargs="+", default=DEFAULT_IDS)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    options = mp.tasks.vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(
            model_asset_path=str(get_mediapipe_models_dir() / "hand_landmarker.task")),
        running_mode=mp.tasks.vision.RunningMode.VIDEO, num_hands=2,
        min_hand_detection_confidence=0.5, min_tracking_confidence=0.5)
    records = []
    for video_id in args.video_ids:
        video = args.data_dir / "train" / f"{video_id}.mp4"
        cap = cv2.VideoCapture(str(video))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        if not cap.isOpened() or total <= 0 or fps <= 0:
            raise ValueError(f"Cannot read {video}")
        sample_indices = np.linspace(0, total - 1, min(total, 16), dtype=int).tolist()
        positions = {index: sample_index for sample_index, index in enumerate(sample_indices)}
        with np.load(args.base_cache / "train" / f"{video_id}.npz", allow_pickle=False) as cache:
            observed = {side: cache[f"{side}_hand_detected"].astype(bool)
                        for side in ("left", "right")}
            hands = {side: cache[f"{side}_hand_landmarks"] for side in observed}
        if len(observed["left"]) != len(sample_indices):
            raise ValueError(f"Cache/frame mismatch: {video_id}")
        sampled_results = [None] * len(sample_indices)
        thumbs = [None] * len(sample_indices)
        decoded = 0
        started = time.perf_counter()
        with mp.tasks.vision.HandLandmarker.create_from_options(options) as detector:
            for source_index in range(total):
                ok, bgr = cap.read()
                if not ok:
                    raise ValueError(f"Failed decoding {video} frame {source_index}")
                decoded += 1
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                timestamp = int(round(source_index * 1000 / fps))
                result = detector.detect_for_video(mp.Image(
                    image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb)), timestamp)
                if source_index not in positions:
                    continue
                t = positions[source_index]
                height, width = bgr.shape[:2]
                wrists = [[round(float(hand[0].x), 4), round(float(hand[0].y), 4)]
                          for hand in result.hand_landmarks]
                labels = [handedness[0].category_name.lower()
                          for handedness in result.handedness]
                sampled_results[t] = {"source_index": source_index, "count": len(wrists),
                                      "wrists": wrists, "handedness": labels}
                overlay = bgr.copy()
                for side in observed:
                    if observed[side][t]:
                        p = hands[side][t, 0, :2]
                        cv2.circle(overlay, (round(p[0] * width), round(p[1] * height)),
                                   8, (0, 255, 0), 2)
                for hand, label in zip(result.hand_landmarks, labels):
                    color = (255, 255, 0) if label == "left" else (0, 180, 255)
                    for point in hand:
                        cv2.circle(overlay, (round(point.x * width), round(point.y * height)),
                                   2, color, -1)
                thumb = cv2.resize(overlay, (300, 300))
                base_count = int(observed["left"][t]) + int(observed["right"][t])
                cv2.putText(thumb, f"{t+1}: base={base_count} dense={len(wrists)}",
                            (5, 292), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                thumbs[t] = thumb
        cap.release()
        if any(result is None for result in sampled_results):
            raise ValueError(f"Missing sampled result: {video_id}")
        sheet = np.zeros((1200, 1200, 3), dtype=np.uint8)
        for i, thumb in enumerate(thumbs):
            sheet[i // 4 * 300:(i // 4 + 1) * 300,
                  i % 4 * 300:(i % 4 + 1) * 300] = thumb
        cv2.imwrite(str(args.output_dir / f"{video_id}.png"), sheet)
        base = [int(observed["left"][t]) + int(observed["right"][t])
                for t in range(len(sample_indices))]
        dense = [result["count"] for result in sampled_results]
        record = {"video_id": video_id, "source_frames": decoded, "fps": fps,
                  "seconds": round(time.perf_counter() - started, 3),
                  "base_slots": sum(base), "dense_video_slots": sum(dense),
                  "gain_frames": sum(a > b for a, b in zip(dense, base)),
                  "loss_frames": sum(a < b for a, b in zip(dense, base)),
                  "base_counts": base, "sampled_results": sampled_results}
        records.append(record)
        print(video_id, decoded, sum(base), sum(dense), record["gain_frames"],
              record["loss_frames"], record["seconds"], flush=True)
    (args.output_dir / "results.json").write_text(json.dumps(
        {"mode": "VIDEO_all_source_frames_then_evaluate_original_16", "records": records},
        ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
