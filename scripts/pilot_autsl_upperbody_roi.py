"""Read-only train pilot: retry hand detection in an upper-body crop.

This is a diagnostic of visible active-hand loss. It never edits cache files;
new detections are visual candidates, not automatically accepted landmarks.
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
from scripts.pilot_autsl_pose_roi_hand import crop_with_padding
from scripts.preprocess_tid import sample_frames
from src.utils.paths import get_mediapipe_models_dir

TRAIN_IDS = [
    "signer28_sample472_color", "signer19_sample179_color",
    "signer13_sample1751_color", "signer13_sample1216_color",
    "signer38_sample21_color", "signer22_sample484_color",
    "signer8_sample1058_color",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_dir", type=Path, required=True)
    parser.add_argument("--base_cache", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--roi_fraction", type=float, default=0.6)
    parser.add_argument("--video_ids", nargs="+", default=TRAIN_IDS,
                        help="Selected train video IDs; never reads val for parameter selection")
    args = parser.parse_args()
    if not 0.3 <= args.roi_fraction <= 0.8:
        parser.error("roi_fraction must be 0.3..0.8")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    options = mp.tasks.vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(
            model_asset_path=str(get_mediapipe_models_dir() / "hand_landmarker.task")),
        running_mode=mp.tasks.vision.RunningMode.IMAGE,
        num_hands=2,
        min_hand_detection_confidence=0.5,
    )
    records = []
    with mp.tasks.vision.HandLandmarker.create_from_options(options) as detector:
        for video_id in args.video_ids:
            frames = sample_frames(str(args.data_dir / "train" / f"{video_id}.mp4"), 16)
            with np.load(args.base_cache / "train" / f"{video_id}.npz", allow_pickle=False) as cache:
                pose = cache["pose_landmarks"]
                pose_ok = cache["pose_detected"].astype(bool)
                masks = {s: cache[f"{s}_hand_detected"].astype(bool) for s in ("left", "right")}
                hands = {s: cache[f"{s}_hand_landmarks"] for s in ("left", "right")}
            if len(frames) != len(pose):
                raise ValueError(f"Frame mismatch: {video_id}")
            h, w = frames[0].shape[:2]
            roi_side = max(64, round(min(w, h) * args.roi_fraction))
            rows = []
            thumbs = []
            for t, frame in enumerate(frames):
                if not pose_ok[t]:
                    continue
                # Center between nose and shoulder/hip midline. This avoids a
                # wrong pose-wrist estimate pulling the retry onto a passive hand.
                shoulders = (pose[t, 11, :2] + pose[t, 12, :2]) / 2
                hips = (pose[t, 23, :2] + pose[t, 24, :2]) / 2
                nose = pose[t, 0, :2]
                center = 0.35 * nose + 0.4 * shoulders + 0.25 * hips
                x0 = round(center[0] * w - roi_side / 2)
                y0 = round(center[1] * h - roi_side / 2)
                crop = crop_with_padding(frame, (x0, y0, roi_side))
                result = detector.detect(mp.Image(
                    image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(crop)))
                cached_wrists = [hands[s][t, 0, :2] for s in ("left", "right") if masks[s][t]]
                novel = []
                all_wrists = []
                novel_labels = []
                all_labels = []
                overlay = cv2.cvtColor(frame.copy(), cv2.COLOR_RGB2BGR)
                for s in ("left", "right"):
                    if masks[s][t]:
                        p = hands[s][t, 0, :2]
                        cv2.circle(overlay, (round(p[0] * w), round(p[1] * h)), 9,
                                   (0, 255, 0), 2)
                for landmarks, handedness in zip(result.hand_landmarks, result.handedness):
                    p = landmarks[0]
                    label = handedness[0].category_name.lower()
                    wrist = np.array([(x0 + p.x * roi_side) / w,
                                      (y0 + p.y * roi_side) / h])
                    is_novel = not cached_wrists or min(
                        np.linalg.norm(wrist - baseline) for baseline in cached_wrists) > 0.06
                    all_wrists.append(wrist.round(4).tolist())
                    all_labels.append(label)
                    if is_novel:
                        novel.append(wrist.round(4).tolist())
                        novel_labels.append(label)
                    color = (255, 255, 0) if is_novel else (255, 0, 255)
                    for point in landmarks:
                        px = round(x0 + point.x * roi_side)
                        py = round(y0 + point.y * roi_side)
                        if 0 <= px < w and 0 <= py < h:
                            cv2.circle(overlay, (px, py), 2, color, -1)
                cv2.rectangle(overlay, (max(0, x0), max(0, y0)),
                              (min(w-1, x0+roi_side), min(h-1, y0+roi_side)),
                              (255, 255, 255), 1)
                thumb = cv2.resize(overlay, (300, 300))
                cv2.putText(thumb, f"{t+1}: base={int(masks['left'][t])+int(masks['right'][t])} "
                            f"roi={len(all_wrists)} novel={len(novel)}", (4, 292),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                thumbs.append(thumb)
                rows.append({"frame": t, "base_count": int(masks["left"][t])+int(masks["right"][t]),
                             "roi_count": len(all_wrists), "novel_count": len(novel),
                             "roi_wrists": all_wrists, "novel_wrists": novel,
                             "roi_handedness": all_labels, "novel_handedness": novel_labels})
            sheet = np.zeros((4 * 300, 4 * 300, 3), dtype=np.uint8)
            for i, thumb in enumerate(thumbs):
                sheet[(i//4)*300:(i//4+1)*300, (i%4)*300:(i%4+1)*300] = thumb
            cv2.imwrite(str(args.output_dir / f"{video_id}.png"), sheet)
            summary = {"video_id": video_id, "baseline_hand_slots": sum(r["base_count"] for r in rows),
                       "roi_hand_slots": sum(r["roi_count"] for r in rows),
                       "novel_candidate_slots": sum(r["novel_count"] for r in rows),
                       "frames": rows}
            records.append(summary)
            print(video_id, summary["baseline_hand_slots"], summary["roi_hand_slots"],
                  summary["novel_candidate_slots"], flush=True)
    (args.output_dir / "results.json").write_text(json.dumps(
        {"roi_fraction": args.roi_fraction, "records": records},
        ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
