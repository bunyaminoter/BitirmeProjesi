"""Diagnostic only: retry missing hands inside a pose-wrist ROI on selected videos.

This never modifies a cache. Candidate detections need visual review before
they can count as recovered hand landmarks.
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


def roi_bounds(cx: float, cy: float, side: int) -> tuple[int, int, int]:
    x0 = int(round(cx - side / 2))
    y0 = int(round(cy - side / 2))
    return x0, y0, side


def crop_with_padding(frame: np.ndarray, bounds: tuple[int, int, int]) -> np.ndarray:
    height, width = frame.shape[:2]
    x0, y0, side = bounds
    out = np.zeros((side, side, 3), dtype=np.uint8)
    sx0, sy0 = max(0, x0), max(0, y0)
    sx1, sy1 = min(width, x0 + side), min(height, y0 + side)
    if sx1 > sx0 and sy1 > sy0:
        out[sy0-y0:sy1-y0, sx0-x0:sx1-x0] = frame[sy0:sy1, sx0:sx1]
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True,
                        help="JSON list of {'split':'train'|'val','video_id':'..._color'}")
    parser.add_argument("--data_dir", type=Path, required=True)
    parser.add_argument("--base_cache", type=Path, required=True)
    parser.add_argument("--roi_fraction", type=float, default=0.4,
                        help="Square ROI side as fraction of smaller video dimension")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 0.1 <= args.roi_fraction <= 0.8:
        parser.error("--roi_fraction must be between 0.1 and 0.8")
    selection = json.loads(args.selection.read_text(encoding="utf-8"))
    if not isinstance(selection, list) or not selection or any(
        x.get("split") not in ("train", "val") or not isinstance(x.get("video_id"), str)
        or not x["video_id"].endswith("_color") for x in selection):
        parser.error("Invalid selection")
    options = mp.tasks.vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(
            model_asset_path=str(get_mediapipe_models_dir() / "hand_landmarker.task")),
        running_mode=mp.tasks.vision.RunningMode.IMAGE,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    records = []
    with mp.tasks.vision.HandLandmarker.create_from_options(options) as detector:
        for item in selection:
            split, video_id = item["split"], item["video_id"]
            video = args.data_dir / split / f"{video_id}.mp4"
            cache = args.base_cache / split / f"{video_id}.npz"
            frames = sample_frames(str(video), 16)
            with np.load(cache, allow_pickle=False) as data:
                pose = data["pose_landmarks"]
                pose_ok = data["pose_detected"].astype(bool)
                observed = {s: data[f"{s}_hand_detected"].astype(bool) for s in ("left", "right")}
                cached_hands = {s: data[f"{s}_hand_landmarks"] for s in ("left", "right")}
            if len(frames) != len(pose):
                raise ValueError(f"Frame/cache mismatch for {video_id}")
            height, width = frames[0].shape[:2]
            side = max(64, int(round(min(width, height) * args.roi_fraction)))
            candidates = []
            for index, frame in enumerate(frames):
                if not pose_ok[index]:
                    continue
                for hand_side, wrist_index, other_index in (("left", 15, 16), ("right", 16, 15)):
                    if observed[hand_side][index]:
                        continue
                    wrist = pose[index, wrist_index, :2]
                    other = pose[index, other_index, :2]
                    if not np.isfinite(wrist).all() or not (0 <= wrist[0] <= 1 and 0 <= wrist[1] <= 1):
                        continue
                    bounds = roi_bounds(wrist[0] * width, wrist[1] * height, side)
                    crop = crop_with_padding(frame, bounds)
                    result = detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB,
                                                      data=np.ascontiguousarray(crop)))
                    ranked = []
                    for hand in result.hand_landmarks:
                        point = hand[0]
                        x = (bounds[0] + point.x * side) / width
                        y = (bounds[1] + point.y * side) / height
                        own_distance = float(np.linalg.norm(np.array([x, y]) - wrist))
                        other_distance = float(np.linalg.norm(np.array([x, y]) - other))
                        ranked.append((own_distance, other_distance, x, y))
                    ranked.sort(key=lambda x: x[0])
                    best = ranked[0] if ranked else None
                    raw_candidate = best is not None and best[0] <= 0.12
                    opposite_side = "right" if hand_side == "left" else "left"
                    duplicate = bool(raw_candidate and observed[opposite_side][index] and
                                     np.linalg.norm(np.array(best[2:4]) -
                                                    cached_hands[opposite_side][index, 0, :2]) < 0.06)
                    ambiguous = bool(raw_candidate and best[1] <= best[0] + 0.03)
                    # Conservative: do not insert a detection that might be the other hand.
                    candidate = raw_candidate and not duplicate and not ambiguous
                    candidates.append({"frame_index": index, "side": hand_side,
                                       "raw_detection_count": len(ranked),
                                       "raw_candidate": bool(raw_candidate),
                                       "candidate": bool(candidate),
                                       "own_wrist_distance": round(best[0], 4) if best else None,
                                       "other_wrist_distance": round(best[1], 4) if best else None,
                                       "ambiguous": ambiguous,
                                       "duplicate_other_detected_hand": duplicate,
                                       "candidate_wrist": [round(best[2], 4), round(best[3], 4)] if best else None})
            records.append({"split": split, "video_id": video_id,
                            "baseline_detected": {s: int(observed[s].sum()) for s in observed},
                            "baseline_missing": {s: int((~observed[s]).sum()) for s in observed},
                            "roi_raw_candidate_frames": sum(x["raw_candidate"] for x in candidates),
                            "roi_candidate_frames": sum(x["candidate"] for x in candidates),
                            "roi_ambiguous_frames": sum(x["ambiguous"] for x in candidates),
                            "roi_duplicate_frames": sum(x["duplicate_other_detected_hand"] for x in candidates),
                            "candidates": candidates})
            print(split, video_id, records[-1]["baseline_detected"],
                  "ROI raw/accepted", records[-1]["roi_raw_candidate_frames"],
                  records[-1]["roi_candidate_frames"], "ambiguous",
                  records[-1]["roi_ambiguous_frames"], "duplicate",
                  records[-1]["roi_duplicate_frames"], flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"roi_fraction": args.roi_fraction,
                                       "model_path": str(get_mediapipe_models_dir() / "hand_landmarker.task"),
                                       "records": records}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
