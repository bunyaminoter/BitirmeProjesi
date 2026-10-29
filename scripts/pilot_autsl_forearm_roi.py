"""Read-only AUTSL train pilot of the forearm/shoulder hand ROI from De Coster et al.

Only retry a side when the baseline cache lacks that side. Candidate counts are
diagnostic; they are never written back to the production cache.
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
from scripts.pilot_autsl_pose_roi_hand import crop_with_padding  # noqa: E402
from scripts.preprocess_tid import sample_frames  # noqa: E402
from src.utils.paths import get_mediapipe_models_dir  # noqa: E402

DEFAULT_IDS = [
    "signer28_sample472_color", "signer19_sample179_color",
    "signer13_sample1751_color", "signer13_sample1216_color",
    "signer38_sample21_color", "signer22_sample484_color",
    "signer8_sample1058_color",
]
POSE = {"left": (13, 15), "right": (14, 16)}  # elbow, wrist


def forearm_bounds(pose: np.ndarray, side: str, width: int, height: int,
                   scale: float = 1.2) -> tuple[int, int, int] | None:
    """Paper equations 6/7 in pixel coordinates; reject nonsensical pose ROIs."""
    xy = pose[:, :2] * np.array([width, height], dtype=np.float32)
    elbow, wrist = (xy[index] for index in POSE[side])
    shoulder_distance = float(np.linalg.norm(xy[11] - xy[12]))
    if (not np.isfinite(xy[[11, 12, *POSE[side]]]).all()
            or not 0.05 * width <= shoulder_distance <= 0.8 * width
            or not (0 <= wrist[0] < width and 0 <= wrist[1] < height)):
        return None
    center = wrist + 0.15 * (wrist - elbow)
    box_side = int(round(scale * shoulder_distance))
    if box_side < 32 or box_side > min(width, height):
        return None
    return round(float(center[0] - box_side / 2)), round(float(center[1] - box_side / 2)), box_side


def choose_candidate(landmarks: list, bounds: tuple[int, int, int], pose: np.ndarray,
                     side: str, cached_opposite: np.ndarray | None,
                     width: int, height: int) -> dict:
    x0, y0, box_side = bounds
    own = pose[POSE[side][1], :2]
    opposite = pose[POSE["right" if side == "left" else "left"][1], :2]
    ranked = []
    for hand in landmarks:
        p = hand[0]
        wrist = np.array([(x0 + p.x * box_side) / width,
                          (y0 + p.y * box_side) / height])
        own_distance = float(np.linalg.norm(wrist - own))
        other_distance = float(np.linalg.norm(wrist - opposite))
        duplicate = cached_opposite is not None and np.linalg.norm(wrist - cached_opposite) < 0.06
        accepted = (own_distance < 0.12 and own_distance + 0.03 < other_distance
                    and not duplicate and np.isfinite(wrist).all()
                    and ((wrist >= 0) & (wrist <= 1)).all())
        ranked.append({"wrist": wrist.round(4).tolist(), "own_distance": round(own_distance, 4),
                       "other_distance": round(other_distance, 4), "duplicate": bool(duplicate),
                       "accepted": bool(accepted)})
    ranked.sort(key=lambda item: item["own_distance"])
    return {"candidates": ranked, "best": ranked[0] if ranked else None}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_dir", type=Path, required=True)
    parser.add_argument("--base_cache", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--video_ids", nargs="+", default=DEFAULT_IDS)
    parser.add_argument("--scale", type=float, default=1.2)
    args = parser.parse_args()
    if not 0.7 <= args.scale <= 2.0:
        parser.error("scale must be 0.7..2.0")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    options = mp.tasks.vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(get_mediapipe_models_dir() / "hand_landmarker.task")),
        running_mode=mp.tasks.vision.RunningMode.IMAGE, num_hands=2,
        min_hand_detection_confidence=0.5, min_tracking_confidence=0.5)
    records = []
    with mp.tasks.vision.HandLandmarker.create_from_options(options) as detector:
        for video_id in args.video_ids:
            frames = sample_frames(str(args.data_dir / "train" / f"{video_id}.mp4"), 16)
            with np.load(args.base_cache / "train" / f"{video_id}.npz", allow_pickle=False) as cache:
                pose = cache["pose_landmarks"]
                pose_ok = cache["pose_detected"].astype(bool)
                observed = {side: cache[f"{side}_hand_detected"].astype(bool) for side in POSE}
                hands = {side: cache[f"{side}_hand_landmarks"] for side in POSE}
            if len(frames) != len(pose):
                raise ValueError(f"Frame mismatch for {video_id}")
            height, width = frames[0].shape[:2]
            attempts = []
            panels = []
            started = time.perf_counter()
            for t, frame in enumerate(frames):
                overlay = cv2.cvtColor(frame.copy(), cv2.COLOR_RGB2BGR)
                for side in POSE:
                    if observed[side][t] or not pose_ok[t]:
                        continue
                    bounds = forearm_bounds(pose[t], side, width, height, args.scale)
                    if bounds is None:
                        attempts.append({"frame": t, "side": side, "status": "invalid_pose_roi"})
                        continue
                    crop = crop_with_padding(frame, bounds)
                    result = detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB,
                                                       data=np.ascontiguousarray(crop)))
                    other = "right" if side == "left" else "left"
                    cached_opposite = hands[other][t, 0, :2] if observed[other][t] else None
                    candidate = choose_candidate(result.hand_landmarks, bounds, pose[t], side,
                                                 cached_opposite, width, height)
                    best = candidate["best"]
                    attempts.append({"frame": t, "side": side, "status": "detected" if best else "empty",
                                     "bounds": bounds, **candidate})
                    x0, y0, box_side = bounds
                    cv2.rectangle(overlay, (x0, y0), (x0 + box_side, y0 + box_side),
                                  (255, 180, 0) if side == "left" else (0, 180, 255), 1)
                    if best:
                        wx, wy = best["wrist"]
                        cv2.circle(overlay, (round(wx * width), round(wy * height)), 7,
                                   (0, 255, 0) if best["accepted"] else (0, 0, 255), 2)
                thumb = cv2.resize(overlay, (300, 300))
                cv2.putText(thumb, f"{t+1}: base {int(observed['left'][t])}+{int(observed['right'][t])}",
                            (5, 292), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                panels.append(thumb)
            sheet = np.zeros((1200, 1200, 3), dtype=np.uint8)
            for i, thumb in enumerate(panels):
                sheet[i // 4 * 300:(i // 4 + 1) * 300,
                      i % 4 * 300:(i % 4 + 1) * 300] = thumb
            cv2.imwrite(str(args.output_dir / f"{video_id}.png"), sheet)
            record = {"video_id": video_id,
                      "baseline_detected": {side: int(observed[side].sum()) for side in POSE},
                      "attempts": len(attempts),
                      "raw_detection_attempts": sum(a.get("best") is not None for a in attempts),
                      "accepted_attempts": sum(bool(a.get("best") and a["best"]["accepted"]) for a in attempts),
                      "seconds": round(time.perf_counter() - started, 3),
                      "frames": attempts}
            records.append(record)
            print(video_id, record["baseline_detected"], record["attempts"],
                  record["raw_detection_attempts"], record["accepted_attempts"],
                  record["seconds"], flush=True)
    (args.output_dir / "results.json").write_text(json.dumps(
        {"method": "De Coster et al. 2021 equations 6/7", "scale": args.scale,
         "records": records}, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
