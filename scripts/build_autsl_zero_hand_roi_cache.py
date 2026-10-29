"""Derive a separate AUTSL cache with an upper-body hand retry on empty frames.

Only frames where the source has zero detected hands can change. The pose and
already detected hand arrays stay byte-for-byte equal. Train/val only.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from scripts.pilot_autsl_pose_roi_hand import crop_with_padding  # noqa: E402
from scripts.preprocess_tid import sample_frames  # noqa: E402
from src.core.types import LandmarkResult  # noqa: E402
from src.preprocessing.hand_cropper import HandCropper  # noqa: E402
from src.utils.paths import get_mediapipe_models_dir  # noqa: E402

METHOD = "zero_hand_upperbody_roi"
VERSION = 1


def roi_bounds(pose: np.ndarray, width: int, height: int, fraction: float = .6):
    points = pose[[0, 11, 12, 23, 24], :2]
    if not np.isfinite(points).all():
        return None
    shoulders = (points[1] + points[2]) / 2
    hips = (points[3] + points[4]) / 2
    center = .35 * points[0] + .4 * shoulders + .25 * hips
    side = max(64, round(min(width, height) * fraction))
    return round(center[0] * width - side / 2), round(center[1] * height - side / 2), side


def convert_hand(landmarks, bounds, width: int, height: int) -> np.ndarray | None:
    x0, y0, side = bounds
    hand = np.asarray([[(x0 + p.x * side) / width,
                        (y0 + p.y * side) / height, p.z * side / width]
                       for p in landmarks], dtype=np.float32)
    if (hand.shape != (21, 3) or not np.isfinite(hand).all()
            or not ((hand[:, :2] >= -.05) & (hand[:, :2] <= 1.05)).all()
            or not ((hand[0, :2] >= 0) & (hand[0, :2] <= 1)).all()):
        return None
    return hand


def assign_hands(candidates: list[tuple[str, float, np.ndarray]], pose: np.ndarray):
    """Use detector labels; resolve two equal labels only with a clear arm trace."""
    if len(candidates) == 0:
        return {}, "empty"
    candidates = sorted(candidates, key=lambda item: item[1], reverse=True)[:2]
    if len(candidates) == 1 or candidates[0][0] != candidates[1][0]:
        return {label: hand for label, _, hand in candidates}, "detector_label"
    arms = {side: pose[wrist, :2] + .25 * (pose[wrist, :2] - pose[elbow, :2])
            for side, elbow, wrist in (("left", 13, 15), ("right", 14, 16))}
    if not all(np.isfinite(point).all() for point in arms.values()):
        return {candidates[0][0]: candidates[0][2]}, "duplicate_uncertain"
    first, second = (item[2][0, :2] for item in candidates)
    direct = np.linalg.norm(first - arms["left"]) + np.linalg.norm(second - arms["right"])
    swapped = np.linalg.norm(first - arms["right"]) + np.linalg.norm(second - arms["left"])
    if abs(direct - swapped) < .02:
        return {candidates[0][0]: candidates[0][2]}, "duplicate_uncertain"
    return ({"left": candidates[0][2], "right": candidates[1][2]} if direct < swapped else
            {"right": candidates[0][2], "left": candidates[1][2]}), "arm_pair"


def transform_video(features: dict, frames: list[np.ndarray], detector, cropper: HandCropper):
    count = int(features["num_frames"])
    if len(frames) != count:
        raise ValueError("Source cache and video frame count differ")
    out = {key: value.copy() for key, value in features.items()}
    out["roi_retry_frame"] = np.zeros(count, dtype=np.bool_)
    out["roi_rescued_left"] = np.zeros(count, dtype=np.bool_)
    out["roi_rescued_right"] = np.zeros(count, dtype=np.bool_)
    stats = Counter()
    for t, frame in enumerate(frames):
        if bool(features["left_hand_detected"][t]) or bool(features["right_hand_detected"][t]):
            continue
        stats["zero_hand_frames"] += 1
        if not bool(features["pose_detected"][t]):
            stats["no_pose"] += 1
            continue
        height, width = frame.shape[:2]
        bounds = roi_bounds(features["pose_landmarks"][t], width, height)
        if bounds is None:
            stats["invalid_pose"] += 1
            continue
        crop = crop_with_padding(frame, bounds)
        result = detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB,
                                          data=np.ascontiguousarray(crop)))
        candidates = []
        for landmarks, handedness in zip(result.hand_landmarks, result.handedness):
            hand = convert_hand(landmarks, bounds, width, height)
            if hand is None:
                stats["invalid_hand"] += 1
                continue
            label = handedness[0].category_name.lower()
            if label in ("left", "right"):
                candidates.append((label, float(handedness[0].score), hand))
        assigned, reason = assign_hands(candidates, features["pose_landmarks"][t])
        stats[reason] += 1
        if not assigned:
            continue
        out["roi_retry_frame"][t] = True
        stats["rescued_frames"] += 1
        for side, hand in assigned.items():
            result_for_crop = LandmarkResult(
                pose=None, left_hand=hand if side == "left" else None,
                right_hand=hand if side == "right" else None)
            left, right = cropper.crop_hands(frame, result_for_crop)
            out[f"{side}_hand_landmarks"][t] = hand
            out[f"{side}_hand_crops"][t] = (left if side == "left" else right).image
            out[f"{side}_hand_detected"][t] = True
            out[f"roi_rescued_{side}"][t] = True
            if f"{side}_crop_source" in out:
                out[f"{side}_crop_source"][t] = 4
            for suffix in ("crop_center", "crop_box"):
                key = f"{side}_{suffix}"
                if key in out:
                    out[key][t] = 0
            if f"{side}_crop_center_imputed" in out:
                out[f"{side}_crop_center_imputed"][t] = False
            stats["rescued_hand_slots"] += 1
    return out, dict(stats)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source_cache", type=Path, required=True)
    parser.add_argument("--dest_cache", type=Path, required=True)
    parser.add_argument("--data_dir", type=Path, required=True)
    parser.add_argument("--splits", nargs="+", choices=["train", "val"], default=["train", "val"])
    parser.add_argument("--ids_csv", type=Path, help="Optional train QA CSV with video_id column")
    parser.add_argument("--limit", type=int, help="Optional per-split small pilot limit")
    args = parser.parse_args()
    source, dest = args.source_cache.resolve(), args.dest_cache.resolve()
    if source == dest or source in dest.parents or dest in source.parents:
        parser.error("Source and destination must be separate sibling cache directories")
    if args.ids_csv and args.splits != ["train"]:
        parser.error("--ids_csv currently requires --splits train")
    source_bytes = (source / "manifest.json").read_bytes()
    manifest = json.loads(source_bytes)
    if manifest.get("dataset") != "autsl" or manifest.get("num_frames") != 16:
        parser.error("Source must be a 16-frame AUTSL cache")
    if (manifest.get("derivation") or {}).get("method") != "pose_wrist_fallback_crop":
        parser.error("Source must be the strict pose-wrist crop cache")
    manifest["derivation"] = {
        "method": METHOD, "version": VERSION, "max_gap": None,
        "source_manifest_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "source_derivation": (json.loads(source_bytes).get("derivation")),
        "trigger": "only_when_both_source_hands_missing",
        "roi_fraction": .6,
    }
    dest.mkdir(parents=True, exist_ok=True)
    manifest_path = dest / "manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text(encoding="utf-8")) != manifest:
        raise ValueError("Destination cache manifest differs")
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    chosen = None
    if args.ids_csv:
        with args.ids_csv.open(newline="", encoding="utf-8") as handle:
            chosen = [row["video_id"] for row in csv.DictReader(handle)]
        if len(chosen) != len(set(chosen)):
            parser.error("Duplicate video IDs in --ids_csv")
    cropper = HandCropper(output_size=tuple(manifest["hand_image_size"]), padding_ratio=.2)
    options = mp.tasks.vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(
            get_mediapipe_models_dir() / "hand_landmarker.task")),
        running_mode=mp.tasks.vision.RunningMode.IMAGE, num_hands=2,
        min_hand_detection_confidence=.5)
    summary = {}
    with mp.tasks.vision.HandLandmarker.create_from_options(options) as detector:
        for split in args.splits:
            ids = chosen if chosen is not None else sorted(path.stem for path in (source / split).glob("*.npz"))
            if args.limit is not None:
                ids = ids[:args.limit]
            counts = Counter()
            records = []
            for video_id in tqdm(ids, desc=f"zero-hand ROI {split}"):
                original = source / split / f"{video_id}.npz"
                output = dest / split / f"{video_id}.npz"
                video = args.data_dir / split / f"{video_id}.mp4"
                if not original.is_file() or not video.is_file():
                    raise FileNotFoundError(f"Missing source cache or video: {video_id}")
                if output.exists():
                    counts["skipped_existing"] += 1
                    continue
                with np.load(original, allow_pickle=False) as npz:
                    features = dict(npz)
                frames = sample_frames(str(video), 16)
                derived, stat = transform_video(features, frames, detector, cropper)
                for key in features:
                    if key in ("pose_landmarks", "pose_detected", "label", "num_frames") and not np.array_equal(derived[key], features[key]):
                        raise AssertionError(f"Invariant changed: {video_id} {key}")
                    if key.endswith("hand_detected") and np.any(features[key] & ~derived[key]):
                        raise AssertionError(f"Source hand lost: {video_id} {key}")
                output.parent.mkdir(parents=True, exist_ok=True)
                temporary = output.with_suffix(".npz.part")
                with temporary.open("wb") as handle:
                    np.savez_compressed(handle, **derived)
                temporary.replace(output)
                counts.update(stat)
                counts["videos"] += 1
                records.append({"video_id": video_id, **stat})
            summary[split] = {"totals": dict(counts), "records": records}
    (dest / "roi_retry_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({split: item["totals"] for split, item in summary.items()},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
