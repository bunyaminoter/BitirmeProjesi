"""Verify a derived zero-hand ROI pilot against its frozen source and video."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from scripts.preprocess_tid import sample_frames
from src.core.types import LandmarkResult
from src.preprocessing.hand_cropper import HandCropper


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source_cache", type=Path, required=True)
    parser.add_argument("--derived_cache", type=Path, required=True)
    parser.add_argument("--data_dir", type=Path, required=True)
    parser.add_argument("--queue_csv", type=Path, required=True)
    parser.add_argument("--roi_results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with args.queue_csv.open(newline="", encoding="utf-8") as handle:
        queue = list(csv.DictReader(handle))
    groups = {r["video_id"]: r["selection_group"] for r in queue}
    roi = {r["video_id"]: r for r in json.loads(
        args.roi_results.read_text(encoding="utf-8"))["records"]}
    if set(groups) != set(roi):
        raise ValueError("Pilot video IDs differ")
    cropper = HandCropper((160, 160), padding_ratio=.2)
    totals = Counter()
    by_group = {}
    for video_id, group in groups.items():
        with np.load(args.source_cache / "train" / f"{video_id}.npz", allow_pickle=False) as source_npz:
            source = dict(source_npz)
        with np.load(args.derived_cache / "train" / f"{video_id}.npz", allow_pickle=False) as derived_npz:
            derived = dict(derived_npz)
        frames = sample_frames(str(args.data_dir / "train" / f"{video_id}.mp4"), 16)
        rescued = derived["roi_retry_frame"].astype(bool)
        expected = np.array([f["base_count"] == 0 and f["roi_count"] > 0
                             for f in roi[video_id]["frames"]])
        if not np.array_equal(rescued, expected):
            raise AssertionError(f"Rescued frames differ from original ROI pilot: {video_id}")
        if np.any(rescued & (source["left_hand_detected"] | source["right_hand_detected"])):
            raise AssertionError(f"Source detected hand overwritten: {video_id}")
        for key in source:
            a, b = source[key], derived[key]
            if key in ("label", "num_frames", "pose_landmarks", "pose_detected"):
                if not np.array_equal(a, b):
                    raise AssertionError(f"Fixed field changed: {video_id} {key}")
            elif a.ndim and a.shape[0] == 16:
                if not np.array_equal(a[~rescued], b[~rescued]):
                    raise AssertionError(f"Non-rescued frame changed: {video_id} {key}")
                if not np.array_equal(a[source["left_hand_detected"] | source["right_hand_detected"]],
                                      b[source["left_hand_detected"] | source["right_hand_detected"]]):
                    raise AssertionError(f"Observed source frame changed: {video_id} {key}")
        if group == "well_detected_control" and rescued.any():
            raise AssertionError(f"Healthy control changed: {video_id}")
        for t in np.flatnonzero(rescued):
            for side in ("left", "right"):
                if not derived[f"roi_rescued_{side}"][t]:
                    continue
                hand = derived[f"{side}_hand_landmarks"][t]
                result = LandmarkResult(pose=None,
                                        left_hand=hand if side == "left" else None,
                                        right_hand=hand if side == "right" else None)
                left, right = cropper.crop_hands(frames[t], result)
                expected_crop = (left if side == "left" else right).image
                if not np.array_equal(expected_crop, derived[f"{side}_hand_crops"][t]):
                    raise AssertionError(f"Recovered crop pixel mismatch: {video_id} {side} {t}")
                totals["pixel_verified_rescued_hand_slots"] += 1
        totals["videos"] += 1
        totals["rescued_frames"] += int(rescued.sum())
        totals["control_unchanged"] += int(group == "well_detected_control")
        by_group.setdefault(group, Counter())["videos"] += 1
        by_group[group]["rescued_frames"] += int(rescued.sum())
    summary = {"totals": dict(totals), "by_group": {key: dict(value) for key, value in by_group.items()}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
