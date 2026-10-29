"""Summarize cached detection quality for an official val prediction file.

This is a diagnostic of detection/fallback frequency, not ground truth for
landmark accuracy or a proof of why the classifier made an error.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def longest_false_run(mask: np.ndarray) -> int:
    longest = current = 0
    for found in mask:
        current = 0 if found else current + 1
        longest = max(longest, current)
    return longest


def audit(predictions_path: Path, cache_dir: Path) -> dict:
    payload = json.loads(predictions_path.read_text(encoding="utf-8"))
    if payload.get("split") != "val":
        raise ValueError("Only official val predictions may be audited")
    records = payload.get("predictions", [])
    ids = [item["video_id"] for item in records]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate prediction video IDs")

    by_signer = defaultdict(lambda: dict(videos=0, correct=0, frames=0,
        detected_hands=0, fallback_crops=0, repaired_crops=0,
        pose_frames=0, max_missing_run=0))
    details = []
    for record in records:
        video_id = record["video_id"]
        cache_path = cache_dir / "val" / f"{video_id}.npz"
        with np.load(cache_path, allow_pickle=False) as cache:
            left = cache["left_hand_detected"].astype(bool)
            right = cache["right_hand_detected"].astype(bool)
            pose = cache["pose_detected"].astype(bool)
            sources = np.concatenate([cache["left_crop_source"], cache["right_crop_source"]])
            cached_label = int(cache["label"])
        if len(left) != len(right) or len(left) != len(pose):
            raise ValueError(f"Frame length mismatch: {video_id}")
        if cached_label != int(record["true_label"]):
            raise ValueError(f"Cache/prediction label mismatch: {video_id}")
        detected = int(left.sum() + right.sum())
        fallback = int(np.count_nonzero(np.isin(sources, [1, 3])))
        repaired = int(np.count_nonzero(sources == 3))
        max_gap = max(longest_false_run(left), longest_false_run(right))
        correct = int(record["true_label"]) == int(record["pred_label"])
        signer = video_id.split("_", 1)[0]
        summary = by_signer[signer]
        summary["videos"] += 1
        summary["correct"] += int(correct)
        summary["frames"] += len(left)
        summary["detected_hands"] += detected
        summary["fallback_crops"] += fallback
        summary["repaired_crops"] += repaired
        summary["pose_frames"] += int(pose.sum())
        summary["max_missing_run"] = max(summary["max_missing_run"], max_gap)
        details.append(dict(video_id=video_id, signer=signer,
            true_label=int(record["true_label"]), pred_label=int(record["pred_label"]),
            top1_confidence=float(record["top1_confidence"]), correct=correct,
            frames=len(left), detected_hands=detected, fallback_crops=fallback,
            repaired_crops=repaired, pose_frames=int(pose.sum()),
            max_missing_run=max_gap))

    return dict(split="val", config=payload.get("config"),
        checkpoint_sha256=payload.get("checkpoint_sha256"), n=len(details),
        by_signer=dict(sorted(by_signer.items())),
        errors=[row for row in details if not row["correct"]],
        videos=details)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--cache_dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.predictions, args.cache_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(dict(n=result["n"], errors=len(result["errors"]),
        by_signer=result["by_signer"]), ensure_ascii=False))


if __name__ == "__main__":
    main()
