"""Dry-run temporal and two-detector gates against visual role annotations.

This evaluates proposal selection, not landmark accuracy or an NPZ merge.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def temporal_support(rows: list[dict], radius: float = .14) -> list[int]:
    """Return contiguous sampled-frame run length through each proposal.

    The detector's reported side is deliberately ignored here; it is known to
    flip near overlapping hands. This is therefore an optimistic support test.
    """
    by_video = defaultdict(lambda: defaultdict(list))
    for i, row in enumerate(rows):
        by_video[row["video_id"]][int(row["frame_1based"])].append(i)
    xy = [np.array([float(row["wrist_x"]), float(row["wrist_y"])]) for row in rows]
    run = [1] * len(rows)
    for frames in by_video.values():
        for frame in sorted(frames):
            for i in frames[frame]:
                if frame - 1 in frames:
                    neighbours = [j for j in frames[frame - 1]
                                  if np.linalg.norm(xy[i] - xy[j]) <= radius]
                    if neighbours:
                        run[i] = max(run[i], 1 + max(run[j] for j in neighbours))
        for frame in sorted(frames, reverse=True):
            for i in frames[frame]:
                if frame + 1 in frames:
                    neighbours = [j for j in frames[frame + 1]
                                  if np.linalg.norm(xy[i] - xy[j]) <= radius]
                    if neighbours:
                        run[i] = max(run[i], max(run[j] for j in neighbours))
    return run


def outcome(rows: list[dict], selected: list[bool]) -> dict:
    chosen = [row for row, keep in zip(rows, selected) if keep]
    return {
        "candidates": len(chosen),
        "visual_roles": dict(Counter(row["visual_sign_role"] for row in chosen)),
        "control_candidates": sum(row["selection_group"] == "well_detected_control"
                                  for row in chosen),
        "videos": len({row["video_id"] for row in chosen}),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dense_visual", type=Path, required=True)
    parser.add_argument("--dense_raw", type=Path, required=True)
    parser.add_argument("--roi_only_visual", type=Path, required=True)
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    dense = read_csv(args.dense_visual)
    raw = read_csv(args.dense_raw)
    roi = read_csv(args.roi_only_visual)
    groups = {r["video_id"]: r["selection_group"] for r in read_csv(args.queue)}
    if len(dense) != len(raw) or len(roi) != 130:
        raise ValueError("Pilot candidate count changed")
    for row, source in zip(dense, raw):
        if (row["video_id"], row["frame_1based"], row["detection_index"]) != (
                source["video_id"], source["frame_1based"], source["detection_index"]):
            raise ValueError("Dense visual/raw order changed")
        row["selection_group"] = groups[row["video_id"]]
        row["nearest_pose_side"] = source["nearest_pose_side"]
        row["pose_margin"] = source["pose_margin"]
    for row in roi:
        row["selection_group"] = groups[row["video_id"]]
    dense_run = temporal_support(dense)
    roi_run = temporal_support(roi)
    # Match IDs are in the persisted ROI union queue beside the dense raw CSV.
    union_queue = read_csv(args.dense_raw.parent / "roi_candidate_visual_queue.csv")
    roi_matched = {int(r["dense_candidate_number"]) for r in union_queue
                   if r["spatial_match_dense"] == "1"}
    cases = {
        "dense_all": outcome(dense, [True] * len(dense)),
        "dense_temporal3": outcome(dense, [r >= 3 for r in dense_run]),
        "dense_two_detector": outcome(dense, [i + 1 in roi_matched for i in range(len(dense))]),
        "dense_temporal3_and_two_detector": outcome(
            dense, [r >= 3 and i + 1 in roi_matched for i, r in enumerate(dense_run)]),
        "dense_temporal3_pose_agree_missing_side": outcome(dense, [
            r >= 3 and row["detector_side"] == row["nearest_pose_side"]
            and float(row["pose_margin"]) >= .03
            and row[f"base_{row['detector_side']}_seen"] == "0"
            for row, r in zip(dense, dense_run)]),
        "roi_only_all": outcome(roi, [True] * len(roi)),
        "roi_only_temporal3": outcome(roi, [r >= 3 for r in roi_run]),
    }
    payload = {"cases": cases,
               "note": "Temporal adjacency and detector agreement are proxies; no anatomical side validation or cache modification."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
