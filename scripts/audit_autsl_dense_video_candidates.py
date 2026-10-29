"""Audit dense VIDEO hand proposals against the frozen 16-frame cache.

Geometric flags are triage hints, not anatomical or active-hand ground truth.
This script never changes the cache. Visual labels are intentionally left empty.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--base_cache", type=Path, required=True)
    parser.add_argument("--dense_results", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with args.queue.open(newline="", encoding="utf-8") as handle:
        queue = list(csv.DictReader(handle))
    dense = {item["video_id"]: item for item in json.loads(
        args.dense_results.read_text(encoding="utf-8"))["records"]}
    if set(dense) != {item["video_id"] for item in queue}:
        raise ValueError("Queue/dense video IDs differ")

    rows = []
    frame_rows = []
    by_group: dict[str, Counter] = defaultdict(Counter)
    by_video: dict[str, Counter] = defaultdict(Counter)
    for item in queue:
        video_id, group = item["video_id"], item["selection_group"]
        with np.load(args.base_cache / "train" / f"{video_id}.npz", allow_pickle=False) as cache:
            pose = cache["pose_landmarks"]
            base_seen = {side: cache[f"{side}_hand_detected"].astype(bool)
                         for side in ("left", "right")}
            base_wrist = {side: cache[f"{side}_hand_landmarks"][:, 0, :2]
                          for side in ("left", "right")}
        results = dense[video_id]["sampled_results"]
        if len(results) != len(pose):
            raise ValueError(f"Frame count differs: {video_id}")
        for t, result in enumerate(results):
            wrists = result["wrists"]
            labels = result["handedness"]
            if len(wrists) != len(labels):
                raise ValueError(f"Hand count differs: {video_id}, {t}")
            frame_rows.append({
                "selection_group": group, "video_id": video_id, "frame_1based": t + 1,
                "source_index": result["source_index"],
                "base_left_seen": int(base_seen["left"][t]),
                "base_right_seen": int(base_seen["right"][t]),
                "dense_count": len(wrists), "dense_labels": "|".join(labels),
                "visual_left_hand": "", "visual_right_hand": "",
                "visual_active_hand": "", "visual_occlusion": "",
                "visual_face_overlap": "", "visual_notes": "",
            })
            for detection_index, (xy, label) in enumerate(zip(wrists, labels)):
                wrist = np.asarray(xy, dtype=float)
                matches = [side for side in ("left", "right") if base_seen[side][t]
                           and np.linalg.norm(wrist - base_wrist[side][t]) < 0.06]
                if matches:
                    key = "matched_same_label" if len(matches) == 1 and label == matches[0] else "matched_label_conflict_or_ambiguous"
                    by_group[group][key] += 1
                    by_video[video_id][key] += 1
                    continue
                nose = pose[t, 0, :2]
                width = float(np.linalg.norm(pose[t, 11, :2] - pose[t, 12, :2]))
                face_near = bool(np.isfinite(nose).all() and 0.05 <= width <= 0.8
                                 and np.linalg.norm(wrist - nose) <= max(0.08, 0.9 * width))
                pose_distances = {side: float(np.linalg.norm(
                    wrist - pose[t, 15 if side == "left" else 16, :2]))
                    for side in ("left", "right")}
                nearest = min(pose_distances, key=pose_distances.get)
                other = "right" if nearest == "left" else "left"
                margin = pose_distances[other] - pose_distances[nearest]
                # This conservative gate is measured only; no NPZ is written.
                # Any label/pose disagreement, overlap, or uncertain side is refused.
                one_missing = sum(not base_seen[side][t] for side in ("left", "right")) == 1
                safe_geometry = bool(one_missing and not base_seen[label][t]
                                     and label == nearest and margin >= 0.03
                                     and pose_distances[label] <= 0.18)
                row = {
                    "selection_group": group, "video_id": video_id,
                    "frame_1based": t + 1, "source_index": result["source_index"],
                    "detection_index": detection_index, "wrist_x": xy[0], "wrist_y": xy[1],
                    "detector_side": label, "nearest_pose_side": nearest,
                    "pose_margin": round(margin, 4), "face_near": int(face_near),
                    "left_base_seen": int(base_seen["left"][t]),
                    "right_base_seen": int(base_seen["right"][t]),
                    "conservative_geometry_gate": int(safe_geometry),
                    "visual_real_hand": "", "visual_active_or_passive": "",
                    "visual_anatomical_side": "", "visual_notes": "",
                }
                rows.append(row)
                by_group[group]["novel"] += 1
                by_video[video_id]["novel"] += 1
                if face_near:
                    by_group[group]["novel_face_near"] += 1
                    by_video[video_id]["novel_face_near"] += 1
                if safe_geometry:
                    by_group[group]["conservative_geometry_gate"] += 1
                    by_video[video_id]["conservative_geometry_gate"] += 1
    with (args.output_dir / "novel_candidate_annotation_queue.csv").open(
            "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (args.output_dir / "frame_annotation_queue.csv").open(
            "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(frame_rows[0]))
        writer.writeheader()
        writer.writerows(frame_rows)
    summary = {
        "videos": len(queue), "sampled_frames": sum(len(r["sampled_results"]) for r in dense.values()),
        "candidates": len(rows), "by_group": {k: dict(v) for k, v in by_group.items()},
        "by_video": {k: dict(v) for k, v in by_video.items()},
        "note": "Geometric gate is diagnostic, not ground truth or a production merge rule.",
    }
    (args.output_dir / "summary.json").write_text(json.dumps(
        summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "by_video"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
