"""Read-only face-near QA flags for upper-body ROI candidates on selected train clips.

The face zone is only an audit tag. Never suppresses hands or edits caches.
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
    parser.add_argument("--roi_results", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with args.queue.open(newline="", encoding="utf-8") as handle:
        queue = list(csv.DictReader(handle))
    roi = {record["video_id"]: record for record in json.loads(
        args.roi_results.read_text(encoding="utf-8"))["records"]}
    if set(roi) != {row["video_id"] for row in queue}:
        raise ValueError("Queue and ROI results differ")
    records = []
    per_video: dict[str, Counter] = defaultdict(Counter)
    by_group: dict[str, Counter] = defaultdict(Counter)
    for item in queue:
        video_id, group = item["video_id"], item["selection_group"]
        with np.load(args.base_cache / "train" / f"{video_id}.npz", allow_pickle=False) as cache:
            pose = cache["pose_landmarks"]
            pose_ok = cache["pose_detected"].astype(bool)
            base = {side: cache[f"{side}_hand_detected"].astype(bool)
                    for side in ("left", "right")}
            hands = {side: cache[f"{side}_hand_landmarks"]
                     for side in ("left", "right")}
        if len(roi[video_id]["frames"]) != len(pose):
            raise ValueError(f"Frame mismatch: {video_id}")
        for frame in roi[video_id]["frames"]:
            t = frame["frame"]
            if not pose_ok[t]:
                continue
            # Elliptical face-near zone from pose nose/ears and shoulder width.
            # Pure geometry for triage, not a face mask or hand detector.
            nose = pose[t, 0, :2]
            shoulders = pose[t, [11, 12], :2]
            shoulder_width = float(np.linalg.norm(shoulders[0] - shoulders[1]))
            if not np.isfinite(nose).all() or not 0.05 <= shoulder_width <= 0.8:
                continue
            radius = max(0.08, 0.9 * shoulder_width)
            for wrist in frame["novel_wrists"]:
                wrist = np.asarray(wrist, dtype=float)
                if not np.isfinite(wrist).all():
                    continue
                face_distance = float(np.linalg.norm(wrist - nose) / radius)
                near_face = face_distance <= 1.0
                base_seen = {side: bool(base[side][t]) for side in base}
                pose_distances = {side: float(np.linalg.norm(
                    wrist - pose[t, 15 if side == "left" else 16, :2]))
                    for side in base}
                own_rank = sorted(pose_distances, key=pose_distances.get)
                duplicate = any(base_seen[side] and np.linalg.norm(
                    wrist - hands[side][t, 0, :2]) < 0.06 for side in base)
                row = {
                    "selection_group": group, "video_id": video_id, "frame_1based": t + 1,
                    "wrist_x": round(float(wrist[0]), 4), "wrist_y": round(float(wrist[1]), 4),
                    "face_near": int(near_face), "face_distance_over_radius": round(face_distance, 3),
                    "left_base_seen": int(base_seen["left"]), "right_base_seen": int(base_seen["right"]),
                    "nearest_pose_side": own_rank[0],
                    "pose_side_margin": round(pose_distances[own_rank[1]] - pose_distances[own_rank[0]], 4),
                    "possible_duplicate": int(duplicate),
                    "visual_actual_hand": "", "visual_active_hand": "", "visual_side": "", "visual_notes": "",
                }
                records.append(row)
                per_video[video_id]["novel"] += 1
                by_group[group]["novel"] += 1
                if near_face:
                    per_video[video_id]["face_near"] += 1
                    by_group[group]["face_near"] += 1
                    if not any(base_seen.values()):
                        per_video[video_id]["face_near_base_both_missing"] += 1
                        by_group[group]["face_near_base_both_missing"] += 1
                if duplicate:
                    per_video[video_id]["possible_duplicate"] += 1
                    by_group[group]["possible_duplicate"] += 1
    output = args.output_dir / "face_zone_candidate_queue.csv"
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    summary = {"policy": "nose-centered radius=max(0.08,0.9*shoulder_width); audit tag only",
               "videos": len(queue), "candidates": len(records),
               "by_group": {key: dict(value) for key, value in by_group.items()},
               "per_video": {key: dict(value) for key, value in per_video.items()}}
    (args.output_dir / "summary.json").write_text(json.dumps(
        summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"videos": summary["videos"], "candidates": summary["candidates"],
                      "by_group": summary["by_group"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
