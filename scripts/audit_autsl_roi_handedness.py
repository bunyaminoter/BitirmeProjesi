"""Compare ROI handedness to observed cache hands; diagnostic, never cache repair."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base_cache", type=Path, required=True)
    parser.add_argument("--roi_results", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = json.loads(args.roi_results.read_text(encoding="utf-8"))["records"]
    totals = Counter()
    by_video = defaultdict(Counter)
    novel = []
    for record in records:
        video_id = record["video_id"]
        with np.load(args.base_cache / "train" / f"{video_id}.npz", allow_pickle=False) as cache:
            pose = cache["pose_landmarks"]
            observed = {side: cache[f"{side}_hand_detected"].astype(bool)
                        for side in ("left", "right")}
            hands = {side: cache[f"{side}_hand_landmarks"]
                     for side in ("left", "right")}
        for frame in record["frames"]:
            t = frame["frame"]
            wrists, labels = frame["roi_wrists"], frame["roi_handedness"]
            if len(wrists) != len(labels):
                raise ValueError(f"Missing labels: {video_id} frame {t}")
            for wrist_xy, label in zip(wrists, labels):
                wrist = np.asarray(wrist_xy, dtype=float)
                matches = [side for side in observed if observed[side][t]
                           and np.linalg.norm(wrist - hands[side][t, 0, :2]) < 0.06]
                if len(matches) == 1:
                    key = "same_hand_label_agree" if label == matches[0] else "same_hand_label_disagree"
                    totals[key] += 1
                    by_video[video_id][key] += 1
                elif len(matches) > 1:
                    totals["same_hand_ambiguous"] += 1
                    by_video[video_id]["same_hand_ambiguous"] += 1
            for wrist_xy, label in zip(frame["novel_wrists"], frame["novel_handedness"]):
                wrist = np.asarray(wrist_xy, dtype=float)
                nose = pose[t, 0, :2]
                shoulder_width = float(np.linalg.norm(pose[t, 11, :2] - pose[t, 12, :2]))
                near_face = bool(np.isfinite(nose).all() and 0.05 <= shoulder_width <= 0.8
                                 and np.linalg.norm(wrist - nose) <= max(0.08, 0.9 * shoulder_width))
                nearest_pose = min(("left", "right"), key=lambda side:
                                   np.linalg.norm(wrist - pose[t, 15 if side == "left" else 16, :2]))
                other = "right" if nearest_pose == "left" else "left"
                margin = float(np.linalg.norm(wrist - pose[t, 15 if other == "left" else 16, :2])
                               - np.linalg.norm(wrist - pose[t, 15 if nearest_pose == "left" else 16, :2]))
                novel.append({"video_id": video_id, "frame_1based": t + 1,
                              "wrist_x": wrist_xy[0], "wrist_y": wrist_xy[1],
                              "roi_handedness": label, "face_near": int(near_face),
                              "nearest_pose_side": nearest_pose, "pose_margin": round(margin, 4),
                              "left_base_seen": int(observed["left"][t]),
                              "right_base_seen": int(observed["right"][t]),
                              "visual_actual_hand": "", "visual_side": "", "visual_notes": ""})
                totals["novel"] += 1
                by_video[video_id]["novel"] += 1
                if near_face:
                    totals["novel_face_near"] += 1
                    by_video[video_id]["novel_face_near"] += 1
                    if label != nearest_pose:
                        totals["face_label_pose_disagree"] += 1
                        by_video[video_id]["face_label_pose_disagree"] += 1
    with (args.output_dir / "novel_candidates.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(novel[0]))
        writer.writeheader()
        writer.writerows(novel)
    summary = {"videos": len(records), "totals": dict(totals),
               "by_video": {video_id: dict(counts) for video_id, counts in by_video.items()}}
    (args.output_dir / "summary.json").write_text(json.dumps(
        summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary["totals"], ensure_ascii=False))


if __name__ == "__main__":
    main()
