"""Read-only audit of VIDEO and upper-body ROI novel wrist proposals.

Spatial matches describe proposal overlap, not a correct anatomical hand.
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
    parser.add_argument("--dense_csv", type=Path, required=True)
    parser.add_argument("--visual_csv", type=Path, required=True)
    parser.add_argument("--roi_results", type=Path, required=True)
    parser.add_argument("--roi_only_visual", type=Path)
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    with args.dense_csv.open(newline="", encoding="utf-8") as handle:
        dense_rows = list(csv.DictReader(handle))
    with args.visual_csv.open(newline="", encoding="utf-8") as handle:
        visual_rows = list(csv.DictReader(handle))
    if len(dense_rows) != len(visual_rows):
        raise ValueError("Dense and visual candidate counts differ")
    visual = {}
    dense_by_frame = defaultdict(list)
    for number, (dense, marked) in enumerate(zip(dense_rows, visual_rows), 1):
        key = (dense["video_id"], int(dense["frame_1based"]), int(dense["detection_index"]))
        marked_key = (marked["video_id"], int(marked["frame_1based"]), int(marked["detection_index"]))
        if key != marked_key or int(marked["candidate_number"]) != number:
            raise ValueError(f"Visual annotation order mismatch: {number}")
        for coord in ("wrist_x", "wrist_y"):
            if abs(float(dense[coord]) - float(marked[coord])) > 0.0001:
                raise ValueError(f"Visual coordinate mismatch: {number}")
        if marked["visual_sign_role"] not in {"active", "passive", "neutral", "uncertain"}:
            raise ValueError(f"Missing role: {number}")
        visual[key] = marked
        dense_by_frame[key[:2]].append((number, np.array([
            float(dense["wrist_x"]), float(dense["wrist_y"])], dtype=float)))

    roi_records = json.loads(args.roi_results.read_text(encoding="utf-8"))["records"]
    roi_rows = []
    matched_dense = set()
    roi_counts = Counter()
    for record in roi_records:
        video_id = record["video_id"]
        for frame in record["frames"]:
            frame_1based = frame["frame"] + 1
            key = (video_id, frame_1based)
            for index, point in enumerate(frame["novel_wrists"]):
                matches = [(number, float(np.linalg.norm(np.asarray(point) - xy)))
                           for number, xy in dense_by_frame[key]]
                closest = min(matches, key=lambda item: item[1]) if matches else None
                matched = closest is not None and closest[1] < 0.06
                role = visual[(video_id, frame_1based,
                               int(dense_rows[closest[0] - 1]["detection_index"]))]["visual_sign_role"] if matched else "unreviewed"
                roi_rows.append({
                    "video_id": video_id, "frame_1based": frame_1based,
                    "roi_index": index, "wrist_x": point[0], "wrist_y": point[1],
                    "spatial_match_dense": int(matched),
                    "dense_candidate_number": closest[0] if matched else "",
                    "match_distance": round(closest[1], 4) if closest else "",
                    "matched_dense_visual_role": role,
                    "roi_handedness": frame.get("novel_handedness", [""] * len(frame["novel_wrists"]))[index],
                    "roi_visual_role": "", "roi_anatomical_side": "",
                })
                roi_counts["all_novel"] += 1
                roi_counts["matched_dense" if matched else "roi_only"] += 1
                if matched:
                    matched_dense.add(closest[0])
                    roi_counts[f"matched_dense_{role}"] += 1
    out = args.output_dir / "roi_candidate_visual_queue.csv"
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(roi_rows[0]))
        writer.writeheader()
        writer.writerows(roi_rows)

    summary = {
        "videos": len(roi_records),
        "dense_candidates": len(dense_rows),
        "dense_visual_roles": dict(Counter(r["visual_sign_role"] for r in visual_rows)),
        "roi_counts": dict(roi_counts),
        "distinct_dense_candidates_matched_by_roi": len(matched_dense),
        "roi_only_count": roi_counts["roi_only"],
        "note": "Overlap is wrist geometry only; coarse visual roles and anatomical side have different certainty.",
    }
    if args.roi_only_visual:
        with args.roi_only_visual.open(newline="", encoding="utf-8") as handle:
            roi_only_visual = list(csv.DictReader(handle))
        roi_only = [row for row in roi_rows if row["spatial_match_dense"] == 0]
        if len(roi_only_visual) != len(roi_only):
            raise ValueError("ROI-only visual annotation count differs")
        for raw, reviewed in zip(roi_only, roi_only_visual):
            if ((raw["video_id"], raw["frame_1based"], raw["roi_index"])
                    != (reviewed["video_id"], int(reviewed["frame_1based"]), int(reviewed["roi_index"]))):
                raise ValueError("ROI-only visual annotation order differs")
        frame_key = lambda row: (row["video_id"], int(row["frame_1based"]))
        dense_active = {frame_key(row) for row in visual_rows
                        if row["visual_sign_role"] == "active"}
        roi_active = {frame_key(visual_rows[number - 1]) for number in matched_dense
                      if visual_rows[number - 1]["visual_sign_role"] == "active"}
        roi_active |= {frame_key(row) for row in roi_only_visual
                       if row["visual_sign_role"] == "active"}
        union_active = dense_active | roi_active
        summary["active_wrist_candidate_frames"] = {
            "dense": len(dense_active), "roi": len(roi_active),
            "union": len(union_active), "roi_only_gain_over_dense": len(roi_active - dense_active),
            "dense_only_gain_over_roi": len(dense_active - roi_active),
            "sampled_frames_denominator": len(roi_records) * 16,
            "note": "Contact-sheet active-role candidate positions, NOT verified hands, missing-hand recall, or model accuracy."
        }
    (args.output_dir / "candidate_union_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
