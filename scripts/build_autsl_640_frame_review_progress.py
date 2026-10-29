"""Join reviewed wrist candidates onto the 640-frame annotation queue.

Candidate roles are not promoted into full-frame anatomical ground truth.
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path


def read(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frame_queue", type=Path, required=True)
    parser.add_argument("--dense_visual", type=Path, required=True)
    parser.add_argument("--roi_only_visual", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    frames = read(args.frame_queue)
    if len(frames) != 640 or len({(r["video_id"], r["frame_1based"]) for r in frames}) != 640:
        raise ValueError("640-frame queue changed")
    by_frame = defaultdict(lambda: {"dense": Counter(), "roi_only": Counter()})
    for method, path in (("dense", args.dense_visual), ("roi_only", args.roi_only_visual)):
        for row in read(path):
            by_frame[(row["video_id"], row["frame_1based"])][method][row["visual_sign_role"]] += 1
    for row in frames:
        annotation = by_frame[(row["video_id"], row["frame_1based"])]
        for method in ("dense", "roi_only"):
            for role in ("active", "passive", "neutral", "uncertain"):
                row[f"reviewed_{method}_{role}_wrist_candidates"] = annotation[method][role]
        row["full_frame_human_anatomical_sides"] = ""
        row["full_frame_human_active_passive_roles"] = ""
        row["full_frame_human_landmark_geometry"] = ""
        row["review_status"] = "candidate_locations_only" if any(
            annotation[method] for method in ("dense", "roi_only")) else "unreviewed"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(frames[0]))
        writer.writeheader()
        writer.writerows(frames)
    print(dict(Counter(row["review_status"] for row in frames)))


if __name__ == "__main__":
    main()
