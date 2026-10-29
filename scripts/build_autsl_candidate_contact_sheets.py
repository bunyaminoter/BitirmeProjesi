"""Show every novel hand candidate in its sampled frame with a marked wrist."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate_csv", type=Path, required=True)
    parser.add_argument("--panel_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--only_roi_unmatched", action="store_true",
                        help="Use the ROI union queue and show only proposals absent from dense VIDEO")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with args.candidate_csv.open(newline="", encoding="utf-8") as handle:
        candidates = list(csv.DictReader(handle))
    if args.only_roi_unmatched:
        candidates = [row for row in candidates if row["spatial_match_dense"] == "0"]
    index_rows = []
    panel_cache = {}
    for start in range(0, len(candidates), 16):
        sheet = np.zeros((1200, 1200, 3), dtype=np.uint8)
        for tile_index, row in enumerate(candidates[start:start + 16]):
            video_id = row["video_id"]
            if video_id not in panel_cache:
                panel = cv2.imread(str(args.panel_dir / f"{video_id}.png"))
                if panel is None or panel.shape[:2] != (1200, 1200):
                    raise ValueError(f"Missing 1200x1200 panel: {video_id}")
                panel_cache[video_id] = panel
            frame = int(row["frame_1based"]) - 1
            original = panel_cache[video_id]
            tile = original[(frame // 4) * 300:(frame // 4 + 1) * 300,
                            (frame % 4) * 300:(frame % 4 + 1) * 300].copy()
            x = round(float(row["wrist_x"]) * 300)
            y = round(float(row["wrist_y"]) * 300)
            cv2.circle(tile, (x, y), 17, (255, 0, 255), 2)
            cv2.line(tile, (x - 8, y), (x + 8, y), (255, 0, 255), 2)
            cv2.line(tile, (x, y - 8), (x, y + 8), (255, 0, 255), 2)
            cv2.rectangle(tile, (0, 0), (300, 43), (0, 0, 0), -1)
            number = start + tile_index + 1
            cv2.putText(tile, f"{number}: {video_id.replace('_color', '')} f{frame+1}",
                        (4, 17), cv2.FONT_HERSHEY_SIMPLEX, .42, (255, 255, 255), 1)
            info = (f"roi={row['roi_handedness']} dense={row['spatial_match_dense']}"
                    if args.only_roi_unmatched else
                    f"{row['detector_side']} face={row['face_near']} gate={row['conservative_geometry_gate']}")
            cv2.putText(tile, info,
                        (4, 36), cv2.FONT_HERSHEY_SIMPLEX, .42, (255, 255, 255), 1)
            sheet[(tile_index // 4) * 300:(tile_index // 4 + 1) * 300,
                  (tile_index % 4) * 300:(tile_index % 4 + 1) * 300] = tile
            index_rows.append({"candidate_number": number, **row})
        cv2.imwrite(str(args.output_dir / f"candidate_contact_{start // 16 + 1:02d}.png"), sheet)
    with (args.output_dir / "candidate_contact_index.csv").open(
            "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(index_rows[0]))
        writer.writeheader()
        writer.writerows(index_rows)
    print(f"{len(candidates)} candidates in {(len(candidates)+15)//16} sheets")


if __name__ == "__main__":
    main()
