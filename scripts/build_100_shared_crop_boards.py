"""Display strict-cache left/right hand crops for recurrent shared errors."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from build_100_shared_error_boards import GROUPS


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache_dir", type=Path, required=True)
    parser.add_argument("--audit_csv", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args()
    rows = {r["video_id"]: r for r in csv.DictReader(
        args.audit_csv.open(newline="", encoding="utf-8-sig"))}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 14)
    for word, ids in GROUPS.items():
        tile = 128
        row_height = 167
        sheet = Image.new("RGB", (8 * tile + 20, len(ids) * row_height + 40), "white")
        draw = ImageDraw.Draw(sheet)
        draw.text((10, 9), f"{word}: strict crop frames 3,6,9,12; L/R pairs; green=detected",
                  font=font, fill="black")
        for row_index, video_id in enumerate(ids):
            with np.load(args.cache_dir / "val" / f"{video_id}.npz", allow_pickle=False) as d:
                left = d["left_hand_crops"]
                right = d["right_hand_crops"]
                left_ok = d["left_hand_detected"]
                right_ok = d["right_hand_detected"]
                y = 37 + row_index * row_height
                draw.text((10, y), f"{rows[video_id]['status']} {video_id}",
                          font=font, fill="black")
                for j, frame in enumerate((3, 6, 9, 12)):
                    if frame >= len(left):
                        continue
                    for side, crops, detected in ((0, left, left_ok), (1, right, right_ok)):
                        x = 10 + (2 * j + side) * tile
                        im = Image.fromarray(crops[frame]).resize((tile - 5, tile - 5))
                        sheet.paste(im, (x, y + 24))
                        draw.rectangle((x, y + 24, x + tile - 6, y + tile + 19),
                                       outline="#159947" if detected[frame] else "#c53434", width=2)
                        draw.text((x + 3, y + 26), f"{frame}:{'L' if side == 0 else 'R'}",
                                  font=font, fill="yellow", stroke_width=2,
                                  stroke_fill="black")
        dest = args.output_dir / f"crop_shared_{word}.png"
        sheet.save(dest)
        print(dest)


if __name__ == "__main__":
    main()
