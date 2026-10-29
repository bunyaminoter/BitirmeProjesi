"""Overlay cached hand landmarks on exactly sampled raw AUTSL frames."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from build_100_shared_error_boards import GROUPS


EDGES = ((0, 1), (1, 2), (2, 3), (3, 4),
         (0, 5), (5, 6), (6, 7), (7, 8),
         (0, 9), (9, 10), (10, 11), (11, 12),
         (0, 13), (13, 14), (14, 15), (15, 16),
         (0, 17), (17, 18), (18, 19), (19, 20))


def decode(path: Path) -> list[np.ndarray]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open {path}")
    frames = []
    while True:
        ok, bgr = capture.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    capture.release()
    if not frames:
        raise RuntimeError(f"No frames: {path}")
    return frames


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw_val_dir", type=Path, required=True)
    parser.add_argument("--cache_dir", type=Path, required=True)
    parser.add_argument("--audit_csv", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args()
    status = {r["video_id"]: r["status"] for r in csv.DictReader(
        args.audit_csv.open(newline="", encoding="utf-8-sig"))}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 14)
    for group, ids in GROUPS.items():
        # Two recurrent errors and one correct control per group keep the sheet legible.
        selected = ids[:2] + ids[3:4]
        width, height = 320, 240
        row_height = 270
        sheet = Image.new("RGB", (3 * width + 20, len(selected) * row_height + 40), "white")
        draw = ImageDraw.Draw(sheet)
        draw.text((8, 8), f"{group}: cached hand landmarks L=green R=magenta",
                  font=font, fill="black")
        for ri, video_id in enumerate(selected):
            frames = decode(args.raw_val_dir / f"{video_id}.mp4")
            with np.load(args.cache_dir / "val" / f"{video_id}.npz", allow_pickle=False) as d:
                left = d["left_hand_landmarks"]
                right = d["right_hand_landmarks"]
                left_ok = d["left_hand_detected"]
                right_ok = d["right_hand_detected"]
            indices = np.linspace(0, len(frames) - 1, min(len(frames), len(left)),
                                  dtype=int)
            y = 38 + ri * row_height
            draw.text((8, y), f"{status[video_id]} {video_id}", font=font, fill="black")
            for col, k in enumerate((4, 7, 10)):
                if k >= len(indices):
                    continue
                im = Image.fromarray(frames[int(indices[k])]).resize((width, height))
                overlay = ImageDraw.Draw(im)
                for points, found, color in ((left[k], left_ok[k], "#10f74b"),
                                             (right[k], right_ok[k], "#ff2ac4")):
                    if not found:
                        continue
                    xy = [(float(point[0]) * width, float(point[1]) * height)
                          for point in points]
                    for a, b in EDGES:
                        overlay.line((xy[a], xy[b]), fill=color, width=2)
                    for x, yy in xy:
                        overlay.ellipse((x - 2, yy - 2, x + 2, yy + 2), fill=color)
                overlay.text((5, 5), f"sample {k}: L{int(left_ok[k])} R{int(right_ok[k])}",
                             font=font, fill="yellow", stroke_width=2,
                             stroke_fill="black")
                sheet.paste(im, (8 + col * width, y + 24))
        dest = args.output_dir / f"landmark_shared_{group}.png"
        sheet.save(dest)
        print(dest, flush=True)


if __name__ == "__main__":
    main()
