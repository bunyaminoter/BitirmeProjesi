"""Build raw-frame contact sheets for recurring 100-class shared errors.

These are small visual QA samples. They do not replace full video playback or
expert sign annotation.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps


GROUPS = {
    "yakin": [
        "signer1_sample152_color", "signer1_sample441_color", "signer1_sample863_color",
        "signer18_sample152_color", "signer25_sample124_color",
    ],
    "agabey": [
        "signer11_sample44_color", "signer11_sample658_color", "signer25_sample104_color",
        "signer1_sample102_color", "signer18_sample268_color",
    ],
    "seker": [
        "signer11_sample267_color", "signer11_sample485_color", "signer11_sample77_color",
        "signer1_sample607_color", "signer25_sample13_color",
    ],
    "gormek": [
        "signer1_sample1032_color", "signer1_sample655_color", "signer1_sample787_color",
        "signer25_sample186_color", "signer35_sample624_color",
    ],
    "signer16_missing": [
        "signer16_sample22_color", "signer16_sample244_color", "signer16_sample276_color",
        "signer16_sample101_color", "signer16_sample110_color",
    ],
}


def decode_uniform(path: Path, count: int = 6) -> tuple[list[Image.Image], int]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open: {path}")
    frames = []
    while True:
        ok, bgr = capture.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    capture.release()
    if not frames:
        raise RuntimeError(f"No frames: {path}")
    indices = np.rint(np.linspace(0, len(frames) - 1, count)).astype(int)
    return [Image.fromarray(frames[index]) for index in indices], len(frames)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw_val_dir", type=Path, required=True)
    parser.add_argument("--audit_csv", type=Path, required=True)
    parser.add_argument("--class_list", type=Path, required=True)
    parser.add_argument("--class_ids_from_predictions", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args()
    rows = {r["video_id"]: r for r in csv.DictReader(
        args.audit_csv.open(newline="", encoding="utf-8-sig"))}
    class_ids = json.loads(args.class_ids_from_predictions.read_text(
        encoding="utf-8"))["class_ids"]
    words = {class_id: line.strip().split(" ", 1)[1]
             for class_id, line in zip(class_ids, args.class_list.open(encoding="utf-8"))}
    font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 15)
    title_font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 19)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metadata = []
    for word, ids in GROUPS.items():
        width, height = 178, 126
        sheet = Image.new("RGB", (6 * width + 24, len(ids) * 172 + 46), "white")
        draw = ImageDraw.Draw(sheet)
        draw.text((12, 10), f"AUTSL 100 shared-error QA: {word}",
                  font=title_font, fill="black")
        for row_index, video_id in enumerate(ids):
            info = rows[video_id]
            frames, n_raw = decode_uniform(args.raw_val_dir / f"{video_id}.mp4")
            label = ("ERROR" if info["status"] == "both_wrong" else "CONTROL")
            preds = (f"{words[int(info['pose_pred_original_id'])]} / "
                     f"{words[int(info['hybrid_pred_original_id'])]}")
            y = 42 + row_index * 172
            draw.text((12, y), f"{label} {video_id}  pose/RGB: {preds}",
                      font=font, fill="black")
            for column, image in enumerate(frames):
                tile = Image.new("RGB", (width - 6, height), "#252525")
                frame = ImageOps.contain(image, tile.size)
                tile.paste(frame, ((tile.width - frame.width) // 2,
                                   (tile.height - frame.height) // 2))
                sheet.paste(tile, (12 + column * width, y + 27))
            metadata.append({"group": word, "video_id": video_id,
                             "status": info["status"], "raw_frames": n_raw,
                             "both_missing_sampled_frames": int(info["both_missing_frames"])})
        dest = args.output_dir / f"raw_shared_{word}.png"
        sheet.save(dest)
        print(dest, flush=True)
    (args.output_dir / "raw_board_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
