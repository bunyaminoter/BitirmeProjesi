"""Create visual quality checks for AUTSL sampling and cached landmarks.

This script does not train a model or run MediaPipe. It uses cached features
when the frame count matches the cache manifest; other counts show RGB only.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from scripts.preprocess_tid import sample_frames
from src.core.config import load_config
from src.data.datasets.tid_dataset import TIDDataset


HAND_EDGES = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
              (5, 9), (9, 10), (10, 11), (11, 12), (9, 13), (13, 14),
              (14, 15), (15, 16), (13, 17), (17, 18), (18, 19), (19, 20),
              (0, 17)]
ARM_EDGES = [(11, 12), (11, 13), (13, 15), (12, 14), (14, 16)]


def choose_samples(dataset: TIDDataset, video_id: str | None,
                   max_videos: int, cache_dir: Path):
    if video_id:
        selected = [(i, sample) for i, sample in enumerate(dataset.samples)
                    if sample.video_id == video_id or sample.video_id.removesuffix("_color") == video_id]
        if not selected:
            raise ValueError(f"Video is not in the selected split/subset: {video_id}")
        return selected
    by_class: dict[int, list[tuple[int, object]]] = defaultdict(list)
    for i, sample in enumerate(dataset.samples):
        by_class[sample.label].append((i, sample))
    selected = []
    signer_counts: dict[str, int] = defaultdict(int)
    for class_id in sorted(by_class):
        candidates = by_class[class_id]
        cached = [entry for entry in candidates if
                  (cache_dir / dataset.split / f"{entry[1].video_id}.npz").is_file()]
        chosen = min(cached or candidates,
                     key=lambda entry: (signer_counts[entry[1].signer_id], entry[0]))
        selected.append(chosen)
        signer_counts[chosen[1].signer_id] += 1
        if len(selected) >= max_videos:
            break
    return selected


def draw_edges(image: np.ndarray, points: np.ndarray, edges, color):
    h, w = image.shape[:2]
    for a, b in edges:
        p, q = points[a], points[b]
        if not (np.isfinite(p[:2]).all() and np.isfinite(q[:2]).all()):
            continue
        pxy = tuple(np.rint(p[:2] * [w, h]).astype(int))
        qxy = tuple(np.rint(q[:2] * [w, h]).astype(int))
        cv2.line(image, pxy, qxy, color, 2, cv2.LINE_AA)
    for point in points:
        if np.isfinite(point[:2]).all():
            xy = tuple(np.rint(point[:2] * [w, h]).astype(int))
            cv2.circle(image, xy, 2, color, -1, cv2.LINE_AA)


def draw_grid(frames: list[np.ndarray], features: dict | None, count: int,
              title: str, output: Path):
    columns = 4 if count <= 16 else 6
    thumb_w, thumb_h = 300, 220
    rows = (len(frames) + columns - 1) // columns
    canvas = np.full((rows * thumb_h + 35, columns * thumb_w, 3), 28, np.uint8)
    cv2.putText(canvas, title, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (255, 255, 255), 1, cv2.LINE_AA)
    for i, frame in enumerate(frames):
        rgb = frame.copy()
        if features is not None and i < int(features["num_frames"]):
            if features["pose_detected"][i]:
                draw_edges(rgb, features["pose_landmarks"][i], ARM_EDGES, (255, 255, 0))
            for side, color in (("left", (255, 80, 80)), ("right", (80, 255, 80))):
                if features[f"{side}_hand_detected"][i]:
                    draw_edges(rgb, features[f"{side}_hand_landmarks"][i], HAND_EDGES, color)
                elif f"{side}_hand_imputed" in features and features[f"{side}_hand_imputed"][i]:
                    draw_edges(rgb, features[f"{side}_hand_landmarks"][i], HAND_EDGES, (255, 180, 0))
        image = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        image = cv2.resize(image, (thumb_w, thumb_h - 30), interpolation=cv2.INTER_AREA)
        x, y = (i % columns) * thumb_w, (i // columns) * thumb_h + 35
        canvas[y:y + thumb_h - 30, x:x + thumb_w] = image
        detected = ""
        if features is not None and i < int(features["num_frames"]):
            detected = (f" L:{int(features['left_hand_detected'][i])}"
                        f" R:{int(features['right_hand_detected'][i])}")
            if "left_hand_imputed" in features:
                detected += (f" IL:{int(features['left_hand_imputed'][i])}"
                             f" IR:{int(features['right_hand_imputed'][i])}")
        cv2.putText(canvas, f"sample {i + 1}/{count}{detected}",
                    (x + 6, y + thumb_h - 8), cv2.FONT_HERSHEY_SIMPLEX,
                    0.47, (255, 255, 255), 1, cv2.LINE_AA)
    output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output), canvas):
        raise OSError(f"Cannot write {output}")


def draw_crops(features: dict, title: str, output: Path):
    count = int(features["num_frames"])
    width = 170
    cell_height = 365
    columns = 8
    rows = (count + columns - 1) // columns
    canvas = np.full((rows * cell_height + 35, columns * width, 3), 28, np.uint8)
    cv2.putText(canvas, title, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (255, 255, 255), 1, cv2.LINE_AA)
    for i in range(count):
        x, y = (i % columns) * width, (i // columns) * cell_height + 35
        for offset, side in ((0, "left"), (170, "right")):
            crop = features[f"{side}_hand_crops"][i]
            crop = cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)
            crop = cv2.resize(crop, (160, 160))
            canvas[y + offset:y + offset + 160, x:x + 160] = crop
            label = f"{side} {'OK' if features[f'{side}_hand_detected'][i] else 'missing'}"
            cv2.putText(canvas, label, (x + 2, y + offset + 168),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output), canvas):
        raise OSError(f"Cannot write {output}")


def summarize(features: dict) -> dict:
    count = int(features["num_frames"])
    result = {"frames": count}
    for name in ("pose", "left_hand", "right_hand"):
        values = features[f"{name}_detected"][:count].astype(bool)
        result[f"{name}_detected"] = int(values.sum())
        result[f"{name}_longest_missing_run"] = max(
            (len(run) for run in "".join("1" if not v else "0" for v in values).split("0")),
            default=0,
        )
        if name != "pose" and f"{name}_imputed" in features:
            result[f"{name}_imputed"] = int(features[f"{name}_imputed"][:count].sum())
    pose = features["pose_landmarks"][:count]
    for side, own_idx, other_idx in (("left", 15, 16), ("right", 16, 15)):
        hand = features[f"{side}_hand_landmarks"][:count]
        valid = features["pose_detected"][:count] & features[f"{side}_hand_detected"][:count]
        own = np.linalg.norm(hand[valid, 0, :2] - pose[valid, own_idx, :2], axis=1)
        other = np.linalg.norm(hand[valid, 0, :2] - pose[valid, other_idx, :2], axis=1)
        result[f"{side}_wrist_agreement"] = round(float((own < other).mean()), 3) if len(own) else None
        result[f"{side}_wrist_disagreement_frames"] = (
            (np.flatnonzero(valid)[own >= other] + 1).tolist()
        )
        coords = hand[features[f"{side}_hand_detected"][:count], :, :2]
        result[f"{side}_out_of_frame_points"] = int(((coords < 0) | (coords > 1)).any(axis=2).sum())
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path,
                        default=project_root / "configs/experiment/autsl_10_mobilenet.yaml")
    parser.add_argument("--data_dir", type=Path, default=None)
    parser.add_argument("--split", choices=["train", "val"], default="val")
    parser.add_argument("--video_id", default=None)
    parser.add_argument("--max_videos", type=int, default=10)
    parser.add_argument("--frame_counts", type=int, nargs="+", default=[16, 36])
    parser.add_argument("--cache_dir", type=Path, default=None)
    parser.add_argument("--output_dir", type=Path, default=project_root / "outputs/preprocessing_qa")
    args = parser.parse_args()
    if args.max_videos < 1 or any(count < 1 for count in args.frame_counts):
        parser.error("--max_videos and frame counts must be positive")
    if args.data_dir:
        os.environ["AUTSL_DATA_DIR"] = str(args.data_dir.resolve())
    config = load_config(str(args.config))
    dataset = TIDDataset(config.dataset, args.split)
    cache_dir = args.cache_dir or project_root / config.dataset.cache_dir
    manifest_path = cache_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else None
    records = []
    for index, sample in choose_samples(dataset, args.video_id, args.max_videos, cache_dir):
        video_path = dataset.get_video_path(index)
        record = {"split": args.split, "video_id": sample.video_id,
                  "class_id": sample.label, "gloss": sample.gloss,
                  "sheets": {}, "cache_metrics": None}
        for count in args.frame_counts:
            frames = sample_frames(str(video_path), count)
            if not frames:
                raise ValueError(f"Video could not be decoded: {video_path}")
            features = None
            cache_path = cache_dir / args.split / f"{sample.video_id}.npz"
            if manifest and manifest.get("num_frames") == count and cache_path.is_file():
                with np.load(cache_path, allow_pickle=False) as loaded:
                    features = dict(loaded)
                if len(frames) != int(features["num_frames"]):
                    raise ValueError(f"Frame/cache length mismatch: {cache_path}")
                record["cache_metrics"] = summarize(features)
                if "left_hand_crops" in features and "right_hand_crops" in features:
                    crop_output = args.output_dir / f"{args.split}_{sample.video_id}_{count}crops.png"
                    draw_crops(features, f"{args.split} {sample.video_id} cached hand crops", crop_output)
                    record["sheets"][f"{count}_crops"] = str(crop_output)
            output = args.output_dir / f"{args.split}_{sample.video_id}_{count}frames.png"
            draw_grid(frames, features, count,
                      f"{args.split} {sample.video_id} class {sample.label} {sample.gloss} {count} frames"
                      + (" + cached landmarks" if features is not None else " RGB only"), output)
            record["sheets"][str(count)] = str(output)
        records.append(record)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / f"{args.split}_inspection.json"
    summary_path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    print(summary_path)
    for record in records:
        print(record["video_id"], record["gloss"], record["cache_metrics"])


if __name__ == "__main__":
    main()
