"""Safely reuse the frozen 100-class base16 NPZ in a 226-class cache.

Only the scalar model label changes: 100-class index -> official 0..225 class ID.
The source cache is read-only. Destination writes are atomic per video.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import tempfile
from pathlib import Path

import numpy as np


def labels(path: Path) -> dict[str, int]:
    with path.open(newline="", encoding="utf-8-sig") as source:
        rows = [(f"{video_id}_color", int(class_id))
                for video_id, class_id in csv.reader(source)]
    result = dict(rows)
    if len(result) != len(rows):
        raise ValueError(f"Duplicate label video IDs: {path}")
    return result


def check_features(data: dict[str, np.ndarray]) -> None:
    n = int(data["num_frames"])
    if not 1 <= n <= 16:
        raise ValueError(f"Invalid frame count: {n}")
    expected = {
        "pose_landmarks": (n, 33, 3),
        "left_hand_landmarks": (n, 21, 3),
        "right_hand_landmarks": (n, 21, 3),
        "left_hand_crops": (n, 160, 160, 3),
        "right_hand_crops": (n, 160, 160, 3),
        "pose_detected": (n,),
        "left_hand_detected": (n,),
        "right_hand_detected": (n,),
    }
    for field, shape in expected.items():
        if data[field].shape != shape:
            raise ValueError(f"{field} shape: {data[field].shape} != {shape}")
    if not all(np.isfinite(data[field]).all() for field in
               ("pose_landmarks", "left_hand_landmarks", "right_hand_landmarks")):
        raise ValueError("Nonfinite landmarks")


def write_atomic(path: Path, data: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w+b", suffix=".npz", dir=path.parent,
                                     delete=False) as temp:
        temp_path = Path(temp.name)
        try:
            np.savez_compressed(temp, **data)
        except BaseException:
            temp_path.unlink(missing_ok=True)
            raise
    try:
        os.replace(temp_path, path)
    except BaseException:
        temp_path.unlink(missing_ok=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source_cache", type=Path, required=True)
    parser.add_argument("--dest_cache", type=Path, required=True)
    parser.add_argument("--labels_dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit_per_split", type=int, default=None,
                        help="Small pilot only; omit for full 14,368-video reuse")
    args = parser.parse_args()
    if args.source_cache.resolve() == args.dest_cache.resolve():
        raise ValueError("Source and destination must differ")
    source_manifest = json.loads((args.source_cache / "manifest.json").read_text(
        encoding="utf-8"))
    class_ids = source_manifest.get("class_ids")
    if (source_manifest.get("dataset") != "autsl"
            or source_manifest.get("pipeline_version") != 2
            or source_manifest.get("num_frames") != 16
            or source_manifest.get("num_classes") != 100
            or source_manifest.get("hand_image_size") != [160, 160]
            or not isinstance(class_ids, list) or len(class_ids) != 100
            or len(set(class_ids)) != 100):
        raise ValueError("Expected frozen 100-class base16 source manifest")
    if any(key in source_manifest for key in ("derivation", "cache_derivation")):
        raise ValueError("This command accepts base16 source only, not strict")
    target_manifest = {key: source_manifest[key] for key in
                       ("dataset", "pipeline_version", "num_frames", "hand_image_size")}
    target_manifest.update(num_classes=226, class_ids=None)
    args.dest_cache.mkdir(parents=True, exist_ok=True)
    dest_manifest = args.dest_cache / "manifest.json"
    if dest_manifest.exists():
        if json.loads(dest_manifest.read_text(encoding="utf-8")) != target_manifest:
            raise ValueError("Destination manifest differs from 226 base16")
    else:
        dest_manifest.write_text(json.dumps(target_manifest, indent=2) + "\n",
                                 encoding="utf-8")

    report = {"source_cache": str(args.source_cache), "dest_cache": str(args.dest_cache),
              "pilot_limit_per_split": args.limit_per_split, "splits": {}}
    for split, expected_count in (("train", 12405), ("val", 1963)):
        source_files = sorted((args.source_cache / split).glob("*.npz"))
        if args.limit_per_split is None and len(source_files) != expected_count:
            raise ValueError(f"Incomplete 100-class {split} source: {len(source_files)}")
        official = labels(args.labels_dir / f"{split}_labels.csv")
        if args.limit_per_split is not None:
            source_files = source_files[:args.limit_per_split]
        processed = skipped = 0
        for source_path in source_files:
            video_id = source_path.stem
            if video_id not in official or official[video_id] not in class_ids:
                raise ValueError(f"Unexpected source video/class: {video_id}")
            original_id = official[video_id]
            local_index = class_ids.index(original_id)
            target_path = args.dest_cache / split / source_path.name
            with np.load(source_path, allow_pickle=False) as source:
                data = {key: source[key] for key in source.files}
            if int(data["label"]) != local_index:
                raise ValueError(f"100-class label mismatch: {video_id}")
            check_features(data)
            data["label"] = np.array(original_id, dtype=np.int64)
            if target_path.exists():
                with np.load(target_path, allow_pickle=False) as existing:
                    if set(existing.files) != set(data) or any(
                        not np.array_equal(existing[key], value) for key, value in data.items()
                    ):
                        raise ValueError(f"Existing destination differs: {target_path}")
                skipped += 1
                continue
            write_atomic(target_path, data)
            with np.load(target_path, allow_pickle=False) as written:
                if set(written.files) != set(data) or any(
                    not np.array_equal(written[key], value) for key, value in data.items()
                ):
                    raise ValueError(f"Post-write verification failed: {target_path}")
            processed += 1
        report["splits"][split] = {
            "source_files_available": len(source_files),
            "written": processed, "verified_existing": skipped,
        }
        print(f"{split}: written={processed}, verified_existing={skipped}", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                           encoding="utf-8")


if __name__ == "__main__":
    main()
