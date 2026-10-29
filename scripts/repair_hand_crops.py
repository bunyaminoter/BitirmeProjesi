"""Build a separate AUTSL cache whose missing-hand crops come from the pose wrist.

Source cache and landmarks are untouched; the original video is re-read at the
same frame indices.  A frame-alignment check recomputes one detected crop from
cached landmarks and stops if it does not match the cached crop.
Only train/val are allowed: test is not opened for design decisions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "scripts"))

from preprocess_tid import sample_frames  # noqa: E402
from src.core.config import load_config  # noqa: E402
from src.core.types import LandmarkResult  # noqa: E402
from src.data.datasets.tid_dataset import TIDDataset  # noqa: E402
from src.preprocessing.hand_cropper import HandCropper  # noqa: E402
from src.preprocessing.wrist_fallback_crop import (  # noqa: E402
    DEFAULT_CENTER_REPAIR, fallback_crops, fallback_geometry, center_repair_policy,
)

METHOD = "pose_wrist_fallback_crop"


def alignment_error(frames, data, count, cropper) -> float | None:
    """Mean absolute pixel difference of one recomputed detected crop (None if no detection)."""
    for side in ("left", "right"):
        observed = data[f"{side}_hand_detected"][:count]
        if observed.any():
            index = int(np.flatnonzero(observed)[len(np.flatnonzero(observed)) // 2])
            hand = data[f"{side}_hand_landmarks"][index]
            result = LandmarkResult(pose=None, left_hand=hand if side == "left" else None,
                                    right_hand=hand if side == "right" else None)
            left, right = cropper.crop_hands(frames[index], result)
            crop = (left if side == "left" else right).image
            return float(np.abs(crop.astype(np.int16) - data[f"{side}_hand_crops"][index].astype(np.int16)).mean())
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=project_root / "configs/experiment/autsl_10_mobilenet.yaml",
                        help="Config of the SOURCE cache (frame count, classes, cache_dir)")
    parser.add_argument("--dest_cache", type=Path, required=True)
    parser.add_argument("--splits", nargs="+", choices=["train", "val"], default=["train", "val"])
    parser.add_argument("--video_ids_file", type=Path, default=None)
    parser.add_argument("--max_alignment_error", type=float, default=3.0)
    parser.add_argument("--box_scale", type=float, default=1.0,
                        help="Missing-hand box side multiplier; detected crops are unchanged")
    parser.add_argument("--center_repair", action="store_true",
                        help="Conservative short-gap bbox center repair; creates derivation version 3")
    parser.add_argument("--center_policy", type=Path,
                        help="Explicit policy JSON for reproducible thresholds (requires --center_repair)")
    parser.add_argument("--data_dir", type=str, default=None)
    args = parser.parse_args()
    if not np.isfinite(args.box_scale) or args.box_scale <= 0:
        parser.error("--box_scale must be finite and positive")
    if args.center_policy and not args.center_repair:
        parser.error("--center_policy requires --center_repair")
    if args.data_dir:
        os.environ["AUTSL_DATA_DIR"] = args.data_dir

    config = load_config(str(args.config))
    if config.dataset.cache_derivation is not None:
        parser.error("Source config must describe an unmodified MediaPipe cache")
    source = (project_root / config.dataset.cache_dir).resolve()
    dest = args.dest_cache.resolve()
    if source == dest or source in dest.parents or dest in source.parents:
        parser.error("Source and destination caches must be separate sibling directories")
    source_manifest_bytes = (source / "manifest.json").read_bytes()
    manifest = json.loads(source_manifest_bytes)
    if manifest.get("dataset") != "autsl" or manifest.get("derivation"):
        parser.error("Source must be an unmodified AUTSL cache")
    expected = {"pipeline_version": 2, "num_frames": config.dataset.num_frames,
                "hand_image_size": list(config.model.hand_encoder.input_size),
                "num_classes": config.dataset.num_classes, "class_ids": config.dataset.class_ids}
    if any(manifest.get(key) != value for key, value in expected.items()):
        parser.error("Source manifest differs from config (frames, classes, crop size or pipeline)")
    manifest["derivation"] = {
        "method": METHOD, "version": 3 if args.center_repair else (1 if args.box_scale == 1.0 else 2), "max_gap": None,
        "source_manifest_sha256": hashlib.sha256(source_manifest_bytes).hexdigest(),
    }
    if args.box_scale != 1.0:
        manifest["derivation"]["box_scale"] = args.box_scale
    policy = (center_repair_policy(json.loads(args.center_policy.read_text(encoding="utf-8")))
              if args.center_policy else (dict(DEFAULT_CENTER_REPAIR) if args.center_repair else None))
    if policy is not None:
        manifest["derivation"]["center_repair"] = policy
    dest_manifest = dest / "manifest.json"
    if dest_manifest.is_file() and json.loads(dest_manifest.read_text(encoding="utf-8")) != manifest:
        raise ValueError(f"Destination manifest differs: {dest_manifest}")
    dest.mkdir(parents=True, exist_ok=True)
    dest_manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    cropper = HandCropper(output_size=tuple(manifest["hand_image_size"]), padding_ratio=0.2)
    chosen = json.loads(args.video_ids_file.read_text(encoding="utf-8")) if args.video_ids_file else None
    if chosen is not None and (not isinstance(chosen, dict) or any(
        split not in {"train", "val"} or not isinstance(ids, list) or not ids
        or not all(isinstance(i, str) for i in ids) or len(set(ids)) != len(ids)
        for split, ids in chosen.items())):
        parser.error("Selection must map train/val to nonempty unique video ID lists")
    summary = {}
    for split in args.splits:
        dataset = TIDDataset(config.dataset, split=split)
        index_of = {sample.video_id: i for i, sample in enumerate(dataset.samples)}
        ids = chosen[split] if chosen else sorted(index_of)
        if not set(ids).issubset(index_of):
            raise ValueError(f"Unknown {split} IDs: {sorted(set(ids)-set(index_of))}")
        missing = [video_id for video_id in ids if not (source/split/f"{video_id}.npz").is_file()]
        if missing:
            raise FileNotFoundError(f"{split}: {len(missing)} source caches missing; first: {missing[:5]}")
        stats = {"videos": 0, "skipped_existing": 0, "frames": 0, "fallback_left": 0, "fallback_right": 0,
                 "alignment_checked": 0, "alignment_max_error": 0.0,
                 "center_repaired_left": 0, "center_repaired_right": 0}
        for video_id in tqdm(ids, desc=f"fallback {split}"):
            output = dest / split / f"{video_id}.npz"
            if output.is_file():
                stats["skipped_existing"] += 1
                continue
            with np.load(source / split / f"{video_id}.npz", allow_pickle=False) as data:
                features = dict(data)
            count = int(features["num_frames"])
            frames = sample_frames(str(dataset.get_video_path(index_of[video_id])), manifest["num_frames"])
            if len(frames) != count:
                raise ValueError(f"{video_id}: decoded {len(frames)} frames, cache has {count}")
            error = alignment_error(frames, features, count, cropper)
            if error is not None:
                stats["alignment_checked"] += 1
                stats["alignment_max_error"] = round(max(stats["alignment_max_error"], error), 3)
                if error > args.max_alignment_error:
                    raise ValueError(f"{video_id}: frame alignment error {error:.2f} px; aborting")
            for side in ("left", "right"):
                crops, src = fallback_crops(
                    frames, features[f"{side}_hand_crops"], features[f"{side}_hand_landmarks"],
                    features[f"{side}_hand_detected"].astype(bool), features["pose_landmarks"],
                    features["pose_detected"].astype(bool), side, box_scale=args.box_scale, center_repair=policy)
                features[f"{side}_hand_crops"] = crops
                features[f"{side}_crop_source"] = src
                stats[f"fallback_{side}"] += int(np.isin(src, [1, 3]).sum())
                stats[f"center_repaired_{side}"] += int((src == 3).sum())
                if policy is not None:
                    height, width = frames[0].shape[:2]
                    _, centers, boxes = fallback_geometry(features[f"{side}_hand_landmarks"],
                        features[f"{side}_hand_detected"], features["pose_landmarks"], features["pose_detected"],
                        side, width, height, box_scale=args.box_scale, center_repair=policy)
                    features[f"{side}_crop_center"] = centers
                    features[f"{side}_crop_box"] = boxes
                    features[f"{side}_crop_center_imputed"] = src == 3
            output.parent.mkdir(parents=True, exist_ok=True)
            temporary = output.with_suffix(".npz.part")
            with temporary.open("wb") as handle:
                np.savez_compressed(handle, **features)
            temporary.replace(output)
            stats["videos"] += 1
            stats["frames"] += count
        summary[split] = stats
    (dest / "fallback_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    outputs_dir = project_root / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    (outputs_dir / "autsl_100_fallback_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
