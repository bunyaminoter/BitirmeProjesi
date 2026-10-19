"""Audit the official 226-class AUTSL split before preprocessing or training."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.core.config import DatasetConfig, load_config
from src.data.datasets.tid_dataset import TIDDataset


EXPECTED = {"train": 28142, "val": 4418, "test": 3742}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_dir", type=Path, default=None)
    parser.add_argument("--config", type=Path, default=None,
                        help="AUTSL deney yapılandırması; 10 sınıf alt kümesini de denetler.")
    parser.add_argument("--cache_dir", type=Path, default=None)
    parser.add_argument("--require_cache", action="store_true")
    parser.add_argument("--inspect_features", action="store_true",
                        help="Read cached NPZ files and summarize landmark detection rates")
    parser.add_argument("--output", type=Path, default=Path("outputs/autsl_audit.json"))
    args = parser.parse_args()
    if (args.require_cache or args.inspect_features) and args.cache_dir is None:
        parser.error("--require_cache and --inspect_features need --cache_dir")
    if args.data_dir:
        os.environ["AUTSL_DATA_DIR"] = str(args.data_dir.resolve())

    if args.config:
        config = load_config(str(args.config)).dataset
        if config.name != "autsl":
            parser.error("--config must use dataset.name: autsl")
    else:
        config = DatasetConfig(name="autsl", annotation_file="data/autsl", data_dir="data/autsl", num_classes=226)
    full_dataset = config.class_ids is None
    ids: dict[str, set[str]] = {}
    signers: dict[str, set[str]] = {}
    result: dict = {"dataset_root": "", "splits": {}, "errors": []}
    for split, expected in EXPECTED.items():
        dataset = TIDDataset(config, split)
        ids[split] = {sample.video_id for sample in dataset.samples}
        signers[split] = {sample.signer_id for sample in dataset.samples}
        labels = Counter(sample.label for sample in dataset.samples)
        missing_videos = [sample.video_id for i, sample in enumerate(dataset.samples)
                          if not dataset.get_video_path(i).is_file()]
        cache_count = None
        missing_cache = []
        frames_checked = 0
        detections = Counter()
        wrist_match = Counter()
        wrist_frames = Counter()
        feature_errors = []
        if args.cache_dir:
            cache_count = 0
            for sample in dataset.samples:
                path = args.cache_dir / split / f"{sample.video_id}.npz"
                if path.is_file():
                    cache_count += 1
                    if args.inspect_features:
                        try:
                            with np.load(path, allow_pickle=False) as features:
                                count = int(features["num_frames"])
                                if int(features["label"]) != sample.label:
                                    raise ValueError("label mismatch")
                                for key in ("pose_detected", "left_hand_detected", "right_hand_detected"):
                                    values = features[key]
                                    if len(values) != count:
                                        raise ValueError(f"{key} length mismatch")
                                    detections[key] += int(values.sum())
                                pose = features["pose_landmarks"]
                                for side, pose_wrist, other_wrist in (
                                    ("left", 15, 16), ("right", 16, 15)
                                ):
                                    valid = features["pose_detected"] & features[f"{side}_hand_detected"]
                                    hand_wrist = features[f"{side}_hand_landmarks"][valid, 0, :2]
                                    own = np.linalg.norm(hand_wrist - pose[valid, pose_wrist, :2], axis=1)
                                    other = np.linalg.norm(hand_wrist - pose[valid, other_wrist, :2], axis=1)
                                    wrist_match[side] += int((own < other).sum())
                                    wrist_frames[side] += int(valid.sum())
                                frames_checked += count
                        except (OSError, KeyError, ValueError) as exc:
                            feature_errors.append(f"{sample.video_id}: {exc}")
                else:
                    missing_cache.append(sample.video_id)
        result["dataset_root"] = str(dataset._official_root)
        result["splits"][split] = {
            "labels": len(dataset), "expected_labels": expected if full_dataset else None,
            "classes": len(labels), "signers": sorted(signers[split]),
            "min_class_count": min(labels.values()), "max_class_count": max(labels.values()),
            "missing_videos": len(missing_videos),
            "missing_video_examples": missing_videos[:5],
            "cached": cache_count,
            "missing_cache": len(missing_cache) if args.cache_dir else None,
            "missing_cache_examples": missing_cache[:5],
            "frames_checked": frames_checked if args.inspect_features else None,
            "pose_detected_rate": round(detections["pose_detected"] / frames_checked, 4) if frames_checked else None,
            "left_hand_detected_rate": round(detections["left_hand_detected"] / frames_checked, 4) if frames_checked else None,
            "right_hand_detected_rate": round(detections["right_hand_detected"] / frames_checked, 4) if frames_checked else None,
            "left_wrist_agreement": round(wrist_match["left"] / wrist_frames["left"], 4) if wrist_frames["left"] else None,
            "right_wrist_agreement": round(wrist_match["right"] / wrist_frames["right"], 4) if wrist_frames["right"] else None,
            "feature_errors": feature_errors[:5],
        }
        if (full_dataset and (len(dataset) != expected or len(labels) != 226)) or missing_videos:
            result["errors"].append(f"{split}: incomplete labels, classes, or RGB files")
        if args.require_cache and missing_cache:
            result["errors"].append(f"{split}: {len(missing_cache)} feature caches missing")
        if feature_errors:
            result["errors"].append(f"{split}: {len(feature_errors)} invalid feature caches")
        for side in ("left", "right"):
            if wrist_frames[side] >= 10 and wrist_match[side] / wrist_frames[side] < 0.5:
                result["errors"].append(f"{split}: {side} hand landmarks appear swapped relative to pose")

    for first, second in (("train", "val"), ("train", "test"), ("val", "test")):
        if ids[first] & ids[second] or signers[first] & signers[second]:
            result["errors"].append(f"{first}/{second}: shared video IDs or signers")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result["errors"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
