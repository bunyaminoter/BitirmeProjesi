"""Rank AUTSL preprocessing anomalies without training or predictions.

Scores are triage heuristics, not proof that a finger or wrist is incorrect.
Only official train/val cache files are opened; raw-video review is still needed.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.core.config import load_config  # noqa: E402
from src.data.datasets.tid_dataset import TIDDataset  # noqa: E402

FINGERTIPS = (4, 8, 12, 16, 20)


def longest_false_run(mask: np.ndarray) -> int:
    longest = current = 0
    for value in mask:
        current = 0 if value else current + 1
        longest = max(longest, current)
    return longest


def inspect_file(path: Path, expected_label: int, source_id: int,
                 signer: str, split: str, max_frames: int) -> dict:
    with np.load(path, allow_pickle=False) as data:
        count = int(data["num_frames"])
        if not 1 <= count <= max_frames or int(data["label"]) != expected_label:
            raise ValueError(f"Frame count or label mismatch: {path}")
        pose = data["pose_landmarks"]
        pose_ok = data["pose_detected"].astype(bool)
        if pose.shape != (count, 33, 3) or pose_ok.shape != (count,):
            raise ValueError(f"Pose shape mismatch: {path}")
        record = {"split": split, "video_id": path.stem, "signer": signer,
                  "model_label": expected_label, "source_class_id": source_id,
                  "frames": count}
        score = 0
        for side, own, other in (("left", 15, 16), ("right", 16, 15)):
            hand = data[f"{side}_hand_landmarks"]
            observed = data[f"{side}_hand_detected"].astype(bool)
            source = data[f"{side}_crop_source"]
            if hand.shape != (count, 21, 3) or observed.shape != (count,) or source.shape != (count,):
                raise ValueError(f"{side} shape mismatch: {path}")
            if not np.isin(source, (0, 1, 2, 3)).all():
                raise ValueError(f"Invalid {side} crop source: {path}")
            if not np.isfinite(hand[observed]).all():
                raise ValueError(f"Nonfinite detected {side} landmarks: {path}")
            missing_run = longest_false_run(observed)
            valid = observed & pose_ok
            own_distance = np.linalg.norm(hand[valid, 0, :2] - pose[valid, own, :2], axis=1)
            other_distance = np.linalg.norm(hand[valid, 0, :2] - pose[valid, other, :2], axis=1)
            wrong_wrist = int((other_distance + 0.05 < own_distance).sum())
            adjacent = observed[1:] & observed[:-1]
            wrist_step = np.linalg.norm(np.diff(hand[:, 0, :2], axis=0), axis=1)
            wrist_jumps = int(((wrist_step > 0.25) & adjacent).sum())
            relative_tips = hand[:, FINGERTIPS, :2] - hand[:, 0:1, :2]
            finger_step = np.linalg.norm(np.diff(relative_tips, axis=0), axis=2).max(axis=1)
            finger_jumps = int(((finger_step > 0.25) & adjacent).sum())
            stale = int((source == 2).sum())
            record.update({f"{side}_detected": int(observed.sum()),
                           f"{side}_max_missing_run": missing_run,
                           f"{side}_wrist_mismatch_frames": wrong_wrist,
                           f"{side}_wrist_jump_flags": wrist_jumps,
                           f"{side}_finger_jump_flags": finger_jumps,
                           f"{side}_unchanged_crop_frames": stale,
                           f"{side}_fallback_frames": int(np.isin(source, (1, 3)).sum())})
            score += 4 * wrong_wrist + 3 * wrist_jumps + 2 * finger_jumps
            score += 2 * missing_run + 2 * stale
        record["triage_score"] = score
        return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--cache_dir", type=Path, required=True)
    parser.add_argument("--data_dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--top", type=int, default=200)
    args = parser.parse_args()
    if args.data_dir:
        os.environ["AUTSL_DATA_DIR"] = str(args.data_dir.resolve())
    config = load_config(str(args.config))
    if config.dataset.name != "autsl" or config.dataset.class_ids is None:
        parser.error("An AUTSL subset config with explicit class_ids is required")
    manifest = json.loads((args.cache_dir / "manifest.json").read_text(encoding="utf-8"))
    if (manifest.get("dataset") != "autsl" or manifest.get("class_ids") != config.dataset.class_ids
            or manifest.get("num_classes") != config.dataset.num_classes
            or manifest.get("num_frames") != config.dataset.num_frames
            or manifest.get("derivation", {}).get("version") != 3):
        raise ValueError("Strict cache manifest/config mismatch")

    records, errors, coverage = [], [], {}
    for split in ("train", "val"):
        dataset = TIDDataset(config.dataset, split)
        expected = {sample.video_id: sample for sample in dataset.samples}
        actual = {path.stem: path for path in (args.cache_dir / split).glob("*.npz")}
        missing, extra = sorted(expected.keys() - actual.keys()), sorted(actual.keys() - expected.keys())
        coverage[split] = {"expected": len(expected), "found": len(actual),
                           "missing_count": len(missing), "missing_examples": missing[:20],
                           "extra_count": len(extra), "extra_examples": extra[:20]}
        for video_id in sorted(expected.keys() & actual.keys()):
            sample = expected[video_id]
            try:
                records.append(inspect_file(actual[video_id], sample.label,
                                            config.dataset.class_ids[sample.label],
                                            sample.signer_id, split, config.dataset.num_frames))
            except (OSError, EOFError, zipfile.BadZipFile, KeyError, ValueError, TypeError) as exc:
                errors.append(f"{split}/{video_id}: {exc}")
    records.sort(key=lambda item: (-item["triage_score"], item["split"], item["video_id"]))
    flag_totals = Counter()
    for record in records:
        for side in ("left", "right"):
            for key in ("max_missing_run", "wrist_mismatch_frames", "wrist_jump_flags",
                        "finger_jump_flags", "unchanged_crop_frames"):
                flag_totals[f"{side}_{key}"] += record[f"{side}_{key}"]
    report = {"config": str(args.config), "cache": str(args.cache_dir),
              "coverage": coverage, "invalid_count": len(errors), "invalid_examples": errors[:50],
              "flag_totals": dict(flag_totals), "heuristic_thresholds": {
                  "wrist_opposite_closer_margin": 0.05, "adjacent_wrist_step": 0.25,
                  "adjacent_wrist_relative_fingertip_step": 0.25},
              "top": records[:args.top], "all_videos": records}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"coverage": coverage, "invalid_count": len(errors),
                      "review_candidates": min(args.top, len(records)),
                      "output": str(args.output)}, ensure_ascii=False, indent=2))
    if any(item["missing_count"] or item["extra_count"] for item in coverage.values()) or errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
