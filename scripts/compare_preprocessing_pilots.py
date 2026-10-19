"""Compare matched 16-frame, repaired 16-frame, and 36-frame AUTSL pilots."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent


def metrics(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as data:
        count = int(data["num_frames"])
        result = {"frames": count, "label": int(data["label"]),
                  "pose_detected": int(data["pose_detected"].sum())}
        for side, own_idx, other_idx in (("left", 15, 16), ("right", 16, 15)):
            observed = data[f"{side}_hand_detected"][:count].astype(bool)
            missing = "".join("1" if not value else "0" for value in observed)
            result[f"{side}_observed"] = int(observed.sum())
            result[f"{side}_observed_rate"] = round(float(observed.mean()), 4)
            result[f"{side}_max_missing_run"] = max(map(len, missing.split("0")))
            valid = observed & data["pose_detected"][:count]
            hand_wrist = data[f"{side}_hand_landmarks"][valid, 0, :2]
            pose = data["pose_landmarks"][valid]
            own = np.linalg.norm(hand_wrist - pose[:, own_idx, :2], axis=1)
            other = np.linalg.norm(hand_wrist - pose[:, other_idx, :2], axis=1)
            result[f"{side}_wrist_agreement"] = round(float((own < other).mean()), 4) if len(own) else None
            if f"{side}_hand_imputed" in data:
                imputed = data[f"{side}_hand_imputed"][:count].astype(bool)
                result[f"{side}_imputed"] = int(imputed.sum())
                if imputed.any():
                    inferred = data[f"{side}_hand_landmarks"][imputed]
                    pose_wrist = data["pose_landmarks"][imputed, own_idx, :2]
                    distance = np.linalg.norm(inferred[:, 0, :2] - pose_wrist, axis=1)
                    result[f"{side}_imputed_max_wrist_distance"] = round(float(distance.max()), 4)
                    result[f"{side}_imputed_out_of_frame_points"] = int(
                        ((inferred[:, :, :2] < 0) | (inferred[:, :, :2] > 1)).any(axis=2).sum()
                    )
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path,
                        default=ROOT / "configs/experiment/autsl_10_preprocessing_pilot.json")
    parser.add_argument("--cache16", type=Path, default=ROOT / "cache/autsl_10_features")
    parser.add_argument("--cache16_repaired", type=Path,
                        default=ROOT / "cache/autsl_10_features_gap2_pilot")
    parser.add_argument("--cache36", type=Path,
                        default=ROOT / "cache/autsl_10_features_36_pilot")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "outputs/preprocessing_pilot_comparison.json")
    args = parser.parse_args()
    selection = json.loads(args.selection.read_text(encoding="utf-8"))
    caches = {"base16": args.cache16, "gap2_16": args.cache16_repaired, "base36": args.cache36}
    for name, directory in caches.items():
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        expected = 36 if name == "base36" else 16
        if manifest.get("dataset") != "autsl" or manifest.get("num_frames") != expected:
            raise ValueError(f"Unexpected manifest in {directory}")
        if name == "gap2_16" and manifest.get("derivation", {}).get("method") != "short_hand_gap_linear":
            raise ValueError(f"Not a gap-repaired cache: {directory}")
    records = []
    for split in ("train", "val"):
        for video_id in selection[split]:
            record = {"split": split, "video_id": video_id}
            for name, directory in caches.items():
                record[name] = metrics(directory / split / f"{video_id}.npz")
            if len({record[name]["label"] for name in caches}) != 1:
                raise ValueError(f"Label mismatch for {video_id}")
            records.append(record)
    summary = {}
    for name in caches:
        count = sum(record[name]["frames"] for record in records)
        observed = {side: sum(record[name][f"{side}_observed"] for record in records)
                    for side in ("left", "right")}
        summary[name] = {
            "videos": len(records), "frames": count,
            "pose_detected_rate": round(sum(record[name]["pose_detected"] for record in records) / count, 4),
            **{f"{side}_observed_rate": round(observed[side] / count, 4) for side in observed},
            "both_hands_observed_rate_mean": round(sum(observed.values()) / (2 * count), 4),
        }
        if name == "gap2_16":
            summary[name]["imputed_left"] = sum(record[name]["left_imputed"] for record in records)
            summary[name]["imputed_right"] = sum(record[name]["right_imputed"] for record in records)
    result = {"selection": str(args.selection), "summary": summary, "videos": records}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Per-video results: {args.output}")


if __name__ == "__main__":
    main()
