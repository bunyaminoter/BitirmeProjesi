"""Read-only cache/prediction audit for both-wrong AUTSL 100-class val clips.

Detection flags are quality proxies, not human labels or causal explanations.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


def longest_run(mask: np.ndarray) -> int:
    longest = current = 0
    for value in mask:
        current = current + 1 if value else 0
        longest = max(longest, current)
    return longest


def load(path: Path) -> tuple[dict, dict[str, dict]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("split") != "val":
        raise ValueError("Expected official val predictions")
    records = data["predictions"]
    by_id = {r["video_id"]: r for r in records}
    if len(by_id) != len(records):
        raise ValueError("Duplicate video IDs")
    return data, by_id


def analyze(pose_path: Path, hybrid_path: Path, cache_dir: Path) -> tuple[dict, list[dict]]:
    pose_data, pose = load(pose_path)
    hybrid_data, hybrid = load(hybrid_path)
    class_ids = pose_data["class_ids"]
    if class_ids != hybrid_data["class_ids"] or pose.keys() != hybrid.keys():
        raise ValueError("Class order/video IDs differ")
    rows = []
    missing = []
    for video_id in sorted(pose):
        p, h = pose[video_id], hybrid[video_id]
        label = int(p["true_label"])
        if label != int(h["true_label"]):
            raise ValueError(f"True label differs: {video_id}")
        p_ok = int(p["pred_label"]) == label
        h_ok = int(h["pred_label"]) == label
        status = ("both_correct" if p_ok and h_ok else "pose_only" if p_ok
                  else "hybrid_only" if h_ok else "both_wrong")
        path = cache_dir / "val" / f"{video_id}.npz"
        if not path.exists():
            missing.append(video_id)
            continue
        with np.load(path, allow_pickle=False) as d:
            left = d["left_hand_detected"].astype(bool)
            right = d["right_hand_detected"].astype(bool)
            pose_ok = d["pose_detected"].astype(bool)
            ls = d["left_crop_source"]
            rs = d["right_crop_source"]
            imputed = d["left_crop_center_imputed"].sum() + d["right_crop_center_imputed"].sum()
            cache_label = int(d["label"])
        if cache_label != label:
            raise ValueError(f"Cache label mismatch: {video_id}")
        if not (len(left) == len(right) == len(pose_ok)):
            raise ValueError(f"Frame count mismatch: {video_id}")
        p_probs = np.asarray(p["class_probabilities"])
        h_probs = np.asarray(h["class_probabilities"])
        rank_p = int(np.where(np.argsort(-p_probs) == label)[0][0]) + 1
        rank_h = int(np.where(np.argsort(-h_probs) == label)[0][0]) + 1
        sources = np.concatenate((ls, rs))
        rows.append({
            "video_id": video_id, "signer": video_id.split("_", 1)[0],
            "status": status, "true_original_id": class_ids[label],
            "pose_pred_original_id": class_ids[int(p["pred_label"])],
            "hybrid_pred_original_id": class_ids[int(h["pred_label"])],
            "same_wrong_prediction": int(not p_ok and not h_ok and p["pred_label"] == h["pred_label"]),
            "pose_true_rank": rank_p, "hybrid_true_rank": rank_h,
            "pose_top1_confidence": float(p["top1_confidence"]),
            "hybrid_top1_confidence": float(h["top1_confidence"]),
            "frames": len(left),
            "both_missing_frames": int((~left & ~right).sum()),
            "one_missing_frames": int((left ^ right).sum()),
            "longest_both_missing_run": longest_run(~left & ~right),
            "longest_any_hand_missing_run": max(longest_run(~left), longest_run(~right)),
            "pose_missing_frames": int((~pose_ok).sum()),
            "fallback_crops": int(np.isin(sources, [1, 3]).sum()),
            "repaired_crops": int((sources == 3).sum()),
            "imputed_crop_centers": int(imputed),
        })

    groups = defaultdict(list)
    for row in rows:
        groups[row["status"]].append(row)
    by_status = {}
    for status, group in sorted(groups.items()):
        n = len(group)
        by_status[status] = {
            "videos": n,
            "mean_both_missing_frames": sum(r["both_missing_frames"] for r in group) / n,
            "with_any_both_missing_frame": sum(r["both_missing_frames"] > 0 for r in group),
            "with_long_both_missing_run_3plus": sum(r["longest_both_missing_run"] >= 3 for r in group),
            "mean_fallback_crops": sum(r["fallback_crops"] for r in group) / n,
            "with_imputed_crop_center": sum(r["imputed_crop_centers"] > 0 for r in group),
        }
    shared = groups["both_wrong"]
    summary = {
        "scope": "AUTSL selected 100-class official val, read-only strict cache audit",
        "total_prediction_videos": len(pose),
        "cache_files_found": len(rows),
        "missing_local_cache_video_ids": missing,
        "by_status": by_status,
        "both_wrong_signers": dict(sorted(Counter(r["signer"] for r in shared).items())),
        "both_wrong_same_predicted_class": sum(r["same_wrong_prediction"] for r in shared),
        "both_wrong_true_rank_both_top5": sum(r["pose_true_rank"] <= 5 and r["hybrid_true_rank"] <= 5 for r in shared),
        "both_wrong_true_rank_neither_top5": sum(r["pose_true_rank"] > 5 and r["hybrid_true_rank"] > 5 for r in shared),
        "both_wrong_true_class_counts": dict(sorted(Counter(r["true_original_id"] for r in shared).items())),
    }
    return summary, rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pose", type=Path, required=True)
    parser.add_argument("--hybrid", type=Path, required=True)
    parser.add_argument("--cache_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args()
    summary, rows = analyze(args.pose, args.hybrid, args.cache_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with (args.output_dir / "videos.csv").open("w", newline="", encoding="utf-8-sig") as out:
        writer = csv.DictWriter(out, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
