"""Audit a fixed 50/50 pose/RGB probability fusion on official AUTSL val.

The weight is deliberately fixed; this script never searches val for a weight.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import binomtest
from sklearn.metrics import f1_score


def load_predictions(path: Path) -> tuple[dict, dict[str, dict]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("split") != "val":
        raise ValueError(f"Expected official val predictions: {path}")
    records = data["predictions"]
    indexed = {record["video_id"]: record for record in records}
    if len(indexed) != len(records):
        raise ValueError(f"Duplicate video IDs: {path}")
    return data, indexed


def official_subset(path: Path, class_ids: list[int]) -> dict[str, int]:
    lookup = {original: index for index, original in enumerate(class_ids)}
    expected = {}
    with path.open(newline="", encoding="utf-8") as source:
        for video_id, raw_original in csv.reader(source):
            original = int(raw_original)
            if original in lookup:
                expected[f"{video_id}_color"] = lookup[original]
    return expected


def summarize(name: str, probabilities: np.ndarray, labels: np.ndarray) -> dict:
    predicted = probabilities.argmax(axis=1)
    top5 = np.argsort(-probabilities, axis=1)[:, :5]
    correct = predicted == labels
    return {
        "name": name,
        "correct": int(correct.sum()),
        "top1": float(correct.mean()),
        "top5_correct": int(np.any(top5 == labels[:, None], axis=1).sum()),
        "macro_f1": float(f1_score(labels, predicted, average="macro")),
        "wrong_with_confidence_at_least_0_9": int(
            ((~correct) & (probabilities.max(axis=1) >= 0.9)).sum()
        ),
    }


def paired_counts(base: np.ndarray, candidate: np.ndarray) -> dict:
    rescued = int((~base & candidate).sum())
    broken = int((base & ~candidate).sum())
    return {
        "rescued": rescued,
        "broken": broken,
        "net_gain": rescued - broken,
        "mcnemar_exact_two_sided_p_exploratory": float(
            binomtest(rescued, rescued + broken, 0.5).pvalue
        ) if rescued + broken else 1.0,
    }


def analyze(pose_path: Path, hybrid_path: Path, labels_path: Path) -> dict:
    pose_meta, pose_by_id = load_predictions(pose_path)
    hybrid_meta, hybrid_by_id = load_predictions(hybrid_path)
    class_ids = pose_meta["class_ids"]
    if class_ids != hybrid_meta["class_ids"]:
        raise ValueError("Class order mismatch")
    expected = official_subset(labels_path, class_ids)
    if set(expected) != set(pose_by_id) or set(expected) != set(hybrid_by_id):
        raise ValueError("Predictions do not match the official val subset")
    video_ids = sorted(expected)
    labels = np.array([expected[video_id] for video_id in video_ids])
    for by_id in (pose_by_id, hybrid_by_id):
        if any(int(by_id[video_id]["true_label"]) != expected[video_id]
               for video_id in video_ids):
            raise ValueError("True label mismatch")

    pose = np.array([pose_by_id[video_id]["class_probabilities"]
                     for video_id in video_ids], dtype=np.float64)
    hybrid = np.array([hybrid_by_id[video_id]["class_probabilities"]
                       for video_id in video_ids], dtype=np.float64)
    if pose.shape != hybrid.shape or pose.shape != (len(video_ids), len(class_ids)):
        raise ValueError("Probability shape mismatch")
    if not (np.isfinite(pose).all() and np.isfinite(hybrid).all()
            and np.allclose(pose.sum(axis=1), 1, atol=1e-3)
            and np.allclose(hybrid.sum(axis=1), 1, atol=1e-3)):
        raise ValueError("Invalid probability arrays")
    if any(int(pose_by_id[video_id]["pred_label"]) != int(pose[i].argmax())
           or int(hybrid_by_id[video_id]["pred_label"]) != int(hybrid[i].argmax())
           for i, video_id in enumerate(video_ids)):
        raise ValueError("Prediction/argmax mismatch")

    fused = (pose + hybrid) / 2
    pose_ok = pose.argmax(axis=1) == labels
    hybrid_ok = hybrid.argmax(axis=1) == labels
    fused_ok = fused.argmax(axis=1) == labels
    signers = defaultdict(lambda: {"n": 0, "pose_correct": 0,
                                   "hybrid_correct": 0, "fused_correct": 0})
    for i, video_id in enumerate(video_ids):
        row = signers[video_id.split("_", 1)[0]]
        row["n"] += 1
        row["pose_correct"] += int(pose_ok[i])
        row["hybrid_correct"] += int(hybrid_ok[i])
        row["fused_correct"] += int(fused_ok[i])

    return {
        "split": "official_autsl_val",
        "n": len(video_ids),
        "class_ids": class_ids,
        "pose_checkpoint_sha256_claim": pose_meta.get("checkpoint_sha256"),
        "hybrid_checkpoint_sha256_claim": hybrid_meta.get("checkpoint_sha256"),
        "fusion_method": "fixed 0.5 * pose probability + 0.5 * hybrid probability",
        "pose": summarize("pose", pose, labels),
        "hybrid": summarize("hybrid", hybrid, labels),
        "fusion": summarize("fusion", fused, labels),
        "hybrid_vs_pose": paired_counts(pose_ok, hybrid_ok),
        "fusion_vs_pose": paired_counts(pose_ok, fused_ok),
        "fusion_vs_hybrid": paired_counts(hybrid_ok, fused_ok),
        "both_models_wrong": int((~pose_ok & ~hybrid_ok).sum()),
        "both_models_wrong_but_fusion_correct": int((~pose_ok & ~hybrid_ok & fused_ok).sum()),
        "per_signer": dict(sorted(signers.items())),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pose", type=Path, required=True)
    parser.add_argument("--hybrid", type=Path, required=True)
    parser.add_argument("--official_labels", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.pose, args.hybrid, args.official_labels)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                           encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("pose", "hybrid", "fusion",
                                             "hybrid_vs_pose", "fusion_vs_pose")},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
