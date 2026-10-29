"""Audit paired AUTSL validation predictions against the official labels."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path


def load_predictions(path: Path) -> tuple[dict, dict[str, dict]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("split") != "val":
        raise ValueError(f"Only the official val split may be compared: {path}")
    records = data.get("predictions", [])
    by_id = {record["video_id"]: record for record in records}
    if len(by_id) != len(records):
        raise ValueError(f"Duplicate video IDs: {path}")
    return data, by_id


def official_subset(path: Path, class_ids: list[int]) -> dict[str, int]:
    class_to_index = {class_id: i for i, class_id in enumerate(class_ids)}
    expected = {}
    with path.open(newline="", encoding="utf-8") as source:
        for video_id, raw_class_id in csv.reader(source):
            original_id = int(raw_class_id)
            if original_id in class_to_index:
                expected[f"{video_id}_color"] = class_to_index[original_id]
    return expected


def exact_mcnemar_p(base_only: int, candidate_only: int) -> float:
    discordant = base_only + candidate_only
    if discordant == 0:
        return 1.0
    tail = sum(math.comb(discordant, k) for k in range(min(base_only, candidate_only) + 1))
    return min(1.0, 2.0 * tail / (2 ** discordant))


def analyze(base_path: Path, candidate_path: Path, labels_path: Path) -> dict:
    base_meta, base = load_predictions(base_path)
    candidate_meta, candidate = load_predictions(candidate_path)
    if base_meta.get("class_ids") != candidate_meta.get("class_ids"):
        raise ValueError("Class lists differ")
    class_ids = base_meta["class_ids"]
    expected = official_subset(labels_path, class_ids)
    if not expected or set(base) != set(expected) or set(candidate) != set(expected):
        raise ValueError("Prediction IDs do not match the official selected val subset")

    totals = dict(both_correct=0, base_only=0, candidate_only=0, both_wrong=0)
    examples = {key: [] for key in totals}
    per_class = defaultdict(lambda: dict(n=0, base_correct=0, candidate_correct=0))
    per_signer = defaultdict(lambda: dict(n=0, base_correct=0, candidate_correct=0))

    for video_id in sorted(expected):
        label = expected[video_id]
        first, second = base[video_id], candidate[video_id]
        if int(first["true_label"]) != label or int(second["true_label"]) != label:
            raise ValueError(f"Label mismatch: {video_id}")
        for record in (first, second):
            predicted = int(record["pred_label"])
            confidence = float(record["top1_confidence"])
            if predicted not in range(len(class_ids)) or not math.isfinite(confidence) or not 0 <= confidence <= 1:
                raise ValueError(f"Invalid prediction: {video_id}")
        base_correct = int(first["pred_label"]) == label
        candidate_correct = int(second["pred_label"]) == label
        key = (
            "both_correct" if base_correct and candidate_correct else
            "base_only" if base_correct else
            "candidate_only" if candidate_correct else
            "both_wrong"
        )
        totals[key] += 1
        examples[key].append(video_id)
        signer = video_id.split("_", 1)[0]
        for summary in (per_class[str(class_ids[label])], per_signer[signer]):
            summary["n"] += 1
            summary["base_correct"] += int(base_correct)
            summary["candidate_correct"] += int(candidate_correct)

    n = len(expected)
    return {
        "split": "official_autsl_val",
        "n": n,
        "class_ids": class_ids,
        "base_config": base_meta.get("config"),
        "candidate_config": candidate_meta.get("config"),
        "base_checkpoint_sha256": base_meta.get("checkpoint_sha256"),
        "candidate_checkpoint_sha256": candidate_meta.get("checkpoint_sha256"),
        "base_accuracy": (totals["both_correct"] + totals["base_only"]) / n,
        "candidate_accuracy": (totals["both_correct"] + totals["candidate_only"]) / n,
        "net_correct_videos": totals["candidate_only"] - totals["base_only"],
        "paired_counts": totals,
        "mcnemar_exact_two_sided_p_exploratory": exact_mcnemar_p(
            totals["base_only"], totals["candidate_only"]
        ),
        "examples": examples,
        "per_class_original_id": dict(sorted(per_class.items(), key=lambda x: int(x[0]))),
        "per_signer": dict(sorted(per_signer.items())),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--official_labels", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.base, args.candidate, args.official_labels)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: result[key] for key in (
        "n", "base_accuracy", "candidate_accuracy", "net_correct_videos",
        "paired_counts", "mcnemar_exact_two_sided_p_exploratory"
    )}, ensure_ascii=False))


if __name__ == "__main__":
    main()
