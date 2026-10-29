"""Measure whether simple cache quality signals separate RGB saves from breaks.

Exploratory validation-only diagnosis. This does not train or select a router.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def auc(positives: list[int], negatives: list[int]) -> float:
    wins = 0.0
    for positive in positives:
        for negative in negatives:
            wins += 1.0 if positive > negative else 0.5 if positive == negative else 0.0
    return wins / (len(positives) * len(negatives))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--videos", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with args.videos.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    # In the shared-error audit: pose_only = hybrid broke a correct pose result;
    # hybrid_only = hybrid saved an incorrect pose result.
    broken = [r for r in rows if r["status"] == "pose_only"]
    saved = [r for r in rows if r["status"] == "hybrid_only"]
    if not broken or not saved:
        raise ValueError("Expected both saved and broken paired predictions")
    result = {
        "scope": "exploratory 100-class official val; local strict NPZ available videos only",
        "broken": len(broken), "saved": len(saved),
        "positive_class": "hybrid broke pose-only correct prediction",
        "signals": {},
    }
    for key in ("both_missing_frames", "fallback_crops", "longest_both_missing_run"):
        positive = [int(r[key]) for r in broken]
        negative = [int(r[key]) for r in saved]
        result["signals"][key] = {
            "roc_auc": auc(positive, negative),
            "broken_mean": sum(positive) / len(positive),
            "saved_mean": sum(negative) / len(negative),
            "broken_zero": positive.count(0),
            "saved_zero": negative.count(0),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                           encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
