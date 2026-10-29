"""Check completeness and non-destructive invariants of a derived ROI cache."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source_cache", type=Path, required=True)
    parser.add_argument("--derived_cache", type=Path, required=True)
    parser.add_argument("--splits", nargs="+", choices=["train", "val"], default=["train", "val"])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = {}
    for split in args.splits:
        source = {path.name for path in (args.source_cache / split).glob("*.npz")}
        derived = {path.name for path in (args.derived_cache / split).glob("*.npz")}
        if source != derived:
            raise ValueError(f"{split}: missing={len(source-derived)} extra={len(derived-source)}")
        count = Counter()
        signer = Counter()
        changed_ids = []
        for filename in sorted(source):
            with np.load(args.source_cache / split / filename, allow_pickle=False) as source_npz:
                base = dict(source_npz)
            with np.load(args.derived_cache / split / filename, allow_pickle=False) as derived_npz:
                candidate = dict(derived_npz)
            if not set(base).issubset(candidate):
                raise AssertionError(f"Missing source keys: {filename}")
            rescued = candidate["roi_retry_frame"].astype(bool)
            if rescued.shape != (int(base["num_frames"]),):
                raise AssertionError(f"Invalid rescue shape: {filename}")
            if np.any(rescued & (base["left_hand_detected"] | base["right_hand_detected"])):
                raise AssertionError(f"Overwrote source detection: {filename}")
            for key, value in base.items():
                other = candidate[key]
                if value.shape != other.shape or value.dtype != other.dtype:
                    raise AssertionError(f"Tensor shape/dtype changed: {filename} {key}")
                if value.ndim and value.shape[0] == len(rescued):
                    if not np.array_equal(value[~rescued], other[~rescued]):
                        raise AssertionError(f"Non-rescued frame changed: {filename} {key}")
                elif not np.array_equal(value, other):
                    raise AssertionError(f"Fixed field changed: {filename} {key}")
            for side in ("left", "right"):
                add = candidate[f"roi_rescued_{side}"].astype(bool)
                if np.any(add & ~rescued) or np.any(add & base[f"{side}_hand_detected"]):
                    raise AssertionError(f"Invalid side rescue: {filename} {side}")
                if not np.array_equal(candidate[f"{side}_hand_detected"],
                                      base[f"{side}_hand_detected"] | add):
                    raise AssertionError(f"Detection mask mismatch: {filename} {side}")
                count[f"{side}_rescued_slots"] += int(add.sum())
            count["videos"] += 1
            count["source_zero_hand_frames"] += int((~base["left_hand_detected"] & ~base["right_hand_detected"]).sum())
            count["rescued_frames"] += int(rescued.sum())
            if rescued.any():
                changed_ids.append(Path(filename).stem)
                signer[Path(filename).stem.split("_", 1)[0]] += int(rescued.sum())
        count["videos_with_rescue"] = len(changed_ids)
        summary[split] = {"totals": dict(count), "rescued_frames_by_signer": dict(signer),
                          "changed_video_ids": changed_ids}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({split: item["totals"] for split, item in summary.items()},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
