"""Compare zero-hand ROI rescues with a >=2 adjacent sampled-frame gate."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


def keep_runs(mask: np.ndarray, minimum: int = 2) -> np.ndarray:
    keep = np.zeros_like(mask, dtype=bool)
    start = 0
    while start < len(mask):
        if not mask[start]:
            start += 1
            continue
        end = start
        while end < len(mask) and mask[end]:
            end += 1
        if end - start >= minimum:
            keep[start:end] = True
        start = end
    return keep


def read(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot_cache", type=Path, required=True)
    parser.add_argument("--ten_cache", type=Path, required=True)
    parser.add_argument("--dense_visual", type=Path, required=True)
    parser.add_argument("--roi_queue", type=Path, required=True)
    parser.add_argument("--roi_only_visual", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    dense = read(args.dense_visual)
    roi_only = read(args.roi_only_visual)
    only_by_key = {(r["video_id"], r["frame_1based"], r["roi_index"]): r["visual_sign_role"]
                   for r in roi_only}
    roles = defaultdict(set)
    for row in read(args.roi_queue):
        role = (dense[int(row["dense_candidate_number"]) - 1]["visual_sign_role"]
                if row["spatial_match_dense"] == "1" else
                only_by_key[(row["video_id"], row["frame_1based"], row["roi_index"])])
        roles[(row["video_id"], int(row["frame_1based"]))].add(role)
    pilot = Counter()
    for path in (args.pilot_cache / "train").glob("*.npz"):
        with np.load(path, allow_pickle=False) as npz:
            rescue = npz["roi_retry_frame"].astype(bool)
        keep = keep_runs(rescue)
        for t in np.flatnonzero(rescue):
            role_set = roles[(path.stem, t + 1)]
            role = next((r for r in ("active", "passive", "neutral", "uncertain") if r in role_set), "unreviewed")
            pilot[f"all_{role}"] += 1
            if keep[t]:
                pilot[f"kept_{role}"] += 1
            else:
                pilot[f"removed_{role}"] += 1
    ten = {}
    for split in ("train", "val"):
        counts = Counter()
        removed_ids = []
        for path in (args.ten_cache / split).glob("*.npz"):
            with np.load(path, allow_pickle=False) as npz:
                rescue = npz["roi_retry_frame"].astype(bool)
            keep = keep_runs(rescue)
            counts["videos"] += 1
            counts["all_rescued_frames"] += int(rescue.sum())
            counts["kept_rescued_frames"] += int(keep.sum())
            counts["removed_singleton_frames"] += int((rescue & ~keep).sum())
            if (rescue & ~keep).any():
                removed_ids.append(path.stem)
        ten[split] = {"totals": dict(counts), "videos_with_removed_singletons": removed_ids}
    result = {"pilot_visual_roles": dict(pilot), "ten_class": ten,
              "note": "Adjacent-frame gate is defined on existing ROI rescue masks, not tuned on val predictions."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"pilot_visual_roles": dict(pilot), "ten_class": {
        k: v["totals"] for k, v in ten.items()}}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
