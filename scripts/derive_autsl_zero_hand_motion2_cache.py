"""Keep only ROI rescues that persist for >=2 adjacent sampled frames."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
from tqdm import tqdm

from audit_autsl_zero_hand_motion_gate import keep_runs

METHOD = "zero_hand_upperbody_roi_motion2"
ALLOWED = {f"{side}_{suffix}" for side in ("left", "right") for suffix in (
    "hand_landmarks", "hand_crops", "hand_detected", "crop_source", "crop_center",
    "crop_box", "crop_center_imputed")}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source_cache", type=Path, required=True)
    parser.add_argument("--roi_cache", type=Path, required=True)
    parser.add_argument("--dest_cache", type=Path, required=True)
    parser.add_argument("--splits", nargs="+", choices=["train", "val"], default=["train", "val"])
    args = parser.parse_args()
    paths = [args.source_cache.resolve(), args.roi_cache.resolve(), args.dest_cache.resolve()]
    if len(set(paths)) != 3 or any(a in b.parents for a in paths for b in paths if a != b):
        parser.error("All caches must be separate siblings")
    source_bytes = (paths[0] / "manifest.json").read_bytes()
    roi_bytes = (paths[1] / "manifest.json").read_bytes()
    source_manifest = json.loads(source_bytes)
    roi_manifest = json.loads(roi_bytes)
    if roi_manifest.get("derivation", {}).get("method") != "zero_hand_upperbody_roi":
        parser.error("ROI cache must be version-1 zero-hand retry")
    for key in ("dataset", "pipeline_version", "num_frames", "hand_image_size", "num_classes", "class_ids"):
        if source_manifest.get(key) != roi_manifest.get(key):
            parser.error(f"Source/ROI manifest differs at {key}")
    manifest = dict(source_manifest)
    manifest["derivation"] = {
        "method": METHOD, "version": 1, "max_gap": None,
        "source_manifest_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "roi_manifest_sha256": hashlib.sha256(roi_bytes).hexdigest(),
        "minimum_adjacent_sampled_frames": 2,
    }
    dest = paths[2]
    dest.mkdir(parents=True, exist_ok=True)
    manifest_path = dest / "manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text(encoding="utf-8")) != manifest:
        raise ValueError("Destination manifest differs")
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = {}
    for split in args.splits:
        source = {path.name for path in (paths[0] / split).glob("*.npz")}
        candidate = {path.name for path in (paths[1] / split).glob("*.npz")}
        if source != candidate:
            raise ValueError(f"{split}: source and ROI cache file sets differ")
        counts = Counter()
        for filename in tqdm(sorted(source), desc=f"motion2 {split}"):
            output = dest / split / filename
            if output.exists():
                counts["skipped_existing"] += 1
                continue
            with np.load(paths[0] / split / filename, allow_pickle=False) as npz:
                base = dict(npz)
            with np.load(paths[1] / split / filename, allow_pickle=False) as npz:
                roi = dict(npz)
            keep = keep_runs(roi["roi_retry_frame"].astype(bool))
            derived = {key: value.copy() for key, value in base.items()}
            for key in ALLOWED & set(derived):
                derived[key][keep] = roi[key][keep]
            derived["roi_retry_frame"] = keep
            for side in ("left", "right"):
                flag = np.zeros_like(keep)
                flag[keep] = roi[f"roi_rescued_{side}"][keep]
                derived[f"roi_rescued_{side}"] = flag
                counts[f"{side}_rescued_slots"] += int(flag.sum())
            output.parent.mkdir(parents=True, exist_ok=True)
            temporary = output.with_suffix(".npz.part")
            with temporary.open("wb") as handle:
                np.savez_compressed(handle, **derived)
            temporary.replace(output)
            counts["videos"] += 1
            counts["rescued_frames"] += int(keep.sum())
        summary[split] = dict(counts)
    (dest / "motion2_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
