"""Build a separate AUTSL cache with conservative short hand-gap repair."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.preprocessing.gap_repair import repair_hand_gaps


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source_cache", type=Path,
                        default=project_root / "cache/autsl_10_features")
    parser.add_argument("--dest_cache", type=Path,
                        default=project_root / "cache/autsl_10_features_gap2_pilot")
    parser.add_argument("--video_ids_file", type=Path, default=None)
    parser.add_argument("--splits", nargs="+", choices=["train", "val"], default=["train", "val"])
    parser.add_argument("--max_gap", type=int, default=2)
    args = parser.parse_args()
    source, dest = args.source_cache.resolve(), args.dest_cache.resolve()
    if source == dest or source in dest.parents or dest in source.parents:
        parser.error("Source and destination caches must be separate sibling directories")
    if args.max_gap < 1:
        parser.error("--max_gap must be positive")
    source_manifest_bytes = (source / "manifest.json").read_bytes()
    manifest = json.loads(source_manifest_bytes)
    if manifest.get("dataset") != "autsl" or manifest.get("derivation"):
        parser.error("Source must be an unmodified AUTSL cache")
    manifest["derivation"] = {
        "method": "short_hand_gap_linear",
        "version": 1,
        "max_gap": args.max_gap,
        "source_manifest_sha256": hashlib.sha256(source_manifest_bytes).hexdigest(),
    }
    dest_manifest = dest / "manifest.json"
    if dest_manifest.is_file() and json.loads(dest_manifest.read_text(encoding="utf-8")) != manifest:
        raise ValueError(f"Destination manifest differs: {dest_manifest}")
    dest.mkdir(parents=True, exist_ok=True)
    dest_manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    chosen = json.loads(args.video_ids_file.read_text(encoding="utf-8")) if args.video_ids_file else None
    summary = {}
    for split in args.splits:
        ids = chosen[split] if chosen else [path.stem for path in sorted((source / split).glob("*.npz"))]
        stats = {"videos": 0, "observed_left": 0, "observed_right": 0,
                 "imputed_left": 0, "imputed_right": 0}
        for video_id in ids:
            path = source / split / f"{video_id}.npz"
            if not path.is_file():
                raise FileNotFoundError(path)
            with np.load(path, allow_pickle=False) as data:
                features = dict(data)
            count = int(features["num_frames"])
            for side, own, other in (("left", 15, 16), ("right", 16, 15)):
                observed = features[f"{side}_hand_detected"][:count]
                repaired, imputed = repair_hand_gaps(
                    features[f"{side}_hand_landmarks"][:count], observed,
                    features["pose_landmarks"][:count], features["pose_detected"][:count],
                    wrist_index=own, other_wrist_index=other, max_gap=args.max_gap,
                )
                features[f"{side}_hand_landmarks"] = repaired
                features[f"{side}_hand_imputed"] = imputed
                stats[f"observed_{side}"] += int(observed.sum())
                stats[f"imputed_{side}"] += int(imputed.sum())
            output = dest / split / f"{video_id}.npz"
            output.parent.mkdir(parents=True, exist_ok=True)
            temporary = output.with_suffix(".npz.part")
            with temporary.open("wb") as handle:
                np.savez_compressed(handle, **features)
            temporary.replace(output)
            stats["videos"] += 1
        summary[split] = stats
    summary_path = dest / "repair_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
