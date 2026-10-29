"""Create bounded 226-class train/val extraction batches after 100-class reuse.

The official split labels define the universe. Existing source-cache filenames
define the already extracted subset. No video or cache content is modified.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def official_ids(path: Path) -> set[str]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        rows = [(f"{video_id}_color", int(class_id))
                for video_id, class_id in csv.reader(stream)]
    ids = {video_id for video_id, _ in rows}
    if len(ids) != len(rows) or any(not 0 <= class_id < 226 for _, class_id in rows):
        raise ValueError(f"Invalid or duplicate official labels: {path}")
    return ids


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source_cache", type=Path, required=True)
    parser.add_argument("--labels_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--batch_size", type=int, default=2000)
    args = parser.parse_args()
    if args.batch_size < 1:
        raise ValueError("batch_size must be positive")
    manifest = json.loads((args.source_cache / "manifest.json").read_text(encoding="utf-8"))
    if (manifest.get("dataset") != "autsl" or manifest.get("num_classes") != 100
            or manifest.get("num_frames") != 16 or "derivation" in manifest):
        raise ValueError("Expected 100-class base16 source cache")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = {"source_cache": str(args.source_cache), "batch_size": args.batch_size,
              "splits": {}, "files": []}
    expected = {"train": (28142, 12405), "val": (4418, 1963)}
    for split, (expected_official, expected_source) in expected.items():
        all_ids = official_ids(args.labels_dir / f"{split}_labels.csv")
        existing_ids = {path.stem for path in (args.source_cache / split).glob("*.npz")}
        if len(all_ids) != expected_official or len(existing_ids) != expected_source:
            raise ValueError(f"Unexpected {split} counts: official={len(all_ids)}, source={len(existing_ids)}")
        if not existing_ids <= all_ids:
            raise ValueError(f"Source {split} cache contains nonofficial IDs")
        missing = sorted(all_ids - existing_ids)
        batch_names = []
        for start in range(0, len(missing), args.batch_size):
            name = f"{split}_{start // args.batch_size + 1:02d}.json"
            (args.output_dir / name).write_text(
                json.dumps({split: missing[start:start + args.batch_size]}, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8")
            batch_names.append(name)
            report["files"].append(name)
        report["splits"][split] = {"official": len(all_ids), "reusable": len(existing_ids),
                                    "remaining": len(missing), "batches": batch_names}
        print(f"{split}: reusable={len(existing_ids)}, remaining={len(missing)}, batches={len(batch_names)}")
    (args.output_dir / "manifest.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
