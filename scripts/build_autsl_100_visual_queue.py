"""Build a class-balanced visual QA queue from the read-only motion audit.

The score is a review priority, never a correctness label. Train and val are
kept separate so fixes can be designed on train before independent val QA.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    audit = json.loads(args.audit.read_text(encoding="utf-8"))
    selection = json.loads(args.selection.read_text(encoding="utf-8"))
    classes = {int(c["model_index"]): c for c in selection["classes"]}
    by_group = defaultdict(list)
    for record in audit["all_videos"]:
        key = (record["split"], int(record["model_label"]))
        by_group[key].append(record)

    rows = []
    for split, per_class in (("train", 3), ("val", 1)):
        for model_label in sorted(classes):
            candidates = sorted(
                by_group[(split, model_label)],
                key=lambda r: (
                    -int(r["triage_score"]),
                    int(r["left_detected"]) + int(r["right_detected"]),
                    r["video_id"],
                ),
            )
            used_signers = set()
            selected = []
            for record in candidates:
                if record["signer"] not in used_signers:
                    selected.append(record)
                    used_signers.add(record["signer"])
                    if len(selected) == per_class:
                        break
            for record in candidates:
                if len(selected) == per_class:
                    break
                if record not in selected:
                    selected.append(record)
            if len(selected) != per_class:
                raise ValueError(f"Too few clips for {split}, class {model_label}")
            for rank, record in enumerate(selected, start=1):
                rows.append({
                    "split": split,
                    "model_label": model_label,
                    "source_class_id": classes[model_label]["source_class_id"],
                    "tr": classes[model_label]["tr"],
                    "rank": rank,
                    "video_id": record["video_id"],
                    "signer": record["signer"],
                    "frames": record["frames"],
                    "triage_score": record["triage_score"],
                    "left_detected": record["left_detected"],
                    "right_detected": record["right_detected"],
                    "left_max_missing_run": record["left_max_missing_run"],
                    "right_max_missing_run": record["right_max_missing_run"],
                    "left_wrist_mismatch_frames": record["left_wrist_mismatch_frames"],
                    "right_wrist_mismatch_frames": record["right_wrist_mismatch_frames"],
                    "left_finger_jump_flags": record["left_finger_jump_flags"],
                    "right_finger_jump_flags": record["right_finger_jump_flags"],
                    "visual_label": "",
                    "notes": "",
                })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} clips ({300} train, {100} val): {args.output}")


if __name__ == "__main__":
    main()
