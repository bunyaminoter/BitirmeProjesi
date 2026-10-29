"""Select 40 diverse train videos for frame-level active/passive hand review.

This is a deterministic QA selection from numeric flags, not an error label.
"""

from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "outputs/autsl_100_drive_review/all_video_qa_flags.csv"
DEST = ROOT / "data/autsl/autsl_100_active_hand_annotation_queue.csv"


def take(rows: list[dict], amount: int, chosen: set[str], signer_cap: int,
         class_cap: int) -> list[dict]:
    out: list[dict] = []
    signers: Counter[str] = Counter()
    classes: Counter[str] = Counter()
    for row in rows:
        if row["video_id"] in chosen or signers[row["signer"]] >= signer_cap or classes[row["tr"]] >= class_cap:
            continue
        out.append(row)
        chosen.add(row["video_id"])
        signers[row["signer"]] += 1
        classes[row["tr"]] += 1
        if len(out) == amount:
            break
    if len(out) != amount:
        raise RuntimeError(f"Only {len(out)}/{amount} QA videos satisfy diversity constraints")
    return out


def main() -> None:
    with SOURCE.open(newline="", encoding="utf-8-sig") as handle:
        train = [row for row in csv.DictReader(handle) if row["split"] == "train"]
    train.sort(key=lambda row: (-int(row["triage_score"]), row["video_id"]))
    severe = [row for row in train if int(row["both_hands_detected_le4"])]
    single = [row for row in train if int(row["any_missing_run_ge9"])
              and not int(row["both_hands_detected_le4"])
              and (int(row["left_max_missing_run"]) >= 9) != (int(row["right_max_missing_run"]) >= 9)]
    controls = [row for row in train if int(row["left_detected"]) >= 15
                and int(row["right_detected"]) >= 15
                and not int(row["wrist_mismatch_frames_ge4"])]
    chosen: set[str] = set()
    groups = [
        ("both_hands_sparse", take(severe, 16, chosen, 5, 2)),
        ("one_side_long_gap", take(single, 16, chosen, 2, 2)),
        ("well_detected_control", take(controls, 8, chosen, 1, 1)),
    ]
    columns = ["selection_group", "video_id", "signer", "source_class_id", "tr",
               "left_detected", "right_detected", "left_max_missing_run", "right_max_missing_run",
               "active_missing_frames", "passive_only_loss", "wrong_side_or_duplicate",
               "overlap_or_occlusion", "visual_notes"]
    DEST.parent.mkdir(parents=True, exist_ok=True)
    with DEST.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for name, rows in groups:
            for row in rows:
                writer.writerow({**{key: row.get(key, "") for key in columns}, "selection_group": name,
                                 "active_missing_frames": "", "passive_only_loss": "",
                                 "wrong_side_or_duplicate": "", "overlap_or_occlusion": "",
                                 "visual_notes": ""})
    print(DEST)
    print({name: len(rows) for name, rows in groups})


if __name__ == "__main__":
    main()
