"""Record the 130 ROI-only contact-sheet role judgements (2026-10-22).

Role is a coarse visual judgement from 300 px panels, not a verified 21-point
landmark or anatomical-side annotation. Unclear cases remain uncertain.
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path


def expand(spec: str) -> set[int]:
    values = set()
    for chunk in spec.split(","):
        ends = chunk.split("-")
        values.update(range(int(ends[0]), int(ends[-1]) + 1))
    return values


ACTIVE = expand("8-11,20-25,29,32-33,36,39-40,44-46,54-55,57,59-70,72-73,75-81,86-88,91-100,102,110-111,125-129")
PASSIVE = expand("89-90,105-109,114-121")
UNCERTAIN = expand("3-4,41,104")
ALL = set(range(1, 131))
NEUTRAL = ALL - ACTIVE - PASSIVE - UNCERTAIN


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if ACTIVE & PASSIVE or ACTIVE & UNCERTAIN or PASSIVE & UNCERTAIN:
        raise ValueError("Overlapping manual role sets")
    with args.index.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 130 or {int(row["candidate_number"]) for row in rows} != ALL:
        raise ValueError("Contact sheet index changed: re-review required")
    for row in rows:
        number = int(row["candidate_number"])
        role = ("active" if number in ACTIVE else "passive" if number in PASSIVE
                else "uncertain" if number in UNCERTAIN else "neutral")
        row["visual_sign_role"] = role
        row["visual_anatomical_side"] = "uncertain"
        row["visual_21point_geometry"] = "unverified"
        row["review_note"] = "300px_contact_sheet_role_only"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(dict(Counter(row["visual_sign_role"] for row in rows)))


if __name__ == "__main__":
    main()
