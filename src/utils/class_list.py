"""
Utilities for parsing class list files.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple


def load_class_list(class_list_file: str | Path) -> Tuple[Dict[str, int], Dict[int, str]]:
    """Load class mappings from a class list text file.

    Supports both formats:
        0 book
        0\\tbook

    and gloss-only lines (index inferred from line order):
        book

    Args:
        class_list_file: Path to the class list file.

    Returns:
        Tuple of (class_to_idx, idx_to_class).
    """
    path = Path(class_list_file)
    if not path.exists():
        return {}, {}

    class_to_idx: Dict[str, int] = {}
    idx_to_class: Dict[int, str] = {}

    with open(path, "r", encoding="utf-8") as f:
        for line_no, raw_line in enumerate(f):
            line = raw_line.strip()
            if not line:
                continue

            parts = line.replace("\t", " ").split(maxsplit=1)
            if len(parts) == 2 and parts[0].isdigit():
                idx = int(parts[0])
                gloss = parts[1].strip()
            else:
                idx = line_no
                gloss = line

            class_to_idx[gloss] = idx
            idx_to_class[idx] = gloss

    return class_to_idx, idx_to_class
