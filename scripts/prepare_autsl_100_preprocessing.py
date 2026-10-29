"""Freeze a reproducible 100-class AUTSL subset for preprocessing QA only.

The existing 10 pilot classes are retained. The other 90 are sampled only
from official train labels, before examining new validation quality results.
"""

from __future__ import annotations

import csv
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data/autsl"
CONFIGS = ROOT / "configs/experiment"
PILOT_10 = (8, 14, 15, 20, 64, 106, 114, 143, 144, 196)
SEED = 20261021


def labels(split: str) -> list[tuple[str, int]]:
    with (DATA / f"{split}_labels.csv").open(encoding="utf-8-sig", newline="") as stream:
        rows = [(video, int(label)) for video, label in csv.reader(stream)]
    if len({video for video, _ in rows}) != len(rows):
        raise ValueError(f"Duplicate {split} video IDs")
    return rows


def main() -> None:
    train, val = labels("train"), labels("val")
    source_ids = sorted({label for _, label in train})
    if len(source_ids) != 226 or not set(PILOT_10).issubset(source_ids):
        raise ValueError("Train labels do not contain the expected 226 classes/pilot 10")
    selected = sorted(set(PILOT_10) | set(random.Random(SEED).sample(
        sorted(set(source_ids) - set(PILOT_10)), 90)))
    if len(selected) != 100 or not set(selected).issubset({label for _, label in val}):
        raise ValueError("Selected classes must occur in both train and val")

    with (DATA / "SignList_ClassId_TR_EN.csv").open(encoding="utf-8-sig", newline="") as stream:
        names = {int(row["ClassId"]): {"tr": row["TR"], "en": row["EN"]}
                 for row in csv.DictReader(stream)}
    train_counts, val_counts = Counter(label for _, label in train), Counter(label for _, label in val)
    selection = {
        "purpose": "preprocessing quality audit; no model training in this phase",
        "selection": "existing 10 pilot IDs plus 90 uniformly sampled from remaining train class IDs",
        "random_seed": SEED,
        "pilot_10_source_class_ids": list(PILOT_10),
        "source_train_labels_sha256": hashlib.sha256((DATA / "train_labels.csv").read_bytes()).hexdigest(),
        "source_class_list_sha256": hashlib.sha256((DATA / "SignList_ClassId_TR_EN.csv").read_bytes()).hexdigest(),
        "source_class_ids": selected,
        "expected_train_videos": sum(train_counts[i] for i in selected),
        "expected_val_videos": sum(val_counts[i] for i in selected),
        "classes": [{"model_index": index, "source_class_id": source_id, **names[source_id],
                     "train_videos": train_counts[source_id], "val_videos": val_counts[source_id]}
                    for index, source_id in enumerate(selected)],
    }
    (DATA / "autsl_100_preprocessing_selection.json").write_text(
        json.dumps(selection, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    pilot: dict[str, list[str]] = {}
    for split, rows in (("train", train), ("val", val)):
        by_class: dict[int, list[str]] = defaultdict(list)
        for video_id, label in rows:
            if label in selected:
                by_class[label].append(f"{video_id}_color")
        rng = random.Random(SEED + (0 if split == "train" else 1))
        pilot[split] = [rng.choice(sorted(by_class[class_id])) for class_id in selected]
    (DATA / "autsl_100_preprocessing_pilot_ids.json").write_text(
        json.dumps(pilot, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (DATA / "autsl_class_list_100.txt").write_text(
        "".join(f"{item['model_index']} {item['tr']}\n" for item in selection["classes"]),
        encoding="utf-8")

    original = yaml.safe_load((CONFIGS / "autsl_10_mobilenet.yaml").read_text(encoding="utf-8"))
    original["name"] = "autsl_100_preprocess_base16"
    original["output_dir"] = "outputs/autsl_100_preprocess_base16/"
    original["dataset"].update(
        num_classes=100, class_ids=selected, num_frames=16,
        cache_dir="cache/autsl_100_features_base16",
        class_list_file="data/autsl/autsl_class_list_100.txt")
    original["model"]["num_classes"] = 100
    original["training"]["checkpoint_dir"] = "outputs/autsl_100_preprocess_base16/checkpoints/"
    base = CONFIGS / "autsl_100_preprocess_base16.yaml"
    base.write_text("# Preprocessing input only; no 100-class training authorized yet.\n"
                    + yaml.safe_dump(original, allow_unicode=True, sort_keys=False), encoding="utf-8")

    strict_config = json.loads(json.dumps(original))
    strict_config["name"] = "autsl_100_preprocess_centerrepair_strict"
    strict_config["output_dir"] = "outputs/autsl_100_preprocess_centerrepair_strict/"
    strict_config["dataset"].update(
        cache_dir="cache/autsl_100_features_centerrepair_strict",
        cache_derivation="pose_wrist_fallback_crop", cache_derivation_version=3,
        fallback_box_scale=1.5,
        fallback_center_repair=json.loads(
            (CONFIGS / "center_repair_strict.json").read_text(encoding="utf-8")))
    strict_config["training"]["checkpoint_dir"] = (
        "outputs/autsl_100_preprocess_centerrepair_strict/checkpoints/")
    strict = CONFIGS / "autsl_100_preprocess_centerrepair_strict.yaml"
    strict.write_text("# Derived-cache inspection config; build using repair_hand_crops.py.\n"
                      + yaml.safe_dump(strict_config, allow_unicode=True, sort_keys=False), encoding="utf-8")
    print(f"Frozen 100 classes: {len(selected)}; train={selection['expected_train_videos']}, "
          f"val={selection['expected_val_videos']}")
    print(f"Selection: {DATA / 'autsl_100_preprocessing_selection.json'}")
    print(f"Configs: {base}, {strict}")


if __name__ == "__main__":
    main()
