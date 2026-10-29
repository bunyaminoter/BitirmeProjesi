"""Read-only AUTSL 100 Drive cache coverage and paired pixel audit.

Reports what was independently checked. File-name and size coverage is full;
NPZ members and crop pixels are decoded only for the deterministic sample.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from scripts.preprocess_tid import sample_frames  # noqa: E402
from src.core.config import load_config  # noqa: E402
from src.data.datasets.tid_dataset import TIDDataset  # noqa: E402
from src.preprocessing.wrist_fallback_crop import fallback_crops, fallback_geometry  # noqa: E402

RISK_IDS = {
    "train": [
        "signer28_sample472_color", "signer19_sample179_color",
        "signer13_sample1751_color", "signer13_sample1216_color",
        "signer22_sample484_color", "signer37_sample444_color",
        "signer38_sample21_color", "signer8_sample1058_color",
        "signer33_sample186_color", "signer33_sample505_color",
        "signer41_sample269_color",
    ],
    "val": [
        "signer16_sample70_color", "signer16_sample307_color",
        "signer35_sample62_color", "signer11_sample164_color",
        "signer25_sample65_color", "signer1_sample1012_color",
        "signer16_sample369_color",
    ],
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache_root", type=Path, required=True)
    parser.add_argument("--data_dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--extra_per_split", type=int, default=5)
    args = parser.parse_args()
    os.environ["AUTSL_DATA_DIR"] = str(args.data_dir)
    config = load_config(str(ROOT / "configs/experiment/autsl_100_preprocess_base16.yaml"))
    base = args.cache_root / "autsl_100_features_base16"
    strict = args.cache_root / "autsl_100_features_centerrepair_strict"
    base_bytes = (base / "manifest.json").read_bytes()
    base_manifest = json.loads(base_bytes)
    strict_manifest = json.loads((strict / "manifest.json").read_bytes())
    if strict_manifest.get("derivation", {}).get("source_manifest_sha256") != hashlib.sha256(base_bytes).hexdigest():
        raise ValueError("Strict source manifest hash differs from actual base manifest")
    for name in ("class_ids", "num_classes", "num_frames", "hand_image_size", "dataset"):
        if strict_manifest.get(name) != base_manifest.get(name):
            raise ValueError(f"Manifest mismatch: {name}")
    derivation = strict_manifest["derivation"]
    if derivation.get("method") != "pose_wrist_fallback_crop" or derivation.get("version") != 3:
        raise ValueError(f"Unexpected strict derivation: {derivation}")
    if derivation.get("box_scale") != 1.5 or "center_repair" not in derivation:
        raise ValueError(f"Unexpected strict policy: {derivation}")

    report = {
        "base_manifest_sha256": hashlib.sha256(base_bytes).hexdigest(),
        "strict_manifest_sha256": sha256(strict / "manifest.json"),
        "derivation": derivation,
        "splits": {}, "sample": [],
    }
    for split in ("train", "val"):
        dataset = TIDDataset(config.dataset, split)
        expected = {f"{sample.video_id}.npz" for sample in dataset.samples}
        for variant, root in (("base", base), ("strict", strict)):
            files = list((root / split).glob("*.npz"))
            found = {p.name for p in files}
            zeros = [p.name for p in files if p.stat().st_size == 0]
            report["splits"].setdefault(split, {})[variant] = {
                "expected": len(expected), "found": len(files),
                "missing": sorted(expected - found), "extra": sorted(found - expected),
                "zero_byte_files": zeros,
            }
            if len(files) != len(found) or expected != found or zeros:
                raise ValueError(f"Coverage/size failure: {split} {variant}")
        index_of = {s.video_id: i for i, s in enumerate(dataset.samples)}
        chosen = list(dict.fromkeys(RISK_IDS[split]))
        for item in sorted(dataset.samples, key=lambda s: (s.label, s.signer_id, s.video_id)):
            if len(chosen) >= len(RISK_IDS[split]) + args.extra_per_split:
                break
            if item.video_id not in chosen and item.label not in {dataset.samples[index_of[i]].label for i in chosen}:
                chosen.append(item.video_id)
        for video_id in chosen:
            with (np.load(base / split / f"{video_id}.npz", allow_pickle=False) as source,
                  np.load(strict / split / f"{video_id}.npz", allow_pickle=False) as derived):
                a = {key: source[key] for key in source.files}
                b = {key: derived[key] for key in derived.files}
            for key in ("pose_landmarks", "pose_detected", "left_hand_landmarks",
                        "right_hand_landmarks", "left_hand_detected", "right_hand_detected",
                        "label", "num_frames"):
                if not np.array_equal(a[key], b[key]):
                    raise ValueError(f"Source field changed: {split}/{video_id}/{key}")
            frames = sample_frames(str(dataset.get_video_path(index_of[video_id])), config.dataset.num_frames)
            if len(frames) != int(a["num_frames"]):
                raise ValueError(f"Raw/cache frame mismatch: {split}/{video_id}")
            record = {"split": split, "video_id": video_id, "frames": len(frames),
                      "hand_slots": 0, "fallback_slots": 0, "center_repaired_slots": 0,
                      "changed_crop_slots": 0, "max_pixel_error": 0,
                      "max_geometry_error": 0.0}
            height, width = frames[0].shape[:2]
            for side in ("left", "right"):
                observed = a[f"{side}_hand_detected"].astype(bool)
                expected_crops, expected_source = fallback_crops(
                    frames, a[f"{side}_hand_crops"], a[f"{side}_hand_landmarks"],
                    observed, a["pose_landmarks"], a["pose_detected"].astype(bool),
                    side, box_scale=1.5, center_repair=derivation["center_repair"])
                if not np.array_equal(expected_source, b[f"{side}_crop_source"]):
                    raise ValueError(f"Crop source differs: {split}/{video_id}/{side}")
                pixel_error = int(np.max(np.abs(expected_crops.astype(np.int16) -
                                                b[f"{side}_hand_crops"].astype(np.int16))))
                record["max_pixel_error"] = max(record["max_pixel_error"], pixel_error)
                if pixel_error:
                    raise ValueError(f"Crop pixel mismatch: {split}/{video_id}/{side}: {pixel_error}")
                _, centers, boxes = fallback_geometry(
                    a[f"{side}_hand_landmarks"], observed, a["pose_landmarks"],
                    a["pose_detected"], side, width, height,
                    box_scale=1.5, center_repair=derivation["center_repair"])
                for key, expected_array in (("crop_center", centers), ("crop_box", boxes),
                                            ("crop_center_imputed", expected_source == 3)):
                    actual = b[f"{side}_{key}"]
                    if actual.shape != expected_array.shape:
                        raise ValueError(f"Geometry shape differs: {split}/{video_id}/{side}/{key}")
                    difference = float(np.max(np.abs(actual.astype(float)-expected_array.astype(float))))
                    record["max_geometry_error"] = max(record["max_geometry_error"], difference)
                    if difference > 1e-5:
                        raise ValueError(f"Geometry mismatch: {split}/{video_id}/{side}/{key}: {difference}")
                record["hand_slots"] += len(frames)
                record["fallback_slots"] += int(np.isin(expected_source, (1, 3)).sum())
                record["center_repaired_slots"] += int((expected_source == 3).sum())
                record["changed_crop_slots"] += int(np.any(
                    a[f"{side}_hand_crops"] != b[f"{side}_hand_crops"], axis=(1, 2, 3)).sum())
            report["sample"].append(record)
            print(split, video_id, "pixels", record["max_pixel_error"], flush=True)
    report["summary"] = {
        "sample_videos": len(report["sample"]),
        "sample_hand_slots": sum(x["hand_slots"] for x in report["sample"]),
        "sample_fallback_slots": sum(x["fallback_slots"] for x in report["sample"]),
        "sample_center_repaired_slots": sum(x["center_repaired_slots"] for x in report["sample"]),
        "sample_changed_crop_slots": sum(x["changed_crop_slots"] for x in report["sample"]),
        "max_pixel_error": max(x["max_pixel_error"] for x in report["sample"]),
        "max_geometry_error": max(x["max_geometry_error"] for x in report["sample"]),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report["summary"], ensure_ascii=False))


if __name__ == "__main__":
    main()
