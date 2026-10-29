"""Re-read a reproducible class-balanced raw-video sample and check crop pixels.

All detected frames of the selected videos are checked, plus all fallback
frames. This checks provenance/alignment, not whether an annotated hand is
inside the crop; that requires visual review or ground-truth boxes.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from preprocess_tid import sample_frames
from src.core.config import load_config
from src.core.types import LandmarkResult
from src.data.datasets.tid_dataset import TIDDataset
from src.preprocessing.hand_cropper import HandCropper
from src.preprocessing.wrist_fallback_crop import fallback_crops, video_box_side, pose_hand_center


def main():
    config = load_config(ROOT / "configs/experiment/autsl_10_wristcrop.yaml")
    rng = np.random.default_rng(42)
    report = {"seed": 42, "selection": "one video per class per split plus known hard/residual cases", "videos": []}
    extra = {"train": {"signer17_sample360_color", "signer8_sample1539_color", "signer33_sample505_color"},
             "val": {"signer16_sample184_color", "signer35_sample539_color"}}
    cropper = HandCropper(tuple(config.model.hand_encoder.input_size), padding_ratio=.2)
    for split in ("train", "val"):
        dataset = TIDDataset(config.dataset, split)
        by_label = {}
        for index, sample in enumerate(dataset.samples):
            by_label.setdefault(sample.label, []).append(index)
        chosen = {int(rng.choice(indices)) for _, indices in sorted(by_label.items())}
        chosen.update(i for i, s in enumerate(dataset.samples) if s.video_id in extra[split])
        for index in sorted(chosen):
            video_id = dataset.samples[index].video_id
            frames = sample_frames(str(dataset.get_video_path(index)), config.dataset.num_frames)
            with np.load(ROOT / "cache/autsl_10_features" / split / f"{video_id}.npz") as base, np.load(
                ROOT / config.dataset.cache_dir / split / f"{video_id}.npz") as wrist:
                record = {"split": split, "video_id": video_id, "detected_crops_checked": 0,
                          "fallback_crops_checked": 0, "max_absolute_pixel_error": 0,
                          "label": dataset.samples[index].label,
                          "pose_proxy_frames": 0, "pose_proxy_all_21_points_inside": 0,
                          "pose_proxy_full_hand_by_scale": {str(s): 0 for s in (1., 1.25, 1.5, 1.75, 2.)}}
                for side in ("left", "right"):
                    observed = base[f"{side}_hand_detected"].astype(bool)
                    height, width = frames[0].shape[:2]
                    box = video_box_side(base[f"{side}_hand_landmarks"], observed,
                        base["pose_landmarks"], base["pose_detected"].astype(bool), side, width, height)
                    for t in np.flatnonzero(observed):
                        detected = LandmarkResult(**{f"{side}_hand": base[f"{side}_hand_landmarks"][t]})
                        crops = cropper.crop_hands(frames[t], detected)
                        crop = crops[0 if side == "left" else 1].image
                        error = int(np.abs(crop.astype(np.int16)-wrist[f"{side}_hand_crops"][t].astype(np.int16)).max())
                        record["max_absolute_pixel_error"] = max(record["max_absolute_pixel_error"], error)
                        record["detected_crops_checked"] += 1
                        if box is not None and base["pose_detected"][t]:
                            center = pose_hand_center(base["pose_landmarks"][t], side, width, height)
                            points = base[f"{side}_hand_landmarks"][t, :, :2] * [width, height]
                            inside = (np.abs(points-center) <= max(box, 8.)/2).all(axis=1)
                            record["pose_proxy_frames"] += 1
                            record["pose_proxy_all_21_points_inside"] += int(inside.all())
                            for scale in record["pose_proxy_full_hand_by_scale"]:
                                covered = (np.abs(points-center) <= max(box, 8.)*float(scale)/2).all()
                                record["pose_proxy_full_hand_by_scale"][scale] += int(covered)
                    expected, source = fallback_crops(frames, base[f"{side}_hand_crops"],
                        base[f"{side}_hand_landmarks"], observed, base["pose_landmarks"],
                        base["pose_detected"].astype(bool), side)
                    assert np.array_equal(source, wrist[f"{side}_crop_source"]), video_id
                    assert np.array_equal(expected, wrist[f"{side}_hand_crops"]), video_id
                    record["fallback_crops_checked"] += int((source == 1).sum())
                assert record["max_absolute_pixel_error"] == 0, record
                report["videos"].append(record)
    report["summary"] = {"videos_checked": len(report["videos"]),
        "detected_crops_checked": sum(r["detected_crops_checked"] for r in report["videos"]),
        "fallback_crops_checked": sum(r["fallback_crops_checked"] for r in report["videos"]),
        "max_absolute_pixel_error": max(r["max_absolute_pixel_error"] for r in report["videos"]),
        "pose_proxy_frames": sum(r["pose_proxy_frames"] for r in report["videos"]),
        "pose_proxy_all_21_points_inside": sum(r["pose_proxy_all_21_points_inside"] for r in report["videos"])}
    report["summary"]["pose_proxy_full_hand_by_scale"] = {
        scale: sum(r["pose_proxy_full_hand_by_scale"][scale] for r in report["videos"])
        for scale in report["videos"][0]["pose_proxy_full_hand_by_scale"]}
    output = ROOT / "outputs/wristcrop_raw_pixel_review.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report["summary"]))


if __name__ == "__main__":
    main()
