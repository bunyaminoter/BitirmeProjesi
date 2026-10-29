"""Independently check matched AUTSL train/val caches, including crop pixels.

Never opens test. Rates are per hand observation; paired bootstrap resamples
videos, not correlated frames. Missing-run fractions are descriptive sampling
proxies, not measured occlusion durations in seconds.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.core.config import load_config
from src.data.datasets.tid_dataset import TIDDataset


def longest_gap(mask):
    return max(map(len, "".join("0" if x else "1" for x in mask).split("0")))


def review(root: Path) -> dict:
    names = ["autsl_10_features", "autsl_10_features_36",
             "autsl_10_features_wristcrop", "autsl_10_features_gap2"]
    config = load_config(str(ROOT / "configs/experiment/autsl_10_mobilenet.yaml"))
    result = {"scope": "official 10-class train/val only", "splits": {}}
    signers = {}
    for split in ("train", "val"):
        dataset = TIDDataset(config.dataset, split)
        expected = {s.video_id: s.label for s in dataset.samples}
        signers[split] = {s.signer_id for s in dataset.samples}
        stats = {name: Counter() for name in names}
        rows, residual, outliers = [], [], []
        for name in names:
            actual = {p.stem for p in (root / name / split).glob("*.npz")}
            if actual != set(expected):
                raise ValueError(f"{name}/{split}: missing={set(expected)-actual}, extra={actual-set(expected)}")
            stats[name]["bytes"] = sum(p.stat().st_size for p in (root / name / split).glob("*.npz"))
        for video_id, label in sorted(expected.items()):
            paths = [root / name / split / f"{video_id}.npz" for name in names]
            with np.load(paths[0]) as base, np.load(paths[1]) as dense, np.load(paths[2]) as wrist, np.load(paths[3]) as gap:
                rates = []
                for name, data in zip(names, (base, dense, wrist, gap)):
                    assert int(data["label"]) == label, (name, video_id, "label")
                    count = int(data["num_frames"])
                    assert count == (36 if name == names[1] else 16), (name, video_id, "frames")
                    stats[name]["videos"] += 1
                    stats[name]["frames"] += count
                    gaps, hits = [], 0
                    for side in ("left", "right"):
                        mask = data[f"{side}_hand_detected"].astype(bool)
                        assert mask.shape == (count,)
                        hits += int(mask.sum())
                        run = longest_gap(mask)
                        gaps.append(run / count)
                        stats[name][f"{side}_9plus_videos"] += int(run >= 9)
                        if name == names[3]:
                            imputed = data[f"{side}_hand_imputed"].astype(bool)
                            coords = data[f"{side}_hand_landmarks"][imputed, :, :2]
                            bad = ((coords < 0) | (coords > 1)).any(axis=-1)
                            stats[name][f"{side}_imputed_frames"] += int(imputed.sum())
                            stats[name][f"{side}_out_of_frame_imputed_points"] += int(bad.sum())
                            if bad.any():
                                outliers.append({"video_id": video_id, "side": side,
                                                 "points": int(bad.sum())})
                    stats[name]["hand_hits"] += hits
                    stats[name]["gap_ge_quarter_clip_videos"] += int(max(gaps) >= .25)
                    rates.append(hits / (2 * count))
                for key in ("label", "num_frames", "pose_landmarks", "pose_detected",
                            "left_hand_landmarks", "right_hand_landmarks",
                            "left_hand_detected", "right_hand_detected"):
                    assert np.array_equal(base[key], wrist[key]), (video_id, key)
                for side in ("left", "right"):
                    observed = base[f"{side}_hand_detected"].astype(bool)
                    old, new = base[f"{side}_hand_crops"], wrist[f"{side}_hand_crops"]
                    src = wrist[f"{side}_crop_source"]
                    assert np.isin(src, [0, 1, 2]).all()
                    assert np.array_equal(src == 0, observed)
                    assert np.array_equal(old[observed], new[observed]), (video_id, side, "detected crops changed")
                    seen = np.cumsum(observed) > 0
                    carried = (src == 2) & seen & ~observed
                    stats[names[2]][f"{side}_carried_frames"] += int(carried.sum())
                    stats[names[2]][f"{side}_unchanged_before_first_frames"] += int(((src == 2) & ~seen).sum())
                    stats[names[2]][f"{side}_fallback_frames"] += int((src == 1).sum())
                    duplicates = np.all(new[1:] == new[:-1], axis=(1, 2, 3)) & (src[1:] == 1)
                    stats[names[2]][f"{side}_fallback_identical_to_previous"] += int(duplicates.sum())
                    stats[names[2]][f"{side}_fallback_all_black"] += int((np.all(new == 0, axis=(1, 2, 3)) & (src == 1)).sum())
                    if carried.any():
                        residual.append({"video_id": video_id, "side": side,
                                         "indices": np.flatnonzero(carried).tolist()})
                rows.append({"video_id": video_id, "dense_minus_base": rates[1]-rates[0]})
        delta = np.array([r["dense_minus_base"] for r in rows])
        rng = np.random.default_rng(42)
        bootstrap = np.array([rng.choice(delta, len(delta)).mean() for _ in range(5000)])
        for s in stats.values():
            s["hand_observed_rate"] = s["hand_hits"] / (2*s["frames"])
        result["splits"][split] = {
            "caches": {k: dict(v) for k, v in stats.items()},
            "dense_minus_base_percentage_points": float(delta.mean()*100),
            "paired_video_bootstrap_95pct_interval_pp": (np.quantile(bootstrap, [.025, .975])*100).tolist(),
            "wrist_remaining_carried": residual,
            "gap_out_of_frame": outliers,
        }
        print(split, json.dumps(result["splits"][split], ensure_ascii=False), flush=True)
    assert not signers["train"] & signers["val"]
    result["train_val_signer_overlap"] = []
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache_root", type=Path, default=ROOT / "cache")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/preprocessing_independent_review.json")
    args = parser.parse_args()
    report = review(args.cache_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
