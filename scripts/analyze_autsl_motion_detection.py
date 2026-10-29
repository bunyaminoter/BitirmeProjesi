"""Relate sampled pose-wrist travel to hand detection on an AUTSL base cache.

Descriptive proxy only: pose motion is not optical blur, and an unmoving wrist
can still perform an important finger-shape sign. No model is trained.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.core.config import load_config  # noqa: E402
from src.data.datasets.tid_dataset import TIDDataset  # noqa: E402

BINS = (0, 0.05, 0.1, 0.2, 0.4, 0.8, float("inf"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--cache_dir", type=Path, required=True)
    parser.add_argument("--data_dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--queue_csv", type=Path,
                        help="Optional fixed video queue; avoids streaming the full Drive cache")
    args = parser.parse_args()
    selected = None
    if args.queue_csv:
        with args.queue_csv.open(encoding="utf-8-sig", newline="") as handle:
            selected = {(r["split"], r["video_id"]) for r in csv.DictReader(handle)}
    if args.data_dir:
        os.environ["AUTSL_DATA_DIR"] = str(args.data_dir.resolve())
    config = load_config(str(args.config))
    if config.dataset.name != "autsl" or config.dataset.class_ids is None:
        parser.error("Explicit AUTSL subset config required")
    summaries = {}
    for split in ("train", "val"):
        dataset = TIDDataset(config.dataset, split)
        by_bin = [[0, 0] for _ in range(len(BINS) - 1)]
        by_signer: dict[str, list[list[int]]] = defaultdict(
            lambda: [[0, 0] for _ in range(len(BINS) - 1)])
        missing = []
        dominant_clips = 0
        dominant_observed = [0, 0]
        dominant_total = [0, 0]
        for sample in dataset.samples:
            if selected is not None and (split, sample.video_id) not in selected:
                continue
            path = args.cache_dir / split / f"{sample.video_id}.npz"
            if not path.is_file():
                missing.append(sample.video_id)
                continue
            with np.load(path, allow_pickle=False) as data:
                pose = data["pose_landmarks"][:, :, :2]
                pose_ok = data["pose_detected"].astype(bool)
                observed = {side: data[f"{side}_hand_detected"].astype(bool)
                            for side in ("left", "right")}
            shoulder = np.linalg.norm(pose[:, 11] - pose[:, 12], axis=1)
            valid = pose_ok[1:] & pose_ok[:-1] & (shoulder[1:] > 0.01)
            travel = {}
            for side, index in (("left", 15), ("right", 16)):
                speed = np.linalg.norm(np.diff(pose[:, index], axis=0), axis=1)
                speed = speed / np.maximum(shoulder[1:], 0.01)
                travel[side] = float(speed[valid].sum())
                for value, detected in zip(speed[valid], observed[side][1:][valid]):
                    bin_index = min(int(np.searchsorted(BINS, value, side="right") - 1), len(by_bin)-1)
                    by_bin[bin_index][0] += 1
                    by_bin[bin_index][1] += int(detected)
                    by_signer[sample.signer_id][bin_index][0] += 1
                    by_signer[sample.signer_id][bin_index][1] += int(detected)
            dominant = max(travel, key=travel.get)
            passive = "right" if dominant == "left" else "left"
            if travel[dominant] >= 1.75 * max(travel[passive], 0.05) and travel[dominant] >= 0.8:
                dominant_clips += 1
                for slot, side in enumerate((dominant, passive)):
                    dominant_observed[slot] += int(observed[side].sum())
                    dominant_total[slot] += len(observed[side])
        summaries[split] = {
            "selected_videos": sum(1 for sample in dataset.samples
                                   if selected is None or (split, sample.video_id) in selected),
            "missing_cache": len(missing),
            "missing_examples": missing[:10],
            "motion_bins_in_shoulder_widths_per_sampled_step": [
                {"min": low, "max": high if np.isfinite(high) else None,
                 "hand_frames": total, "detected": detected,
                 "detected_rate": round(detected/total, 4) if total else None}
                for (low, high), (total, detected) in zip(zip(BINS[:-1], BINS[1:]), by_bin)],
            "clear_dominant_wrist_motion_clips": dominant_clips,
            "dominant_hand_detected_rate": round(dominant_observed[0]/dominant_total[0], 4)
            if dominant_total[0] else None,
            "other_hand_detected_rate": round(dominant_observed[1]/dominant_total[1], 4)
            if dominant_total[1] else None,
            "selected_signers": {signer: [
                {"hand_frames": total, "detected_rate": round(detected/total, 4) if total else None}
                for total, detected in by_signer[signer]]
                for signer in ("signer16", "signer28", "signer19") if signer in by_signer},
        }
        print(split, "videos", summaries[split]["selected_videos"], "missing", len(missing),
              "bins", [(x[0], x[1]) for x in by_bin], flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summaries, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if any(summaries[s]["missing_cache"] for s in summaries):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
