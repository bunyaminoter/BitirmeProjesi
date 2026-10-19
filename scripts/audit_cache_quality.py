"""Full-cache quality distribution for AUTSL caches (train/val only, no model).

Summarises per cache and split: hand detection, missing-run length buckets,
how many hand-crop frames are stale carry-forward copies, wrist agreement,
per-signer rates and the worst videos for visual inspection.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

from compare_preprocessing_pilots import metrics  # noqa: E402

RUN_BUCKETS = ((0, 0), (1, 2), (3, 4), (5, 8), (9, 10**6))


def bucket_name(low: int, high: int) -> str:
    return str(low) if low == high else (f"{low}+" if high >= 10**6 else f"{low}-{high}")


def crop_status(path: Path) -> dict:
    """Count crop frames that are fresh, stale (carried forward) or before first detection."""
    out = {}
    with np.load(path, allow_pickle=False) as data:
        count = int(data["num_frames"])
        for side in ("left", "right"):
            observed = data[f"{side}_hand_detected"][:count].astype(bool)
            seen = np.cumsum(observed) > 0
            if f"{side}_crop_source" in data:
                src = data[f"{side}_crop_source"][:count]
                out[f"{side}_fallback_crop"] = int((src == 1).sum())
                out[f"{side}_stale"] = int(((src == 2) & seen).sum())
                out[f"{side}_before_first"] = int(((src == 2) & ~seen).sum())
            else:
                out[f"{side}_stale"] = int((~observed & seen).sum())
                out[f"{side}_before_first"] = int((~observed & ~seen).sum())
            if f"{side}_hand_imputed" in data:
                imputed = data[f"{side}_hand_imputed"][:count].astype(bool)
                out[f"{side}_unrepaired_missing"] = int((~observed & ~imputed).sum())
    return out


def summarise(records: list[dict]) -> dict:
    frames = sum(r["frames"] for r in records)
    result = {"videos": len(records), "frames": frames,
              "pose_detected_rate": round(sum(r["pose_detected"] for r in records) / frames, 4)}
    for side in ("left", "right"):
        observed = sum(r[f"{side}_observed"] for r in records)
        result[f"{side}_observed_rate"] = round(observed / frames, 4)
        result[f"{side}_stale_crop_rate"] = round(sum(r[f"{side}_stale"] for r in records) / frames, 4)
        result[f"{side}_before_first_rate"] = round(
            sum(r[f"{side}_before_first"] for r in records) / frames, 4)
        if any(f"{side}_fallback_crop" in r for r in records):
            result[f"{side}_fallback_crop_rate"] = round(
                sum(r.get(f"{side}_fallback_crop", 0) for r in records) / frames, 4)
        runs = [r[f"{side}_max_missing_run"] for r in records]
        result[f"{side}_max_missing_run_videos"] = {
            bucket_name(lo, hi): sum(lo <= v <= hi for v in runs) for lo, hi in RUN_BUCKETS}
        agreement = [r[f"{side}_wrist_agreement"] for r in records if r[f"{side}_wrist_agreement"] is not None]
        result[f"{side}_wrist_agreement_mean"] = round(float(np.mean(agreement)), 4) if agreement else None
        result[f"{side}_wrist_agreement_below_0.8_videos"] = sum(a < 0.8 for a in agreement)
        if any(f"{side}_imputed" in r for r in records):
            result[f"{side}_imputed_frames"] = sum(r.get(f"{side}_imputed", 0) for r in records)
            result[f"{side}_unrepaired_missing_rate"] = round(
                sum(r.get(f"{side}_unrepaired_missing", 0) for r in records) / frames, 4)
            result[f"{side}_imputed_out_of_frame_points"] = sum(
                r.get(f"{side}_imputed_out_of_frame_points", 0) for r in records)
    result["both_hands_observed_rate_mean"] = round(
        (result["left_observed_rate"] + result["right_observed_rate"]) / 2, 4)
    long_gap = sum(max(r["left_max_missing_run_frac"], r["right_max_missing_run_frac"]) >= 0.25 for r in records)
    result["videos_with_gap_ge_25pct_of_clip"] = long_gap
    return result


def audit(cache: Path, splits: list[str], worst: int) -> dict:
    manifest = json.loads((cache / "manifest.json").read_text(encoding="utf-8"))
    report = {"cache": str(cache), "manifest": manifest, "splits": {}}
    for split in splits:
        files = sorted((cache / split).glob("*.npz"))
        if not files:
            raise FileNotFoundError(f"No cache files in {cache / split}")
        records = []
        for path in files:
            record = {"video_id": path.stem, "signer": path.stem.split("_")[0]}
            record.update(metrics(path))
            record.update(crop_status(path))
            for side in ("left", "right"):
                record[f"{side}_max_missing_run_frac"] = record[f"{side}_max_missing_run"] / record["frames"]
            records.append(record)
        by_signer: dict[str, list[dict]] = defaultdict(list)
        by_label: dict[int, list[dict]] = defaultdict(list)
        for record in records:
            by_signer[record["signer"]].append(record)
            by_label[record["label"]].append(record)

        def compact(group: list[dict]) -> dict:
            frames = sum(r["frames"] for r in group)
            return {"videos": len(group),
                    "left_observed_rate": round(sum(r["left_observed"] for r in group) / frames, 4),
                    "right_observed_rate": round(sum(r["right_observed"] for r in group) / frames, 4)}

        ranked = sorted(records, key=lambda r: -(r["left_max_missing_run_frac"] + r["right_max_missing_run_frac"]))
        report["splits"][split] = {
            "summary": summarise(records),
            "by_signer": {k: compact(v) for k, v in sorted(by_signer.items())},
            "by_label": {str(k): compact(v) for k, v in sorted(by_label.items())},
            "worst_videos": [
                {key: r[key] for key in ("video_id", "label", "frames", "left_observed", "right_observed",
                                         "left_max_missing_run", "right_max_missing_run")}
                for r in ranked[:worst]
            ],
        }
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--caches", nargs="+", type=Path, default=[
        ROOT / "cache/autsl_10_features", ROOT / "cache/autsl_10_features_gap2",
        ROOT / "cache/autsl_10_features_36"])
    parser.add_argument("--splits", nargs="+", choices=["train", "val"], default=["train", "val"])
    parser.add_argument("--worst", type=int, default=15)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/full_cache_quality.json")
    args = parser.parse_args()
    reports = {cache.name: audit(cache, args.splits, args.worst) for cache in args.caches}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(reports, ensure_ascii=False, indent=2), encoding="utf-8")
    for name, report in reports.items():
        for split, block in report["splits"].items():
            print(f"== {name} / {split}")
            print(json.dumps(block["summary"], ensure_ascii=False, indent=2))
    print(f"Report: {args.output}")


if __name__ == "__main__":
    main()
