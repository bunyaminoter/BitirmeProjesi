"""Read-only anatomical-side audit from pose forearm trajectories and ROI wrists.

The proxy comparison uses baseline side labels on nearby wrists. It is not
human side ground truth, especially on crossed/occluded hands.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from itertools import permutations
from pathlib import Path

import numpy as np

SIDES = ("left", "right")
ELBOW = {"left": 13, "right": 14}
WRIST = {"left": 15, "right": 16}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--roi_results", type=Path, required=True)
    parser.add_argument("--base_cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    records = json.loads(args.roi_results.read_text(encoding="utf-8"))["records"]
    rows = []
    counts = Counter()
    by_video = {}
    for record in records:
        video_id = record["video_id"]
        video_counts = Counter()
        with np.load(args.base_cache / "train" / f"{video_id}.npz", allow_pickle=False) as cache:
            pose = cache["pose_landmarks"]
            base_seen = {s: cache[f"{s}_hand_detected"].astype(bool) for s in SIDES}
            base_wrist = {s: cache[f"{s}_hand_landmarks"][:, 0, :2] for s in SIDES}
        for frame in record["frames"]:
            t = frame["frame"]
            hands = np.asarray(frame["roi_wrists"], dtype=float).reshape(-1, 2)
            if len(hands) == 0:
                continue
            forearm = {s: pose[t, WRIST[s], :2] + .25 * (
                pose[t, WRIST[s], :2] - pose[t, ELBOW[s], :2]) for s in SIDES}
            if not all(np.isfinite(forearm[s]).all() for s in SIDES):
                continue
            if len(hands) == 1:
                costs = {s: float(np.linalg.norm(hands[0] - forearm[s])) for s in SIDES}
                assignment = min(costs, key=costs.get)
                gap = max(costs.values()) - min(costs.values())
                assigned = [assignment]
            else:
                costs = [(sum(np.linalg.norm(hands[i] - forearm[side]) for i, side in enumerate(order)), order)
                         for order in permutations(SIDES)]
                costs.sort(key=lambda pair: pair[0])
                gap = float(costs[1][0] - costs[0][0])
                assigned = list(costs[0][1])
            for i, side in enumerate(assigned):
                distances = {s: float(np.linalg.norm(hands[i] - base_wrist[s][t]))
                             for s in SIDES if base_seen[s][t]}
                matched = min(distances, key=distances.get) if distances and min(distances.values()) < .06 else ""
                row = {"video_id": video_id, "frame_1based": t + 1, "roi_index": i,
                       "roi_wrist": hands[i].round(4).tolist(), "assigned_side": side,
                       "arm_gap": round(gap, 4), "base_hand_count": frame["base_count"],
                       "roi_hand_count": len(hands), "matched_base_side": matched,
                       "proxy_agrees": side == matched if matched else None}
                rows.append(row)
                if matched:
                    counts["proxy_matched"] += 1
                    counts["proxy_agree" if side == matched else "proxy_disagree"] += 1
                    video_counts["proxy_matched"] += 1
                    video_counts["proxy_agree" if side == matched else "proxy_disagree"] += 1
                    if gap >= .03:
                        counts["proxy_high_gap"] += 1
                        counts["proxy_high_gap_agree" if side == matched else "proxy_high_gap_disagree"] += 1
                if frame["base_count"] == 0 and len(hands) == 2:
                    counts["base_empty_roi_two_hands"] += 1
                    if gap >= .03:
                        counts["base_empty_roi_two_hands_high_gap"] += 1
        by_video[video_id] = dict(video_counts)
    special = [row for row in rows if row["video_id"] == "signer17_sample52_color"
               and 5 <= row["frame_1based"] <= 13]
    result = {"videos": len(records), "counts": dict(counts), "by_video": by_video,
              "signer17_sample52_active_window": special,
              "note": "Forearm extrapolation is a side hypothesis. Baseline matches are a proxy, not human anatomical labels."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"videos": len(records), "counts": dict(counts),
                      "signer17_rows": len(special)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
