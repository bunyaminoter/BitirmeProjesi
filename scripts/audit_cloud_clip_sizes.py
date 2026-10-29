"""Summarize AUTSL clip file sizes and ideal upload-time lower bounds.

Reads file metadata only; no video decoding or network measurement. The values
must not be presented as real phone latency or a mobile codec benchmark.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video_dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sizes = sorted(p.stat().st_size for p in args.video_dir.glob("*_color.mp4"))
    if not sizes:
        raise ValueError(f"No AUTSL RGB MP4 files found: {args.video_dir}")
    count = len(sizes)
    percentiles = {f"p{p}": sizes[math.floor(p / 100 * (count - 1))]
                   for p in (50, 90, 99)}
    result = {
        "source": "AUTSL_Dataset/val/*_color.mp4",
        "method": "file metadata; actual phone encoding, network RTT, server inference excluded",
        "count": count,
        "min_bytes": sizes[0], "mean_bytes": round(sum(sizes) / count),
        **{f"{key}_bytes": value for key, value in percentiles.items()},
        "max_bytes": sizes[-1],
        "ideal_upload_ms_by_effective_uplink_mbps": {
            str(rate): {key: round(value * 8 / (rate * 1000))
                        for key, value in percentiles.items()}
            for rate in (5, 10, 20, 50)
        },
        "raw_16x512x512_rgb_bytes": 16 * 512 * 512 * 3,
        "raw_32x160x160_rgb_bytes": 32 * 160 * 160 * 3,
        "raw_16x75x3_float32_landmarks_bytes": 16 * 75 * 3 * 4,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                           encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
