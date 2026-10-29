"""Copy a complete AUTSL cache to a new destination and verify SHA-256.

The manifest is published last so an incomplete upload cannot look ready.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def sha(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def copy_one(pair: tuple[Path, Path]) -> tuple[str, bool]:
    source, target = pair
    target.parent.mkdir(parents=True, exist_ok=True)
    digest = sha(source)
    if target.is_file() and target.stat().st_size == source.stat().st_size and sha(target) == digest:
        return str(target), False
    temporary = target.with_suffix(target.suffix + ".part")
    shutil.copyfile(source, temporary)
    # Drive Desktop can expose a just-copied virtual file before readback is ready.
    verified = False
    for delay in (0, .25, .5, 1, 2, 4):
        if delay:
            time.sleep(delay)
        try:
            verified = temporary.stat().st_size == source.stat().st_size and sha(temporary) == digest
        except OSError:
            verified = False
        if verified:
            break
    if not verified:
        raise IOError(f"Copy verification failed: {source} -> {target}")
    temporary.replace(target)
    return str(target), True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source_cache", type=Path, required=True)
    parser.add_argument("--dest_cache", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source, dest = args.source_cache.resolve(), args.dest_cache.resolve()
    if source == dest or source in dest.parents or dest in source.parents:
        parser.error("Source and destination must be separate caches")
    if not 1 <= args.workers <= 8:
        parser.error("Workers must be 1..8")
    files = sorted(path for split in ("train", "val")
                   for path in (source / split).glob("*.npz"))
    if not files or not (source / "manifest.json").is_file():
        parser.error("Source cache is incomplete")
    pairs = [(path, dest / path.relative_to(source)) for path in files]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(copy_one, pairs))
    counts = {split: len(list((dest / split).glob("*.npz"))) for split in ("train", "val")}
    expected = {split: len(list((source / split).glob("*.npz"))) for split in ("train", "val")}
    if counts != expected:
        raise IOError(f"Destination file counts differ: {counts} != {expected}")
    for name in ("motion2_summary.json", "manifest.json"):
        path = source / name
        if path.is_file():
            copy_one((path, dest / name))
    report = {"source": str(source), "destination": str(dest), "counts": counts,
              "bytes": sum(path.stat().st_size for path in files),
              "new_or_replaced": sum(changed for _, changed in results),
              "verified_sha256_files": len(results),
              "manifest_sha256": sha(dest / "manifest.json")}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
