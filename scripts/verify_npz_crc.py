"""Stream and CRC-check every member of each NPZ in a selected cache folder."""

from __future__ import annotations

import argparse
import json
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def check(path: Path, headers_only: bool) -> tuple[str, str | None]:
    try:
        with zipfile.ZipFile(path) as archive:
            members = archive.namelist()
            if not members or any(not name.endswith(".npy") for name in members):
                return str(path), "unexpected/empty NPZ members"
            bad = None if headers_only else archive.testzip()
            return str(path), f"CRC failure: {bad}" if bad else None
    except (OSError, EOFError, zipfile.BadZipFile, RuntimeError) as exc:
        return str(path), f"{type(exc).__name__}: {exc}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--headers_only", action="store_true",
                        help="Check ZIP directory/member names without decompressing or CRC")
    args = parser.parse_args()
    files = sorted(args.root.glob("train/*.npz")) + sorted(args.root.glob("val/*.npz"))
    if not files:
        parser.error("No NPZ files found")
    errors = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for i, (name, error) in enumerate(pool.map(lambda p: check(p, args.headers_only), files), 1):
            if error:
                errors.append({"path": name, "error": error})
            if i % 500 == 0:
                print(f"{'headers' if args.headers_only else 'CRC'} checked {i}/{len(files)}, errors={len(errors)}", flush=True)
    report = {"root": str(args.root), "checked": len(files),
              "mode": "headers_only" if args.headers_only else "full_crc",
              "errors": errors, "invalid_count": len(errors)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"{'headers' if args.headers_only else 'CRC'} checked {len(files)}, errors={len(errors)}")
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
