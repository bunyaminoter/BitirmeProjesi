"""Package AUTSL 10-class assets into single zip files for instant, error-free Google Drive upload.

Modes:
  --mode videos (default):
      Packages the 1,444 raw MP4 videos (train: 1,245, val: 199) and 4 official CSVs
      into outputs/autsl_10_raw_dataset.zip (~485 MB).
      Uploads in ~1 minute to Google Drive. Colab can then run preprocessing directly.

  --mode caches:
      Packages the pre-extracted cache directories (autsl_10_features and
      autsl_10_features_centerrepair_strict) into outputs/autsl_10_caches.zip (~2.8 GB).
      Uploads in ~3-4 minutes to Google Drive. Colab can unzip in 10s and skip preprocessing.

  --mode all:
      Creates both zip archives.
"""

from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src.core.config import load_config
from src.data.datasets.tid_dataset import TIDDataset


def package_videos(output_zip: Path):
    print("=" * 60)
    print("AUTSL 10 SINIF HAM VİDEO VE METADATA PAKETLEME")
    print("=" * 60)

    config = load_config(str(ROOT / "configs/experiment/autsl_10_mobilenet.yaml"))
    output_zip.parent.mkdir(parents=True, exist_ok=True)

    # 1. Collect all 1,444 video paths
    video_entries: list[tuple[Path, str]] = []  # (source_path, arcname)
    for split in ("train", "val"):
        ds = TIDDataset(config.dataset, split=split)
        for i in range(len(ds.samples)):
            path = ds.get_video_path(i)
            if not path.is_file():
                raise FileNotFoundError(f"Video not found: {path}")
            arcname = f"{split}/{path.name}"
            video_entries.append((path, arcname))

    print(f"Toplam 10 sınıf video sayısı: {len(video_entries)}")

    # 2. Collect 4 official CSV files
    official_root = TIDDataset(config.dataset, split="train")._autsl_root()
    csv_files = ["train_labels.csv", "val_labels.csv", "test_labels.csv", "SignList_ClassId_TR_EN.csv"]
    csv_entries: list[tuple[Path, str]] = []
    for name in csv_files:
        p = official_root / name
        if not p.is_file():
            raise FileNotFoundError(f"CSV not found: {p}")
        csv_entries.append((p, name))

    all_entries = csv_entries + video_entries

    print(f"Arşiv oluşturuluyor -> {output_zip}")
    # Compression: STORE (ZIP_STORED) because MP4 is already h264 compressed;
    # ZIP_STORED avoids wasting CPU compressing already-compressed video and is 10x faster to write & extract!
    with zipfile.ZipFile(output_zip, "w", compression=zipfile.ZIP_STORED) as zf:
        for src, arc in tqdm(all_entries, desc="Paketleniyor"):
            zf.write(src, arcname=arc)

    size_mb = output_zip.stat().st_size / (1024 * 1024)
    print(f"✅ Paket tamamlandı: {output_zip} ({size_mb:.1f} MB, {len(all_entries)} dosya)")


def package_caches(output_zip: Path):
    print("=" * 60)
    print("AUTSL 10 SINIF ÖNBELLEK (CACHE) PAKETLEME")
    print("=" * 60)

    cache_dirs = [
        ROOT / "cache/autsl_10_features",
        ROOT / "cache/autsl_10_features_centerrepair_strict",
    ]

    for d in cache_dirs:
        if not d.is_dir():
            raise FileNotFoundError(f"Cache dir not found: {d}")

    output_zip.parent.mkdir(parents=True, exist_ok=True)

    files_to_pack: list[tuple[Path, str]] = []
    for d in cache_dirs:
        for p in d.rglob("*"):
            if p.is_file():
                arcname = str(p.relative_to(ROOT / "cache"))
                files_to_pack.append((p, arcname))

    print(f"Toplam cache dosyası: {len(files_to_pack)} adet (.npz + manifest.json)")
    print(f"Arşiv oluşturuluyor -> {output_zip}")

    with zipfile.ZipFile(output_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for src, arc in tqdm(files_to_pack, desc="Cache Paketleniyor"):
            zf.write(src, arcname=arc)

    size_mb = output_zip.stat().st_size / (1024 * 1024)
    print(f"✅ Cache paketi tamamlandı: {output_zip} ({size_mb:.1f} MB, {len(files_to_pack)} dosya)")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["videos", "caches", "all"], default="videos")
    parser.add_argument("--output_dir", type=Path, default=ROOT / "outputs")
    args = parser.parse_args()

    if args.mode in ("videos", "all"):
        package_videos(args.output_dir / "autsl_10_raw_dataset.zip")
    if args.mode in ("caches", "all"):
        package_caches(args.output_dir / "autsl_10_caches.zip")


if __name__ == "__main__":
    main()
