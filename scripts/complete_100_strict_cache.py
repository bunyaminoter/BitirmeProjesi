"""Complete the missing strict hand crop cache files locally and sync to Google Drive.

Reads base16 landmarks from local cache/autsl_100_features_base16,
crops hand regions with 1.5x scale and center repair from C:\\Users\\Bünyamin\\Desktop\\AUTSL_Dataset,
and syncs to G:\\Drive'ım\\BitirmeProjesi\\cache\\autsl_100_features_centerrepair_strict.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.core.config import load_config
from src.data.datasets.tid_dataset import TIDDataset

DRIVE_ROOT = Path(r"G:\Drive'ım\BitirmeProjesi")
DATA_DIR = Path(r"C:\Users\Bünyamin\Desktop\AUTSL_Dataset")


def copy_parallel(src_dir: Path, dst_dir: Path, file_ids: list[str], desc: str, max_workers: int = 16):
    """Copy files in parallel with progress bar, skipping existing."""
    dst_dir.mkdir(parents=True, exist_ok=True)
    to_copy = [vid for vid in file_ids if not (dst_dir / f"{vid}.npz").is_file()]
    if not to_copy:
        print(f"   ✓ {desc}: Tüm dosyalar ({len(file_ids)}) zaten mevcut, atlandı.")
        return

    print(f"   ⏳ {desc}: {len(to_copy)} / {len(file_ids)} dosya kopyalanıyor...")

    def _copy(vid):
        src = src_dir / f"{vid}.npz"
        dst = dst_dir / f"{vid}.npz"
        if src.is_file():
            shutil.copy2(src, dst)

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        list(tqdm(ex.map(_copy, to_copy), total=len(to_copy), desc=desc))


def main():
    print("=" * 70)
    print("🚀 AUTSL 100 STRICT ÖNBELLEK TAMAMLAMA VE SENKRONİZASYON BAŞLATILDI")
    print("=" * 70)

    os.environ["AUTSL_DATA_DIR"] = str(DATA_DIR)
    config = load_config("configs/experiment/autsl_100_preprocess_base16.yaml")

    ds_train = TIDDataset(config.dataset, split="train")
    ds_val = TIDDataset(config.dataset, split="val")

    all_train_ids = [s.video_id for s in ds_train.samples]
    all_val_ids = [s.video_id for s in ds_val.samples]
    print(f"📋 Beklenen Toplam: Train={len(all_train_ids)}, Val={len(all_val_ids)} (Toplam={len(all_train_ids)+len(all_val_ids)})")

    # 2. Drive'da mevcut dosyaları tara
    drive_strict_dir = DRIVE_ROOT / "cache" / "autsl_100_features_centerrepair_strict"
    drive_train_dir = drive_strict_dir / "train"
    drive_val_dir = drive_strict_dir / "val"

    drive_train_ids = set()
    if drive_train_dir.is_dir():
        drive_train_ids = {p.stem for p in drive_train_dir.glob("*.npz")}

    drive_val_ids = set()
    if drive_val_dir.is_dir():
        drive_val_ids = {p.stem for p in drive_val_dir.glob("*.npz")}

    missing_train = sorted(list(set(all_train_ids) - drive_train_ids))
    missing_val = sorted(list(set(all_val_ids) - drive_val_ids))

    print(f"📊 Drive Mevcut: Train={len(drive_train_ids)}, Val={len(drive_val_ids)}")
    print(f"🔍 Eksik Kalan: Train={len(missing_train)}, Val={len(missing_val)} (Toplam Eksik={len(missing_train)+len(missing_val)})")

    if not missing_train and not missing_val:
        print("\n🎉 Tebrikler! Bütün 14.368 dosya Drive'da zaten eksiksiz mevcut!")
        return

    # 3. Eksik ID listesini kaydet
    missing_ids = {"train": missing_train, "val": missing_val}
    missing_ids_file = PROJECT_ROOT / "data" / "autsl" / "autsl_100_missing_ids.json"
    with open(missing_ids_file, "w", encoding="utf-8") as f:
        json.dump(missing_ids, f, indent=2)
    print(f"\n💾 Eksik ID listesi kaydedildi: {missing_ids_file}")

    # 4. Yerel base16 klasörünün hazır olduğundan emin ol
    local_base16 = PROJECT_ROOT / "cache" / "autsl_100_features_base16"
    local_base16.mkdir(parents=True, exist_ok=True)
    drive_base16 = DRIVE_ROOT / "cache" / "autsl_100_features_base16"

    # Manifest kopyala
    if not (local_base16 / "manifest.json").exists():
        manifest_data = (drive_base16 / "manifest.json").read_bytes()
        (local_base16 / "manifest.json").write_bytes(manifest_data)

    # 5. Eksik dosyalar için base16 dosyalarını Drive'dan yerel SSD'ye çek (çok kanallı hızlı kopya)
    print("\n📥 Eksik videolar için base16 önbelleği yerel SSD'ye çekiliyor...")
    if missing_val:
        copy_parallel(drive_base16 / "val", local_base16 / "val", missing_val, "base16 val indir", max_workers=16)
    if missing_train:
        copy_parallel(drive_base16 / "train", local_base16 / "train", missing_train, "base16 train indir", max_workers=16)

    # 6. Yerel strict üretimi çalıştır (MediaPipe yok, sadece hızlı kutu kırpma)
    local_strict = PROJECT_ROOT / "cache" / "autsl_100_features_centerrepair_strict"
    local_strict.mkdir(parents=True, exist_ok=True)

    splits_to_run = []
    if missing_val:
        splits_to_run.append("val")
    if missing_train:
        splits_to_run.append("train")

    print(f"\n⚡ Yerel strict onarımı çalıştırılıyor: splits={splits_to_run}...")
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "repair_hand_crops.py"),
        "--config", "configs/experiment/autsl_100_preprocess_base16.yaml",
        "--data_dir", str(DATA_DIR),
        "--dest_cache", str(local_strict),
        "--splits", *splits_to_run,
        "--video_ids_file", str(missing_ids_file),
        "--box_scale", "1.5",
        "--center_repair",
        "--center_policy", "configs/experiment/center_repair_strict.json",
        "--max_alignment_error", "0",
    ]
    t0 = time.time()
    res = subprocess.run(cmd)
    dt = time.time() - t0
    print(f"\n⏱️ Yerel kırpım tamamlandı! Süre: {dt:.1f} saniye ({dt/60:.2f} dakika)")

    if res.returncode != 0:
        print(f"❌ HATA: repair_hand_crops.py {res.returncode} koduyla başarısız oldu!")
        sys.exit(res.returncode)

    # 7. Üretilen strict dosyalarını Google Drive'a kopyala (çok kanallı hızlı kopya)
    print("\n🚀 Üretilen strict dosyaları Google Drive'a yükleniyor...")
    drive_strict_dir.mkdir(parents=True, exist_ok=True)
    if (local_strict / "manifest.json").exists():
        (drive_strict_dir / "manifest.json").write_bytes((local_strict / "manifest.json").read_bytes())

    if missing_val:
        copy_parallel(local_strict / "val", drive_strict_dir / "val", missing_val, "strict val yükle", max_workers=16)
    if missing_train:
        copy_parallel(local_strict / "train", drive_strict_dir / "train", missing_train, "strict train yükle", max_workers=16)

    # 8. Son Doğrulama
    final_train_count = len(list(drive_train_dir.glob("*.npz"))) if drive_train_dir.is_dir() else 0
    final_val_count = len(list(drive_val_dir.glob("*.npz"))) if drive_val_dir.is_dir() else 0
    total_count = final_train_count + final_val_count

    print("\n" + "=" * 70)
    print(f"🏁 DRIVE SON DURUM: Train={final_train_count}/12405, Val={final_val_count}/1963 | Toplam={total_count}/14368")
    if total_count == 14368:
        print("🎉 TEBRİKLER! Bütün 14.368 dosya başarıyla tamamlandı ve Drive'a eşitlendi!")
    else:
        print(f"⚠️ UYARI: Toplam {total_count} dosya var (Beklenen: 14368)")
    print("=" * 70)


if __name__ == "__main__":
    main()
