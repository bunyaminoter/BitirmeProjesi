"""
AUTSL Veri Setini Google Drive Desktop (G:) Üzerine Eşitleme Betiği.

Sıralama:
1. 200 adet pilot video (100 train + 100 val) öncelikli kopyalanır (Google Drive hemen buluta yüklesin).
2. Kalan tüm val videoları (4,418 video) kopyalanır.
3. Kalan tüm train videoları (28,142 video) kopyalanır.
4. Kalan tüm test videoları (3,742 video) kopyalanır.
5. Tüm CSV dosyaları kopyalanır.

Zaten kopyalanmış dosyalar otomatik atlanır. Çoklu iş parçacığı (multi-threading) ile hızlı kopyalama yapar.
"""

import os
import sys
import json
import shutil
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

DESKTOP_AUTSL = Path(r"C:\Users\Bünyamin\Desktop\AUTSL_Dataset")


def get_gdrive_dataset_dir() -> Path:
    g_root = Path("G:/")
    if not g_root.exists():
        raise FileNotFoundError("G: sürücüsü bulunamadı! Google Drive Desktop açık mı?")
    drive_dirs = [d for d in g_root.iterdir() if "Drive" in d.name and d.is_dir()]
    if not drive_dirs:
        raise FileNotFoundError("G: üzerinde Drive klasörü bulunamadı!")
    dataset_dir = drive_dirs[0] / "BitirmeProjesi" / "AUTSL_Dataset"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    return dataset_dir


def copy_single_file(src_dst: tuple[Path, Path]) -> bool:
    src, dst = src_dst
    if dst.exists():
        return False
    try:
        shutil.copyfile(src, dst)
        return True
    except Exception as e:
        print(f"Hata ({src.name}): {e}", flush=True)
        return False


def main():
    print("=" * 60)
    print("🚀 AUTSL Veri Seti Google Drive Senkronizasyonu Başlatılıyor")
    print("=" * 60)

    gdrive_dataset = get_gdrive_dataset_dir()
    print(f"Kaynak Klasör : {DESKTOP_AUTSL}")
    print(f"Hedef Klasör  : {gdrive_dataset}\n")

    # 1. CSV dosyalarını kopyala
    print("[1/5] CSV etiket dosyaları kopyalanıyor...")
    for csv_file in DESKTOP_AUTSL.glob("*.csv"):
        dst_csv = gdrive_dataset / csv_file.name
        shutil.copyfile(csv_file, dst_csv)
        print(f"  ✅ {csv_file.name}")

    # 2. Pilot 200 videoyu öncelikli kopyala
    pilot_file = Path("data/autsl/autsl_100_preprocessing_pilot_ids.json")
    if pilot_file.exists():
        print("\n[2/5] 200 Adet Pilot Video Öncelikli Olarak Kopyalanıyor...")
        pilot_data = json.loads(pilot_file.read_text(encoding="utf-8"))
        pilot_tasks = []
        for split in ["val", "train"]:
            split_dst = gdrive_dataset / split
            split_dst.mkdir(parents=True, exist_ok=True)
            for vid_id in pilot_data.get(split, []):
                fname = f"{vid_id}.mp4"
                src = DESKTOP_AUTSL / split / fname
                dst = split_dst / fname
                if src.exists():
                    pilot_tasks.append((src, dst))

        with ThreadPoolExecutor(max_workers=8) as ex:
            copied = sum(list(ex.map(copy_single_file, pilot_tasks)))
        print(f"  ✅ {len(pilot_tasks)} pilot videonun kontrolü tamamlandı ({copied} yeni kopyalandı).")
        print("  💡 Google Drive Desktop bu 200 videoyu hemen buluta yüklemeye başladı!")

    # 3. Kalan VAL videoları (Küçük küme - 4.418 video)
    for split in ["val", "train", "test"]:
        src_split = DESKTOP_AUTSL / split
        if not src_split.exists():
            continue
        dst_split = gdrive_dataset / split
        dst_split.mkdir(parents=True, exist_ok=True)

        all_mp4s = sorted(list(src_split.glob("*.mp4")))
        tasks = [(f, dst_split / f.name) for f in all_mp4s]
        total = len(tasks)
        print(f"\n[*] {split.upper()} Klasörü Senkronize Ediliyor: Toplam {total} Video...")

        t0 = time.time()
        copied_count = 0
        skipped_count = 0

        # Çoklu iş parçacığıyla kopyalama
        batch_size = 1000
        for i in range(0, total, batch_size):
            chunk = tasks[i:i+batch_size]
            with ThreadPoolExecutor(max_workers=8) as ex:
                results = list(ex.map(copy_single_file, chunk))
            c = sum(results)
            copied_count += c
            skipped_count += len(chunk) - c
            print(f"  -> {min(i + batch_size, total)}/{total} video kontrol edildi ({copied_count} kopyalandı, {skipped_count} önceden vardı)...", flush=True)

        t1 = time.time()
        print(f"  ✅ {split.upper()} bitti: {copied_count} yeni kopyalandı, {skipped_count} atlandı ({t1 - t0:.1f} sn).")

    print("\n" + "=" * 60)
    print("🎉 TÜM AUTSL VİDEOLARI GOOGLE DRIVE'A AKTARILDI!")
    print(f"Hedef: {gdrive_dataset}")
    print("Google Drive Masaüstü arka planda buluta yüklemeyi sürdürecektir.")
    print("=" * 60)


if __name__ == "__main__":
    main()
