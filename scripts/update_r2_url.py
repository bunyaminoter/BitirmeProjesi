"""
Yeni Cloudflare R2 Public URL'ini mobil uygulamaya ve SQLite veritabanına bağlama scripti.

Kullanım:
    python scripts/update_r2_url.py --url https://pub-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx.r2.dev
"""

import sys
import sqlite3
import argparse
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

mobile_root = Path(r"C:\Projelerim\BitirmeProjesiBirlesim\SignLanguageMobileApp")
constants_file = mobile_root / "lib" / "config" / "constants.dart"
db_file = mobile_root / "assets" / "database" / "dictionary.db"


def parse_args():
    parser = argparse.ArgumentParser(description="Update R2 Base URL in Mobile App")
    parser.add_argument("--url", required=True, help="New Cloudflare R2 Public URL (e.g. https://pub-xxx.r2.dev)")
    return parser.parse_args()


def main():
    args = parse_args()
    raw_url = args.url.strip().rstrip("/")

    print("=" * 70)
    print(f"🔗 R2 YENİ URL BAĞLANIYOR: {raw_url}")
    print("=" * 70)

    # 1. constants.dart Güncelle
    if constants_file.exists():
        code = constants_file.read_text(encoding="utf-8")
        if "tidR2BaseUrl" in code:
            import re
            code = re.sub(r"static const String tidR2BaseUrl = '.*?';", f"static const String tidR2BaseUrl = '{raw_url}';", code)
        else:
            # cloudflareWorkerProxyUrl'in altına ekle
            target_line = "static const String cloudflareWorkerProxyUrl = 'https://r2-video-proxy.bunyaminoterr.workers.dev';"
            replacement = f"{target_line}\n  static const String tidR2BaseUrl = '{raw_url}';"
            code = code.replace(target_line, replacement)

        constants_file.write_text(code, encoding="utf-8")
        print(f"✅ constants.dart güncellendi: tidR2BaseUrl = '{raw_url}'")
    else:
        print(f"⚠️ constants.dart bulunamadı: {constants_file}")

    # 2. SQLite dictionary.db Güncelle
    if db_file.exists():
        conn = sqlite3.connect(str(db_file))
        cur = conn.cursor()

        # Mevcut kayıtların videoUrl kolonunu güncelle
        cur.execute("SELECT id, videoUrl FROM dictionary")
        rows = cur.fetchall()

        updated_count = 0
        for row_id, old_url in rows:
            filename = old_url.split("/")[-1]
            new_url = f"{raw_url}/{filename}"
            cur.execute("UPDATE dictionary SET videoUrl = ? WHERE id = ?", (new_url, row_id))
            updated_count += 1

        conn.commit()
        conn.close()
        print(f"✅ dictionary.db güncellendi: {updated_count} kelimenin videoUrl'i '{raw_url}/...' olarak ayarlandı.")
    else:
        print(f"⚠️ dictionary.db bulunamadı: {db_file}")

    print("=" * 70)
    print("🎉 MOBİL UYGULAMA VE VERİTABANI YENİ R2 BUCKET'INA BAĞLANDI!")
    print("=" * 70)


if __name__ == "__main__":
    main()
