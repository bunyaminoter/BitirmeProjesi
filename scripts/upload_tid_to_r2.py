"""
Cloudflare R2 TİD Videoları Otomatik Yükleyici.

Kullanım:
    python scripts/upload_tid_to_r2.py --account-id <ACCOUNT_ID> --access-key <ACCESS_KEY> --secret-key <SECRET_KEY> --bucket <BUCKET_NAME>

Veya script içine doğrudan değişkenleri yazarak:
    python scripts/upload_tid_to_r2.py
"""

import os
import sys
import argparse
from pathlib import Path
import boto3
from botocore.config import Config

# Ekrana UTF-8 bas
sys.stdout.reconfigure(encoding="utf-8")

# Sabitler
DEFAULT_DIR = Path(r"C:\Users\Bünyamin\Desktop\TID_CLAUDFLARE")

ACCOUNT_ID = "74f50bbb4373519d6ee75969f1c523d8"
ACCESS_KEY_ID = "97722994010ae1d3c3fabf87bf0db485"
SECRET_ACCESS_KEY = "919b3ca21c3561c216a9354f6a4a6d84773b349970e0a4ee5564404506353749"
BUCKET_NAME = "tid-dictionary"


def parse_args():
    parser = argparse.ArgumentParser(description="Upload TID videos to Cloudflare R2")
    parser.add_argument("--account-id", default=ACCOUNT_ID, help="Cloudflare Account ID")
    parser.add_argument("--access-key", default=ACCESS_KEY_ID, help="R2 Access Key ID")
    parser.add_argument("--secret-key", default=SECRET_ACCESS_KEY, help="R2 Secret Access Key")
    parser.add_argument("--bucket", default=BUCKET_NAME, help="R2 Bucket Name")
    parser.add_argument("--dir", default=str(DEFAULT_DIR), help="Directory of TID MP4 videos")
    return parser.parse_args()


def upload_videos():
    args = parse_args()

    account_id = args.account_id.strip()
    access_key = args.access_key.strip()
    secret_key = args.secret_key.strip()
    bucket_name = args.bucket.strip()
    source_dir = Path(args.dir)

    if not account_id or not access_key or not secret_key:
        print("❌ HATA: Account ID, Access Key veya Secret Key eksik!")
        print("Lütfen komut satırından parametre verin veya script içindeki değişkenleri doldurun.")
        print("Örnek:")
        print("  python scripts/upload_tid_to_r2.py --account-id <ID> --access-key <KEY> --secret-key <SECRET> --bucket <BUCKET>")
        return

    if not source_dir.exists():
        print(f"❌ HATA: Video klasörü bulunamadı: {source_dir}")
        return

    mp4_files = sorted(list(source_dir.glob("*.mp4")))
    if not mp4_files:
        print(f"❌ HATA: Klasörde MP4 dosyası bulunamadı: {source_dir}")
        return

    print("=" * 70)
    print("🚀 CLOUDFLARE R2 TİD VİDEO YÜKLEYİCİ BAŞLATILIYOR")
    print(f"📁 Kaynak Klasör : {source_dir}")
    print(f"📦 Dosya Sayısı  : {len(mp4_files)} adet MP4 video")
    print(f"🪣 Hedef Bucket  : {bucket_name}")
    print("=" * 70)

    endpoint_url = f"https://{account_id}.r2.cloudflarestorage.com"

    s3_client = boto3.client(
        service_name="s3",
        endpoint_url=endpoint_url,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        region_name="auto",
        config=Config(s3={"addressing_style": "path"})
    )

    success_count = 0
    fail_count = 0

    for i, file_path in enumerate(mp4_files, 1):
        filename = file_path.name
        file_size_kb = file_path.stat().st_size / 1024

        try:
            print(f"[{i:03d}/{len(mp4_files)}] Yükleniyor: {filename} ({file_size_kb:.1f} KB)...", end=" ")
            s3_client.upload_file(
                Filename=str(file_path),
                Bucket=bucket_name,
                Key=filename,
                ExtraArgs={
                    "ContentType": "video/mp4",
                    "CacheControl": "public, max-age=31536000, immutable"
                }
            )
            print("✅ Başarılı")
            success_count += 1
        except Exception as e:
            print(f"❌ Hata: {e}")
            fail_count += 1

    print("=" * 70)
    print(f"🎉 YÜKLEME TAMAMLANDI!")
    print(f"✅ Başarılı: {success_count}")
    if fail_count > 0:
        print(f"❌ Başarısız: {fail_count}")
    print("=" * 70)


if __name__ == "__main__":
    upload_videos()
