"""
TİD Videolarını Web Uyumlu H.264 (AVC1) + Faststart Formatına Dönüştürüp R2'ye Yükleme Scripti.
"""

import sys
import subprocess
from pathlib import Path
import boto3
from botocore.config import Config
import imageio_ffmpeg

sys.stdout.reconfigure(encoding="utf-8")

SOURCE_DIR = Path(r"C:\Users\Bünyamin\Desktop\TID_CLAUDFLARE")
CONVERTED_DIR = SOURCE_DIR / "h264_web"
CONVERTED_DIR.mkdir(parents=True, exist_ok=True)

ACCOUNT_ID = "74f50bbb4373519d6ee75969f1c523d8"
ACCESS_KEY_ID = "97722994010ae1d3c3fabf87bf0db485"
SECRET_ACCESS_KEY = "919b3ca21c3561c216a9354f6a4a6d84773b349970e0a4ee5564404506353749"
BUCKET_NAME = "tid-dictionary"

ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()


def main():
    mp4_files = sorted(list(SOURCE_DIR.glob("[0-9]*.mp4")))
    print("=" * 70)
    print(f"🎬 H.264 (AVC1) DÖNÜŞTÜRÜCÜ VE R2 YÜKLEYİCİ")
    print(f"📦 Toplam Video: {len(mp4_files)} adet")
    print("=" * 70)

    # S3 Client
    s3 = boto3.client(
        "s3",
        endpoint_url=f"https://{ACCOUNT_ID}.r2.cloudflarestorage.com",
        aws_access_key_id=ACCESS_KEY_ID,
        aws_secret_access_key=SECRET_ACCESS_KEY,
        region_name="auto",
        config=Config(s3={"addressing_style": "path"})
    )

    success_count = 0

    for idx, in_file in enumerate(mp4_files, 1):
        out_file = CONVERTED_DIR / in_file.name

        # 1. FFmpeg ile H.264 (AVC1) + Faststart (moov atom başta) dönüştür
        cmd = [
            ffmpeg_exe, "-y",
            "-i", str(in_file),
            "-c:v", "libx264",
            "-preset", "veryfast",
            "-crf", "23",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            str(out_file)
        ]
        res = subprocess.run(cmd, capture_output=True)
        if res.returncode != 0:
            print(f"[{idx:03d}/{len(mp4_files)}] ❌ Dönüştürme hatası: {in_file.name}")
            continue

        orig_kb = in_file.stat().st_size / 1024
        new_kb = out_file.stat().st_size / 1024

        # 2. R2'ye yükle (üzerine yaz)
        try:
            s3.upload_file(
                Filename=str(out_file),
                Bucket=BUCKET_NAME,
                Key=in_file.name,
                ExtraArgs={
                    "ContentType": "video/mp4",
                    "CacheControl": "public, max-age=31536000, immutable"
                }
            )
            print(f"[{idx:03d}/{len(mp4_files)}] ✅ {in_file.name}: {orig_kb:.0f}KB -> {new_kb:.0f}KB (R2 Güncellendi)")
            success_count += 1
        except Exception as e:
            print(f"[{idx:03d}/{len(mp4_files)}] ❌ R2 yükleme hatası ({in_file.name}): {e}")

    # Orijinal klasördeki mp4'leri de H.264 sürümleriyle değiştir
    print("\n📁 Masaüstündeki videolar H.264 sürümleriyle eşitleniyor...")
    for h264_file in CONVERTED_DIR.glob("*.mp4"):
        target_path = SOURCE_DIR / h264_file.name
        import shutil
        shutil.copy2(h264_file, target_path)

    print("=" * 70)
    print(f"🎉 İŞLEM TAMAMLANDI! {success_count} / {len(mp4_files)} video H.264 olarak R2'ye yüklendi.")
    print("=" * 70)


if __name__ == "__main__":
    main()
