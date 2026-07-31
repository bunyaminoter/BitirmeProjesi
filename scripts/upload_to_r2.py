import os
import boto3
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

# --- AYARLAR ---
FOLDER_PATH = r"C:\Users\Bünyamin\Desktop\Secilen_Videolar_Cloudflare"
BUCKET_NAME = "asl-dictionary-bitirme"
# GÜVENLİK İÇİN API ANAHTARLARI KALDIRILDI - GITHUB'A YÜKLENEBİLİR
ACCOUNT_ID = "YOUR_ACCOUNT_ID_HERE"
ACCESS_KEY = "YOUR_ACCESS_KEY_HERE"
SECRET_KEY = "YOUR_SECRET_KEY_HERE"

s3 = boto3.client('s3',
  endpoint_url = f"https://{ACCOUNT_ID}.r2.cloudflarestorage.com",
  aws_access_key_id = ACCESS_KEY,
  aws_secret_access_key = SECRET_KEY,
  region_name="auto" 
)

def get_existing_keys():
    print("Cloudflare R2'deki mevcut dosyalar kontrol ediliyor...")
    existing = set()
    paginator = s3.get_paginator('list_objects_v2')
    for page in paginator.paginate(Bucket=BUCKET_NAME):
        for obj in page.get('Contents', []):
            existing.add(obj['Key'])
    return existing

def upload_file(file_path):
    filename = os.path.basename(file_path)
    try:
        s3.upload_file(str(file_path), BUCKET_NAME, filename)
        return True, filename
    except Exception as e:
        return False, f"{filename}: {e}"

def main():
    if not os.path.exists(FOLDER_PATH):
        print(f"HATA: Klasör bulunamadı: {FOLDER_PATH}")
        return

    mp4_files = list(Path(FOLDER_PATH).rglob("*.mp4"))
    existing_keys = get_existing_keys()
    
    missing_files = [f for f in mp4_files if os.path.basename(f) not in existing_keys]
    
    total_missing = len(missing_files)
    if total_missing == 0:
        print("Tebrikler! Eksik hiç dosya yok, 2157 dosyanın tümü Cloudflare R2'de mevcut!")
        return

    print(f"Tespit edilen eksik dosya sayısı: {total_missing}")
    print("Eksik dosyalar güvenli modda (5 paralel bağlantı ile) yükleniyor...\n")
    
    start_time = time.time()
    success_count = 0
    error_count = 0
    
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(upload_file, f): f for f in missing_files}
        
        for i, future in enumerate(as_completed(futures), 1):
            success, msg = future.result()
            if success:
                success_count += 1
                print(f"Yüklendi [{i}/{total_missing}]: {msg}")
            else:
                print(f"HATA [{i}/{total_missing}]: {msg}")
                error_count += 1

    elapsed_time = time.time() - start_time
    print(f"\nEKSIK DOSYA YÜKLEMESİ TAMAMLANDI! (Süre: {elapsed_time:.1f} saniye)")
    print(f"Başarıyla tamamlanan: {success_count}, Hatalı: {error_count}")

if __name__ == '__main__':
    main()
