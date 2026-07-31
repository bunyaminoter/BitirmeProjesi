import os
import sqlite3
import re
from pathlib import Path

# --- AYARLAR ---
# Google Drive'dan bilgisayarınıza indirdiğiniz ve zipten çıkarttığınız klasörün yolu:
DOWNLOADED_FOLDER = r"C:\Users\Bünyamin\Desktop\Secilen_Videolar_Cloudflare"

# Güncellenecek SQLite Veritabanı
DB_PATH = r"C:\Projelerim\BitirmeProjesiBirlesim\SignLanguageMobileApp\assets\database\dictionary.db"

# Cloudflare Public URL
CLOUDFLARE_BASE_URL = "https://pub-5498c4b95ed24665abfc577f97077874.r2.dev"


def get_word_from_filename(filename):
    name = os.path.splitext(filename)[0]
    if '-' in name:
        name = name.split('-')[-1]
    name = name.upper().strip()
    
    # Sonundaki boşluk ve rakamları temizle (Örn: "SOON 3" -> "SOON")
    name = re.sub(r'[\s\d]+$', '', name)
    
    # Başında "SEED" varsa onu da temizle (Örn: "SEEDSOON" -> "SOON")
    if name.startswith("SEED"):
        name = name[4:]
        
    return name

def main():
    if not os.path.exists(DOWNLOADED_FOLDER):
        print(f"HATA: {DOWNLOADED_FOLDER} klasörü bulunamadı!")
        print("Lütfen dosyaları indirdiğiniz yola göre DOWNLOADED_FOLDER değişkenini güncelleyin.")
        return

    print("1. İndirilen videolar taranıyor...")
    
    mp4_files = list(Path(DOWNLOADED_FOLDER).rglob("*.mp4"))
    print(f"Toplam {len(mp4_files)} adet video bulundu.")
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    import uuid # Gerekli import'u ekliyoruz

    print("\n2. Veritabanı güncelleniyor (Olmayan kelimeler eklenecek)...")
    
    updated_db_count = 0
    inserted_db_count = 0
    
    for mp4_file in mp4_files:
        word = get_word_from_filename(mp4_file.name)
        video_url = f"{CLOUDFLARE_BASE_URL}/{mp4_file.name}"
        
        # Kelime veritabanında var mı?
        cursor.execute("SELECT id FROM dictionary WHERE word = ?", (word,))
        result = cursor.fetchone()
        
        if result:
            # Kelime varsa sadece linkini güncelle
            cursor.execute("UPDATE dictionary SET videoUrl = ? WHERE word = ?", (video_url, word))
            updated_db_count += 1
        else:
            # Kelime veritabanında hiç yoksa, YENİDEN EKLE!
            new_id = str(uuid.uuid4())
            cursor.execute(
                "INSERT INTO dictionary (id, word, category, description, difficulty, videoUrl) VALUES (?, ?, ?, ?, ?, ?)",
                (new_id, word, "Genel", "Sözlükten seçilen kelime.", "Orta", video_url)
            )
            inserted_db_count += 1

    # Değişiklikleri veritabanına kaydet
    conn.commit()
    conn.close()
    
    print("\n" + "="*50)
    print("İŞLEM TAMAMLANDI!")
    print("="*50)
    print(f"Veritabanında halihazırda var olup linki GÜNCELLENEN kelime sayısı: {updated_db_count}")
    print(f"Veritabanına yepyeni EKLENEN kelime sayısı: {inserted_db_count}")
        
if __name__ == "__main__":
    main()
