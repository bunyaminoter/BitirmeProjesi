import os
import shutil
import sqlite3
import re
from pathlib import Path

# --- AYARLAR ---
# 1. Videoların bulunduğu ana klasör (İçinde tüm mp4'lerin olduğu klasör)
SOURCE_VIDEO_DIR = r"C:\Projelerim\BitirmeProjesiBirlesim" 

# 2. Seçilen (en kaliteli) videoların kopyalanacağı yeni klasör (Cloudflare'e bunları yükleyeceksiniz)
TARGET_VIDEO_DIR = r"C:\Projelerim\BitirmeProjesiBirlesim\Selected_Videos_For_Cloudflare"

# 3. Güncellenecek SQLite Veritabanı
DB_PATH = r"C:\Projelerim\BitirmeProjesiBirlesim\SignLanguageMobileApp\assets\database\dictionary.db"

# 4. Cloudflare Public URL
CLOUDFLARE_BASE_URL = "https://pub-5498c4b95ed24665abfc577f97077874.r2.dev"


def get_word_from_filename(filename):
    """
    Dosya adından kelimeyi çıkarır ve sonundaki sayıları siler (DOG1 -> DOG).
    """
    name = os.path.splitext(filename)[0]
    if '-' in name:
        name = name.split('-')[-1]
    name = name.upper().strip()
    
    # Sonundaki rakamları (1, 2, 3 vb.) sil (Örn: BITE1 -> BITE, DOG3 -> DOG)
    name = re.sub(r'\d+$', '', name)
    return name


def main():
    print("1. Videolar taranıyor...")
    
    word_to_files = {}
    source_path = Path(SOURCE_VIDEO_DIR)
    
    for mp4_file in source_path.rglob("*.mp4"):
        if "Selected_Videos_For_Cloudflare" in str(mp4_file):
            continue
            
        word = get_word_from_filename(mp4_file.name)
        size = mp4_file.stat().st_size
        
        if word not in word_to_files:
            word_to_files[word] = []
        
        word_to_files[word].append({
            "path": mp4_file,
            "name": mp4_file.name,
            "size": size
        })
        
    if not word_to_files:
        print("Hiç .mp4 videosu bulunamadı! SOURCE_VIDEO_DIR yolunu kontrol edin.")
        return
        
    print(f"Toplam {len(word_to_files)} farklı kelime için videolar bulundu.")
    os.makedirs(TARGET_VIDEO_DIR, exist_ok=True)
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    print("\n2. En yüksek boyutlu (en kaliteli) videolar seçiliyor ve veritabanı güncelleniyor...")
    
    selected_count = 0
    updated_db_count = 0
    
    for word, files in word_to_files.items():
        # Boyuta göre büyükten küçüğe sırala ve en büyük olanı al
        files.sort(key=lambda x: x["size"], reverse=True)
        best_file = files[0]
        selected_count += 1
        
        # Seçilen videoyu Cloudflare klasörüne kopyala
        target_path = os.path.join(TARGET_VIDEO_DIR, best_file["name"])
        if not os.path.exists(target_path):
            shutil.copy2(best_file["path"], target_path)
            
        # Veritabanını güncelle
        video_url = f"{CLOUDFLARE_BASE_URL}/{best_file['name']}"
        
        cursor.execute("SELECT id FROM dictionary WHERE word = ?", (word,))
        if cursor.fetchone():
            cursor.execute("UPDATE dictionary SET videoUrl = ? WHERE word = ?", (video_url, word))
            updated_db_count += 1
        else:
            print(f"Uyarı: '{word}' kelimesi veritabanında bulunamadı!")

    conn.commit()
    conn.close()
    
    print("\n" + "="*50)
    print("İŞLEM TAMAMLANDI! ✅")
    print("="*50)
    print(f"- Ayrılan Video Sayısı: {selected_count}")
    print(f"- DB'de Güncellenen Kelime: {updated_db_count}")
    print(f"\nLÜTFEN ŞU KLASÖRÜ AÇIN:\n{TARGET_VIDEO_DIR}")
    print("Bu klasördeki TÜM VİDEOLARI doğrudan Cloudflare'e sürükleyip bırakabilirsiniz!")

if __name__ == "__main__":
    main()
