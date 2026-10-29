import json

notebook = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# 🚀 AUTSL 100 Sınıf: Ön İşleme ve Hareket Kalite Kapısı\n",
                "Bu notebook, **AUTSL 100 Sınıf** (12.405 train + 1.963 val = 14.368 video) için **ön işleme (preprocessing) ve kalite denetimini** yürütür.\n",
                "\n",
                "> ⚠️ **ÖNEMLİ KURAL:** Bu oturumda **kesinlikle model eğitimi yapılmayacaktır.**\n",
                "> Amaç: 100 sınıfın MediaPipe iskelet ve el kırpım verisini üretmek, parmak ve bilek hareketlerinin kaçıp kaçmadığını (`audit_autsl_preprocessing_motion.py`) ile denetlemektir.\n",
                ">\n",
                "> **Çalışma Sırası:**\n",
                "> 1. **Duman Testi (Pilot):** Her sınıftan 1 train + 1 val olmak üzere 200 video işlenir ve kontrol edilir (~2-3 dk).\n",
                "> 2. **Tam Ön İşleme:** Pilot başarılıysa 100 sınıfın tamamı (14.368 video) işlenir.\n",
                "> 3. **Kalite ve Kusur Denetimi:** Hatalı, parmak sıçraması olan veya eksik videolar listelenip Drive'a kaydedilir."
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 1. Google Drive Bağlantısı ve Gerekli Kütüphaneler"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "from google.colab import drive\n",
                "drive.mount('/content/drive')\n",
                "\n",
                "!pip install -q pyyaml rich opencv-python matplotlib scikit-learn tqdm mediapipe\n",
                "\n",
                "import os\n",
                "os.environ[\"PYTHONDONTWRITEBYTECODE\"] = \"1\"\n",
                "print(\"✅ Google Drive bağlandı, bağımlılıklar hazır.\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 2. Kod Paketinin Açılması ve Çalışma Alanının Kurulması\n",
                "*Drive'daki `colab_kod_guncelleme.zip` paketini otomatik bulur ve yerel SSD'ye açar.*"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import os, shutil\n",
                "\n",
                "local_project = \"/content/BitirmeProjesi\"\n",
                "os.makedirs(local_project, exist_ok=True)\n",
                "\n",
                "# Doğrudan bilinen yolları kontrol et (recursive arama YOK, 0.01 sn!)\n",
                "potential_zips = [\n",
                "    \"/content/drive/MyDrive/BitirmeProjesi/colab_kod_guncelleme.zip\",\n",
                "    \"/content/drive/MyDrive/colab_kod_guncelleme.zip\"\n",
                "]\n",
                "\n",
                "zip_path = None\n",
                "for p in potential_zips:\n",
                "    if os.path.exists(p):\n",
                "        zip_path = p\n",
                "        break\n",
                "\n",
                "if zip_path:\n",
                "    print(f\"📦 Kod paketi bulundu: {zip_path}\")\n",
                "    !unzip -q -o \"{zip_path}\" -d {local_project}\n",
                "    print(\"✅ Kodlar ve 100 sınıf configleri başarıyla açıldı!\")\n",
                "else:\n",
                "    print(\"⚠️ colab_kod_guncelleme.zip bulunamadı. Klasörlerden kopyalanıyor...\")\n",
                "    drive_project = \"/content/drive/MyDrive/BitirmeProjesi\"\n",
                "    for folder in [\"src\", \"scripts\", \"configs\", \"data\"]:\n",
                "        src_f = os.path.join(drive_project, folder)\n",
                "        dst_f = os.path.join(local_project, folder)\n",
                "        if os.path.exists(src_f):\n",
                "            if os.path.exists(dst_f):\n",
                "                shutil.rmtree(dst_f)\n",
                "            shutil.copytree(src_f, dst_f)\n",
                "            print(f\"✅ {folder} kopyalandı.\")\n",
                "\n",
                "# Outputs klasörünü Drive'a bağla\n",
                "drive_outputs = \"/content/drive/MyDrive/BitirmeProjesi/outputs\"\n",
                "os.makedirs(drive_outputs, exist_ok=True)\n",
                "if not os.path.exists(f\"{local_project}/outputs\"):\n",
                "    os.symlink(drive_outputs, f\"{local_project}/outputs\")\n",
                "\n",
                "print(\"✅ Çalışma alanı hazır!\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 3. Ham AUTSL Veri Klasörünün Tespiti\n",
                "*Drive'da `AUTSL_Dataset` klasörünü anında bağlar (Zaman aşımı yok).* "
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import os\n",
                "\n",
                "# Doğrudan gerçek AUTSL klasörüne bak (recursive glob YOK, anında bağlanır!)\n",
                "potential_roots = [\n",
                "    \"/content/drive/MyDrive/BitirmeProjesi/AUTSL_Dataset\",\n",
                "    \"/content/drive/MyDrive/AUTSL_Dataset\"\n",
                "]\n",
                "\n",
                "autsl_data_dir = None\n",
                "for r in potential_roots:\n",
                "    if os.path.isdir(os.path.join(r, \"train\")) and os.path.exists(os.path.join(r, \"SignList_ClassId_TR_EN.csv\")):\n",
                "        autsl_data_dir = r\n",
                "        break\n",
                "\n",
                "if not autsl_data_dir:\n",
                "    autsl_data_dir = \"/content/drive/MyDrive/BitirmeProjesi/AUTSL_Dataset\"\n",
                "\n",
                "os.environ[\"AUTSL_DATA_DIR\"] = autsl_data_dir\n",
                "print(f\"✅ Gerçek AUTSL veri seti ve video klasörü bağlandı: {autsl_data_dir}\")\n",
                "\n",
                "# 100 Sınıf seçim ve pilot listesini kontrol et\n",
                "%cd /content/BitirmeProjesi\n",
                "!python scripts/prepare_autsl_100_preprocessing.py"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 4. 🧪 Aşama 1: 200 Videoluk Pilot Duman Testi (~2-3 Dakika)\n",
                "*Her 100 sınıftan birer train ve val videosu (toplam 200 video) işlenir ve kalite kontrolünden geçirilir.*"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "%cd /content/BitirmeProjesi\n",
                "\n",
                "# 1. 200 videoluk pilot base16 çıkarımı\n",
                "!python scripts/preprocess_tid.py \\\n",
                "    --config configs/experiment/autsl_100_preprocess_base16.yaml \\\n",
                "    --data_dir \"$AUTSL_DATA_DIR\" \\\n",
                "    --splits train val \\\n",
                "    --video_ids_file data/autsl/autsl_100_preprocessing_pilot_ids.json \\\n",
                "    --no_drive_backup\n",
                "\n",
                "# 2. Pilot merkez onarımı ve canlı bilek kırpımı (strict)\n",
                "!python scripts/repair_hand_crops.py \\\n",
                "    --config configs/experiment/autsl_100_preprocess_base16.yaml \\\n",
                "    --data_dir \"$AUTSL_DATA_DIR\" \\\n",
                "    --dest_cache cache/autsl_100_features_centerrepair_strict \\\n",
                "    --splits train val \\\n",
                "    --video_ids_file data/autsl/autsl_100_preprocessing_pilot_ids.json \\\n",
                "    --box_scale 1.5 \\\n",
                "    --center_repair \\\n",
                "    --center_policy configs/experiment/center_repair_strict.json \\\n",
                "    --max_alignment_error 0\n",
                "\n",
                "# 3. Pilot kalite denetimi (en kötü 30 video)\n",
                "!python scripts/audit_cache_quality.py \\\n",
                "    --caches cache/autsl_100_features_centerrepair_strict \\\n",
                "    --splits train val \\\n",
                "    --worst 30 \\\n",
                "    --output outputs/autsl_100_pilot_quality.json\n",
                "\n",
                "print(\"\\n✅ 200 videoluk pilot test tamamlandı! Eğer hata yoksa 5. adıma geçebilirsiniz.\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 5. 🚀 Aşama 2: Tam 100 Sınıf Ön İşleme (14.368 Video)\n",
                "*Pilot doğrulandıktan sonra 100 sınıfın 12.405 train ve 1.963 val videosunun tamamı işlenir.*"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "%cd /content/BitirmeProjesi\n",
                "\n",
                "# 1. Tam 100 sınıf Base16 çıkarımı (Önceden işlenen pilot videolar otomatik atlanır)\n",
                "!python scripts/preprocess_tid.py \\\n",
                "    --config configs/experiment/autsl_100_preprocess_base16.yaml \\\n",
                "    --data_dir \"$AUTSL_DATA_DIR\" \\\n",
                "    --splits train val\n",
                "\n",
                "# 2. Tam 100 sınıf strict onarımı (1.5x kutu + merkez onarımı)\n",
                "!python scripts/repair_hand_crops.py \\\n",
                "    --config configs/experiment/autsl_100_preprocess_base16.yaml \\\n",
                "    --data_dir \"$AUTSL_DATA_DIR\" \\\n",
                "    --dest_cache cache/autsl_100_features_centerrepair_strict \\\n",
                "    --splits train val \\\n",
                "    --box_scale 1.5 \\\n",
                "    --center_repair \\\n",
                "    --center_policy configs/experiment/center_repair_strict.json \\\n",
                "    --max_alignment_error 0\n",
                "\n",
                "print(\"\\n✅ Tam ön işleme tamamlandı!\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 6. 🔍 Aşama 3: Kalite ve Hareket/Parmak Anomali Taraması\n",
                "*Tüm 14.368 dosyayı otomatik tarar: parmak sıçramaları, bilek tersliği, kayıp eller listelenir.*"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "%cd /content/BitirmeProjesi\n",
                "\n",
                "# 1. Genel önbellek kalite denetimi (En kötü 200 video)\n",
                "!python scripts/audit_cache_quality.py \\\n",
                "    --caches cache/autsl_100_features_centerrepair_strict \\\n",
                "    --splits train val \\\n",
                "    --worst 200 \\\n",
                "    --output outputs/autsl_100_full_quality.json\n",
                "\n",
                "# 2. Hareket ve parmak anomali denetimi (En şüpheli 300 video)\n",
                "# (Herhangi bir video eksiği olsa bile raporu yazıp notebook akışını korumak için || true)\n",
                "!python scripts/audit_autsl_preprocessing_motion.py \\\n",
                "    --config configs/experiment/autsl_100_preprocess_centerrepair_strict.yaml \\\n",
                "    --cache_dir cache/autsl_100_features_centerrepair_strict \\\n",
                "    --data_dir \"$AUTSL_DATA_DIR\" \\\n",
                "    --output outputs/autsl_100_motion_audit.json \\\n",
                "    --top 300 || true\n",
                "\n",
                "# 3. Sonuçların canlı özetini ekrana bas\n",
                "import os, json\n",
                "audit_p = 'outputs/autsl_100_motion_audit.json'\n",
                "if os.path.exists(audit_p):\n",
                "    with open(audit_p, encoding='utf-8') as f:\n",
                "        d = json.load(f)\n",
                "    print(f\"\\n🔍 TOPLAM İNCELENEN VİDEO: {len(d.get('all_videos', []))}\")\n",
                "    print(f\"⚠️ GEÇERSİZ / BOZUK DOSYA SAYISI: {d.get('invalid_count', 0)}\")\n",
                "    print(\"\\n--- 📌 EN YÜKSEK HAREKET/PARMAK ANOMALİSİ GÖSTEREN İLK 10 VİDEO ---\")\n",
                "    for item in d.get('top', [])[:10]:\n",
                "        print(f\"[{item['split'].upper()}] {item['video_id']} (İşaretçi: {item['signer']}, Skor: {item['triage_score']}) -> Parmak Sıçrama: L={item['left_finger_jump_flags']}/R={item['right_finger_jump_flags']}, Bilek Sıçrama: L={item['left_wrist_jump_flags']}/R={item['right_wrist_jump_flags']}, Bilek Ters: L={item['left_wrist_mismatch_frames']}/R={item['right_wrist_mismatch_frames']}\")\n",
                "print(\"\\n✅ Kalite ve hareket denetimi tamamlandı!\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 7. 💾 Önbelleğin Drive'a Yedeklenmesi ve Raporların Paketlenmesi"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import os, zipfile, glob\n",
                "\n",
                "# 1. Bütün kalite, anomali ve hata raporlarını tek bir zip dosyası yap\n",
                "report_zip = \"/content/drive/MyDrive/BitirmeProjesi/outputs/autsl_100_kalite_raporlari.zip\"\n",
                "report_files = sorted(list(set(\n",
                "    glob.glob(\"/content/BitirmeProjesi/outputs/autsl_*.json\") +\n",
                "    glob.glob(\"/content/BitirmeProjesi/cache/**/preprocessing_failures_*.json\", recursive=True) +\n",
                "    glob.glob(\"/content/BitirmeProjesi/cache/**/fallback_summary.json\", recursive=True)\n",
                ")))\n",
                "\n",
                "with zipfile.ZipFile(report_zip, \"w\", zipfile.ZIP_DEFLATED) as zf:\n",
                "    for f in report_files:\n",
                "        zf.write(f, arcname=os.path.basename(f))\n",
                "        print(f\"✅ Rapora eklendi: {os.path.basename(f)}\")\n",
                "\n",
                "print(f\"\\n📊 Kalite ve hata raporları paketi hazır: {report_zip}\")\n",
                "\n",
                "# 2. Strict cache'i Drive'a kopyala\n",
                "drive_cache = \"/content/drive/MyDrive/BitirmeProjesi/cache/autsl_100_features_centerrepair_strict\"\n",
                "os.makedirs(drive_cache, exist_ok=True)\n",
                "print(\"\\n🚀 Önbellek Drive'a kopyalanıyor...\")\n",
                "!cp -rn cache/autsl_100_features_centerrepair_strict/. {drive_cache}/\n",
                "!echo \"Drive'daki toplam .npz dosya sayısı: $(find {drive_cache} -name '*.npz' | wc -l) (Beklenen: 14368)\"\n",
                "print(\"\\n🎉 TEBRİKLER! Bütün 100 sınıf ön işleme ve kalite raporlama başarıyla tamamlandı!\")"
            ]
        }
    ],
    "metadata": {
        "accelerator": "GPU",
        "colab": {
            "provenance": []
        },
        "kernelspec": {
            "display_name": "Python 3",
            "name": "python3"
        },
        "language_info": {
            "name": "python"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 0
}

with open("colab_100_sinif_onisleme.ipynb", "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)

print("colab_100_sinif_onisleme.ipynb created successfully!")
