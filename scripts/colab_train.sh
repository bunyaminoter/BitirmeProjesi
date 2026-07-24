#!/bin/bash
# ==============================================================================
# Colab Eğitim Yardımcı Betiği (Google Colab Training Helper Script)
#
# Google Drive üzerinden projeyi çalıştırır.
# Kullanım: !bash scripts/colab_train.sh
# ==============================================================================

set -e  # Hata olursa dur

# ---- Ortam Değişkenleri (Colab'da çalıştırıldığında Drive yolları) ----
DRIVE_PROJECT="/content/drive/MyDrive/BitirmeProjesi"
DRIVE_CACHE="${DRIVE_PROJECT}/cache/asl_citizen_features"
DRIVE_OUTPUTS="${DRIVE_PROJECT}/outputs"
WORK_DIR="/content/BitirmeProjesi"

# ASL Citizen dataset splits (Colab'da Drive'a yüklenen CSV dosyaları)
ASL_SPLITS="${DRIVE_PROJECT}/asl_citizen_data/splits"

echo "=========================================================="
echo "Colab ASL Citizen Eğitim Süreci Başlatılıyor..."
echo "=========================================================="

# 1. Proje kodlarını Drive'dan hızlı çalışma alanına kopyala
echo "[1/4] Proje kodları kopyalanıyor..."
rm -rf ${WORK_DIR}
mkdir -p ${WORK_DIR}
cp -r ${DRIVE_PROJECT}/src ${WORK_DIR}/
cp -r ${DRIVE_PROJECT}/scripts ${WORK_DIR}/
cp -r ${DRIVE_PROJECT}/configs ${WORK_DIR}/
cp ${DRIVE_PROJECT}/requirements.txt ${WORK_DIR}/ 2>/dev/null || true
cp ${DRIVE_PROJECT}/pyproject.toml ${WORK_DIR}/ 2>/dev/null || true

# 2. Bağımlılıkları yükle
echo "[2/4] Bağımlılıklar yükleniyor..."
pip install -q mediapipe pyyaml numpy opencv-python

# 3. Dizinleri hazırla
echo "[3/4] Çıktı dizinleri hazırlanıyor..."
mkdir -p ${DRIVE_OUTPUTS}/asl_citizen_baseline/checkpoints
mkdir -p ${DRIVE_PROJECT}/cache

# 4. Eğitimi başlat
echo "[4/4] Model Eğitimi Başlatılıyor!"
cd ${WORK_DIR}
python scripts/train.py \
    --config configs/experiment/asl_citizen_baseline.yaml \
    --cache_dir "${DRIVE_CACHE}"

echo "=========================================================="
echo "Eğitim tamamlandı!"
echo "Checkpoint dosyaları: ${DRIVE_OUTPUTS}"
echo "=========================================================="
