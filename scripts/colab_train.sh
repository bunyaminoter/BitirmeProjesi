#!/bin/bash
# ==============================================================================
# Colab TİD (Türk İşaret Dili / AUTSL) Eğitim Yardımcı Betiği
#
# Google Drive üzerinden 10 sınıflık TİD modellerini çalıştırır.
# Kullanım: 
#   !bash scripts/colab_train.sh configs/experiment/autsl_10_mobilenet.yaml
#   !bash scripts/colab_train.sh --cv  (5-Fold Cross-Validation)
# ==============================================================================

set -e  # Hata olursa dur

MODE=${1:-"single"}
CONFIG=${2:-"configs/experiment/autsl_10_mobilenet.yaml"}

if [ "$1" != "--cv" ] && [ -n "$1" ]; then
    CONFIG=$1
fi

EXP_NAME=$(basename "${CONFIG}" .yaml)

# ---- Ortam Değişkenleri ----
DRIVE_PROJECT="/content/drive/MyDrive/BitirmeProjesi"
DRIVE_CACHE="${DRIVE_PROJECT}/cache/tid_10_features"
LOCAL_CACHE="/content/BitirmeProjesi/cache/tid_10_features"
DRIVE_OUTPUTS="${DRIVE_PROJECT}/outputs"
WORK_DIR="/content/BitirmeProjesi"

echo "=========================================================="
echo "Colab TİD (Türk İşaret Dili / AUTSL) Eğitim Süreci"
echo "Model Konfigürasyonu: ${CONFIG}"
echo "=========================================================="

# 1. Yerel hızlı disk önbelleği hazırla
echo "[1/3] Cache dosyaları yerel diske eşitleniyor..."
mkdir -p ${LOCAL_CACHE}
if [ -d "${DRIVE_CACHE}" ] && [ "$(ls -A ${DRIVE_CACHE})" ]; then
    echo "1.245 Cache dosyası Drive'dan kopyalanıyor..."
    cp -rn ${DRIVE_CACHE}/* ${LOCAL_CACHE}/ 2>/dev/null || true
fi

# 2. Bağımlılıkları kontrol et
echo "[2/3] Bağımlılıklar doğrulanıyor..."
pip install -q pyyaml matplotlib rich opencv-python scikit-learn

# 3. Eğitimi başlat
cd ${WORK_DIR}
if [ "$1" == "--cv" ]; then
    echo "[3/3] 5-Fold Cross-Validation ve Ensemble Başlatılıyor..."
    python scripts/train_cv.py \
        --config "${CONFIG}" \
        --n_splits 5 \
        --cache_dir "${LOCAL_CACHE}"
else
    echo "[3/3] Standart Model Eğitimi Başlatılıyor..."
    python scripts/train.py \
        --config "${CONFIG}" \
        --cache_dir "${LOCAL_CACHE}"
fi

echo "=========================================================="
echo "✅ İşlem başarıyla tamamlandı!"
echo "Çıktılar Drive'a kaydedildi: ${DRIVE_OUTPUTS}"
echo "=========================================================="
