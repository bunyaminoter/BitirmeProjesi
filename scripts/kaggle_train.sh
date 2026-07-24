#!/bin/bash
# ==============================================================================
# Kaggle Eğitim Yardımcı Betiği (Kaggle Training Helper Script)
# ==============================================================================

# Python cache (.pyc) dosyalarının oluşturulmasını kapat (Read-Only hatalarını önler)
export PYTHONDONTWRITEBYTECODE=1

echo "=========================================================="
echo "Kaggle ASL Citizen Eğitim Süreci Başlatılıyor..."
echo "=========================================================="

echo "[1/5] Proje Dosyaları Çalışma Dizinine Kopyalanıyor..."
# Read-Only sorunlarını çözmek için tüm kodları yazılabilir /kaggle/working klasörüne alıyoruz
cp -r /kaggle/input/datasets/bnyamin01/bitirmeprojesi/BitirmeProjesi/* /kaggle/working/
cd /kaggle/working

echo "[2/5] Bağımlılıklar yükleniyor (MediaPipe vb.)..."
pip install -q mediapipe pyyaml numpy opencv-python

echo "[3/5] Dizinler kontrol ediliyor..."
mkdir -p /kaggle/working/cache/asl_citizen_features
mkdir -p /kaggle/working/outputs

echo "[4/5] MediaPipe Landmark çıkarımı başlatılıyor (Bu işlem zaman alabilir)..."
python scripts/preprocess_asl_citizen.py \
    --config configs/experiment/asl_citizen_baseline.yaml \
    --cache_dir /kaggle/working/cache/asl_citizen_features

echo "[5/5] Model Eğitimi Başlatılıyor!"
python scripts/train.py \
    --config configs/experiment/asl_citizen_baseline.yaml \
    --cache_dir /kaggle/working/cache/asl_citizen_features

echo "=========================================================="
echo "Eğitim tamamlandı! Checkpoint dosyaları /kaggle/working/outputs klasöründedir."
echo "=========================================================="
