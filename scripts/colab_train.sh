#!/bin/bash
# ==============================================================================
# Colab AUTSL 10 Sınıf: Kontrollü Ön İşleme Karşılaştırma Eğitim Betiği
#
# Kullanım:
#   bash scripts/colab_train.sh --preflight
#   bash scripts/colab_train.sh --train-base
#   bash scripts/colab_train.sh --train-candidate
#   bash scripts/colab_train.sh --evaluate
#   bash scripts/colab_train.sh --all
# ==============================================================================

set -e

DRIVE_PROJECT="/content/drive/MyDrive/BitirmeProjesi"
LOCAL_PROJECT="/content/BitirmeProjesi"
LOCAL_CACHE="${LOCAL_PROJECT}/cache"
METADATA_DIR="${DRIVE_PROJECT}/outputs/colab_training_metadata"
CHECKSUM_FILE="${DRIVE_PROJECT}/outputs/colab_cache_checksums.json"

export AUTSL_DATA_DIR="${METADATA_DIR}"

run_sync() {
    echo "[1/4] Yerel disk ve klasörler hazırlanıyor..."
    mkdir -p "${LOCAL_CACHE}"
    if [ ! -d "${LOCAL_CACHE}/autsl_10_features" ]; then
        echo "Temel cache kopyalanıyor..."
        cp -r "${DRIVE_PROJECT}/cache/autsl_10_features" "${LOCAL_CACHE}/"
    fi
    if [ ! -d "${LOCAL_CACHE}/autsl_10_features_centerrepair_strict" ]; then
        echo "Aday cache kopyalanıyor..."
        cp -r "${DRIVE_PROJECT}/cache/autsl_10_features_centerrepair_strict" "${LOCAL_CACHE}/"
    fi
}

run_preflight() {
    run_sync
    echo "[2/4] Preflight SHA256 ve tensor bütünlük denetimi başlatılıyor..."
    cd "${LOCAL_PROJECT}"
    python scripts/check_colab_inputs.py \
        --metadata_dir "${METADATA_DIR}" \
        --checksum_manifest "${CHECKSUM_FILE}" \
        --base_cache "${LOCAL_CACHE}/autsl_10_features" \
        --candidate_cache "${LOCAL_CACHE}/autsl_10_features_centerrepair_strict" \
        --output outputs/colab_preflight_uploaded.json
    echo "✅ Preflight başarıyla tamamlandı!"
}

run_train_base() {
    echo "🚀 Temel model eğitimi başlıyor (autsl_10_base16_screen)..."
    cd "${LOCAL_PROJECT}"
    python scripts/train.py \
        --config configs/experiment/autsl_10_base16_screen.yaml \
        --cache_dir "${LOCAL_CACHE}/autsl_10_features"
}

run_train_candidate() {
    echo "🚀 Merkez onarımlı aday model eğitimi başlıyor (autsl_10_centerrepair_screen)..."
    cd "${LOCAL_PROJECT}"
    python scripts/train.py \
        --config configs/experiment/autsl_10_centerrepair_screen.yaml \
        --cache_dir "${LOCAL_CACHE}/autsl_10_features_centerrepair_strict"
}

run_eval() {
    echo "📊 Resmi val değerlendirmesi yapılıyor..."
    cd "${LOCAL_PROJECT}"
    python scripts/evaluate.py \
        --config configs/experiment/autsl_10_base16_screen.yaml \
        --cache_dir "${LOCAL_CACHE}/autsl_10_features" \
        --checkpoint "${DRIVE_PROJECT}/outputs/autsl_10_base16_screen/checkpoints/best_model.pt" \
        --split val \
        --output outputs/base16_val.json

    python scripts/evaluate.py \
        --config configs/experiment/autsl_10_centerrepair_screen.yaml \
        --cache_dir "${LOCAL_CACHE}/autsl_10_features_centerrepair_strict" \
        --checkpoint "${DRIVE_PROJECT}/outputs/autsl_10_centerrepair_screen/checkpoints/best_model.pt" \
        --split val \
        --output outputs/centerrepair_val.json
    echo "✅ Değerlendirme tamamlandı. Çıktılar outputs/ altında kaydedildi."
}

ACTION=${1:-"--all"}

case "${ACTION}" in
    --sync)
        run_sync
        ;;
    --preflight)
        run_preflight
        ;;
    --train-base)
        run_sync
        run_train_base
        ;;
    --train-candidate)
        run_sync
        run_train_candidate
        ;;
    --evaluate)
        run_eval
        ;;
    --all)
        run_preflight
        run_train_base
        run_train_candidate
        run_eval
        ;;
    *)
        echo "Bilinmeyen parametre: ${ACTION}"
        echo "Kullanım: bash scripts/colab_train.sh [--sync|--preflight|--train-base|--train-candidate|--evaluate|--all]"
        exit 1
        ;;
esac
