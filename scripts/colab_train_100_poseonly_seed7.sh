#!/usr/bin/env bash
# Run after mounting Drive in Colab.
set -euo pipefail

DRIVE_PROJECT=/content/drive/MyDrive/BitirmeProjesi
LOCAL_PROJECT=/content/BitirmeProjesi
CACHE_NAME=autsl_100_features_centerrepair_strict
CONFIG_NAME=autsl_100_poseonly_seed7.yaml
RUN_NAME=autsl_100_poseonly_seed7

test -d "$DRIVE_PROJECT/cache/$CACHE_NAME"
test -f "$DRIVE_PROJECT/configs/experiment/$CONFIG_NAME"
mkdir -p "$LOCAL_PROJECT/configs/experiment" "$LOCAL_PROJECT/data/autsl"

# Kodları ve metadata'yı kopyala
for folder in src scripts configs data; do
  if [ -d "$DRIVE_PROJECT/$folder" ]; then
    rm -rf "$LOCAL_PROJECT/$folder"
    cp -a "$DRIVE_PROJECT/$folder" "$LOCAL_PROJECT/"
  fi
done

cd "$LOCAL_PROJECT"
export AUTSL_DATA_DIR="$LOCAL_PROJECT/data/autsl"

# Önbelleği Colab yerel SSD'sine kopyala (eğitim hızını 100 kat artırır)
LOCAL_CACHE="$LOCAL_PROJECT/cache/$CACHE_NAME"
if [ -d "$LOCAL_CACHE/train" ] && [ "$(ls "$LOCAL_CACHE/train" | wc -l)" -eq 12405 ]; then
  echo "⚡ Önbellek zaten yerel SSD'de tam mevcut ($LOCAL_CACHE)."
else
  echo "🚀 Önbellek Drive'dan yerel SSD'ye aktarılıyor..."
  mkdir -p "$LOCAL_PROJECT/cache"
  cp -rn "$DRIVE_PROJECT/cache/$CACHE_NAME" "$LOCAL_PROJECT/cache/"
  echo "✅ Önbellek yerel SSD'ye kopyalandı!"
fi

# Cache bütünlüğünü, signer ayrılığını ve 1-batch ileri/geri geçişini kontrol et
python - <<'PY'
import json, os, torch
from pathlib import Path
from src.core.config import load_config
from src.data.tid_datamodule import TIDDataModule, collate_fn
from src.models.hybrid_model import HybridASLModel

root = Path('/content/BitirmeProjesi/cache/autsl_100_features_centerrepair_strict')
manifest = json.loads((root/'manifest.json').read_text(encoding='utf-8'))
assert manifest['num_classes'] == 100
assert manifest['derivation']['method'] == 'pose_wrist_fallback_crop'
assert manifest['derivation']['version'] == 3
train_n = len([f for f in os.listdir(root/'train') if f.endswith('.npz')])
val_n = len([f for f in os.listdir(root/'val') if f.endswith('.npz')])
print(f'100 Strict Cache Verified on Local SSD: {train_n} train + {val_n} val')
assert train_n == 12405 and val_n == 1963

cfg = load_config('configs/experiment/autsl_100_poseonly_seed7.yaml')
dm = TIDDataModule(
    dataset_config=cfg.dataset,
    training_config=cfg.training,
    cache_dir=str(root),
    augmentation_config=cfg.augmentation,
    hand_image_size=tuple(cfg.model.hand_encoder.input_size),
    landmark_dim=cfg.model.landmark_encoder.input_dim,
    load_rgb=cfg.model.use_rgb,
)
dm.setup(stage='fit')
train_signers = {s.signer_id for s in dm._train_dataset.base_dataset.samples}
val_signers = {s.signer_id for s in dm._val_dataset.base_dataset.samples}
assert train_signers.isdisjoint(val_signers)
batch = collate_fn([dm._train_dataset[i] for i in range(4)])
assert 'left_hand_images' not in batch
model = HybridASLModel(cfg.model)
logits = model(batch)
assert logits.shape == (4, 100)
print('Pre-flight signer & 1-batch forward check passed on local SSD!')
PY

# 100 Sınıf Pose-Only Eğitimi (35 Epoch, kesinti varsa otomatik --resume)
LATEST_CKPT="$DRIVE_PROJECT/outputs/$RUN_NAME/checkpoints/latest_model.pt"
if [ -f "$LATEST_CKPT" ]; then
  echo "🔄 Yarım kalan checkpoint bulundu ($LATEST_CKPT), kaldığı yerden devam ediliyor..."
  python -u scripts/train.py \
    --config "configs/experiment/$CONFIG_NAME" \
    --cache_dir "$LOCAL_CACHE" \
    --resume "$LATEST_CKPT"
else
  python -u scripts/train.py \
    --config "configs/experiment/$CONFIG_NAME" \
    --cache_dir "$LOCAL_CACHE"
fi

# 1.963 Resmî Val Videosunda Değerlendirme
python scripts/evaluate.py \
  --config "configs/experiment/$CONFIG_NAME" \
  --checkpoint "$DRIVE_PROJECT/outputs/$RUN_NAME/checkpoints/best_model.pt" \
  --cache_dir "$LOCAL_CACHE" \
  --split val \
  --output "$DRIVE_PROJECT/outputs/$RUN_NAME/val_metrics.json" \
  --predictions_output "$DRIVE_PROJECT/outputs/$RUN_NAME/val_predictions.json"

echo "✅ 100 Sınıf Pose-Only Eğitimi ve Değerlendirmesi Başarıyla Tamamlandı!"
