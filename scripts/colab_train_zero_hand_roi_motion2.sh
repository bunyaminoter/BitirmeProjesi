#!/usr/bin/env bash
# Run after mounting Drive and preparing /content/BitirmeProjesi in Colab.
set -euo pipefail

DRIVE_PROJECT=/content/drive/MyDrive/BitirmeProjesi
LOCAL_PROJECT=/content/BitirmeProjesi
CACHE_NAME=autsl_10_features_zero_hand_roi_motion2
CONFIG_NAME=autsl_10_zero_hand_roi_motion2_seed7.yaml
RUN_NAME=autsl_10_zero_hand_roi_motion2_seed7

test -d "$DRIVE_PROJECT/cache/$CACHE_NAME"
test -f "$DRIVE_PROJECT/configs/experiment/$CONFIG_NAME"
test -f "$LOCAL_PROJECT/scripts/train.py"
mkdir -p "$LOCAL_PROJECT/cache" "$LOCAL_PROJECT/configs/experiment"
if [ ! -f "$LOCAL_PROJECT/cache/$CACHE_NAME/manifest.json" ]; then
  cp -a "$DRIVE_PROJECT/cache/$CACHE_NAME" "$LOCAL_PROJECT/cache/"
fi
cp "$DRIVE_PROJECT/configs/experiment/$CONFIG_NAME" "$LOCAL_PROJECT/configs/experiment/$CONFIG_NAME"

cd "$LOCAL_PROJECT"
export AUTSL_DATA_DIR="$DRIVE_PROJECT/outputs/colab_training_metadata"
python - <<'PY'
import json
from pathlib import Path
root = Path('cache/autsl_10_features_zero_hand_roi_motion2')
manifest = json.loads((root/'manifest.json').read_text())
assert manifest['derivation']['method'] == 'zero_hand_upperbody_roi_motion2'
assert manifest['derivation']['minimum_adjacent_sampled_frames'] == 2
assert len(list((root/'train').glob('*.npz'))) == 1245
assert len(list((root/'val').glob('*.npz'))) == 199
print('Motion2 cache ready: 1245 train + 199 val')
PY

python scripts/train.py \
  --config "configs/experiment/$CONFIG_NAME" \
  --cache_dir "$LOCAL_PROJECT/cache/$CACHE_NAME"

python scripts/evaluate.py \
  --config "configs/experiment/$CONFIG_NAME" \
  --checkpoint "$DRIVE_PROJECT/outputs/$RUN_NAME/checkpoints/best_model.pt" \
  --cache_dir "$LOCAL_PROJECT/cache/$CACHE_NAME" \
  --split val \
  --output "$DRIVE_PROJECT/outputs/$RUN_NAME/val_metrics.json" \
  --predictions_output "$DRIVE_PROJECT/outputs/$RUN_NAME/val_predictions.json"
