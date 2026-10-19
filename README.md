# AUTSL İşaret Tanıma Projesi

Bu proje AUTSL'nin resmî, işaretçiden bağımsız `train` / `val` / `test` bölümlerinde RGB el kırpıntıları ve MediaPipe iskelet özellikleriyle izole işaret tanıma eğitir.

## Veri protokolü

- `train`: ağırlıkların öğrenildiği bölüm.
- `val`: erken durdurma, hiperparametre ve model seçimi bölümü.
- `test`: model ve ayarlar sabitlendikten sonra yalnızca son değerlendirme bölümü.

Eski özel 10 sınıf CSV paketi ve `val+test` birleşimi kullanılmaz. `AUTSL_DATA_DIR`, `train_labels.csv`, `val_labels.csv`, `test_labels.csv`, `SignList_ClassId_TR_EN.csv` ve ilgili `train/`, `val/`, `test/` klasörlerini içeren resmî AUTSL kökünü göstermelidir.

```powershell
$env:AUTSL_DATA_DIR = 'C:\Users\Bünyamin\Desktop\AUTSL_Dataset'
```

## İlk 10 sınıf deneyi

`configs/experiment/autsl_10_mobilenet.yaml`, önceki çalışmadaki on AUTSL sınıfını (özgün sınıf kimlikleri `8, 14, 15, 20, 64, 106, 114, 143, 144, 196`) kullanır. Model etiketleri eğitim için 0–9'a eşlenir; video bölümleri resmî AUTSL ayrımından gelir.

Beklenen örnek sayıları:

- train: 1.245
- val: 199
- test: 167

Önce veri kapsamını denetle, sonra cache oluştur ve eğit:

```powershell
python scripts/audit_autsl.py --data_dir "$env:AUTSL_DATA_DIR" --config configs/experiment/autsl_10_mobilenet.yaml --output outputs/autsl_10_audit.json
python scripts/preprocess_tid.py --config configs/experiment/autsl_10_mobilenet.yaml --data_dir "$env:AUTSL_DATA_DIR" --splits train val test --pilot_size 20
python scripts/audit_autsl.py --data_dir "$env:AUTSL_DATA_DIR" --config configs/experiment/autsl_10_mobilenet.yaml --cache_dir cache/autsl_10_features --inspect_features --output outputs/autsl_10_pilot_audit.json
python scripts/inspect_preprocessing.py --split val --max_videos 10
python scripts/inspect_preprocessing.py --split train --max_videos 10
python scripts/preprocess_tid.py --config configs/experiment/autsl_10_mobilenet.yaml --data_dir "$env:AUTSL_DATA_DIR" --splits train val test
python scripts/audit_autsl.py --data_dir "$env:AUTSL_DATA_DIR" --config configs/experiment/autsl_10_mobilenet.yaml --cache_dir cache/autsl_10_features --require_cache --output outputs/autsl_10_full_audit.json
python scripts/train.py --config configs/experiment/autsl_10_mobilenet.yaml --cache_dir cache/autsl_10_features
python scripts/evaluate.py --config configs/experiment/autsl_10_mobilenet.yaml --checkpoint outputs/autsl_10_mobilenet/checkpoints/best_model.pt --split val --cache_dir cache/autsl_10_features
```

### Ön işleme kalite pilotu

Aynı 10 train ve 10 val videosunda 16 kare, kısa el noktası onarımı ve 36 kareyi karşılaştırmak için:

```powershell
python scripts/repair_hand_gaps.py --video_ids_file configs/experiment/autsl_10_preprocessing_pilot.json
python scripts/preprocess_tid.py --config configs/experiment/autsl_10_preprocess_36.yaml --splits train val --video_ids_file configs/experiment/autsl_10_preprocessing_pilot.json --no_drive_backup
python scripts/compare_preprocessing_pilots.py
python scripts/inspect_preprocessing.py --config configs/experiment/autsl_10_preprocess_36.yaml --split val --frame_counts 36 --output_dir outputs/preprocessing_36_qa
```

Pilot sonucu `outputs/preprocessing_pilot_comparison.json` içinde, görsel panolar `outputs/preprocessing_qa/`, `outputs/preprocessing_gap2_qa/` ve `outputs/preprocessing_36_qa/` içindedir. Pilot cache'i yalnızca 20 videoyu kapsar; model doğruluğu karşılaştırması için kullanılmaz. Tam deneylerde 16 kare temel config'i, `autsl_10_gap2.yaml` ve `autsl_10_36.yaml` ayrı cache ve çıktı dizinleri kullanır.

Test komutu yalnızca val sonucu ile karar verildikten sonra çalıştırılır:

```bash
python scripts/evaluate.py --config configs/experiment/autsl_10_mobilenet.yaml --checkpoint outputs/autsl_10_mobilenet/checkpoints/best_model.pt --split test --cache_dir cache/autsl_10_features
```

## 226 sınıfa geçiş

`configs/experiment/autsl_226_mobilenet_frozen.yaml` tüm 226 AUTSL sınıfını ve aynı resmî bölümleri kullanır. 10 sınıflı deneyin sonucu ve hata analiziyle bir sonraki deneyi birlikte seçtikten sonra bu yapılandırmaya geçilir.

Deney sırası ve karar ölçütleri için [AUTSL yol haritasına](docs/AUTSL_226_yol_haritasi.md) ve araştırma özeti için [rapora](chatgpt%20raporu.md) bak.
