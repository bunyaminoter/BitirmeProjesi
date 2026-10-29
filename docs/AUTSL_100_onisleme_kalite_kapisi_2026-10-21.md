# AUTSL 100 sınıf: eğitim öncesi ön işleme kalite kapısı

> **22 Ekim son durum:** Önceki Drive eksikliği tamamlandı; base16 ve strict için 12.405 train + 1.963 val dosya adı doğrulandı. Seçili 28 videoda strict piksel/geometri, 6.732 yerel yeni NPZ'de CRC kontrolü geçti. Aktif el kaybı için yedi train videosunda üst gövde ROI tanı pilotu yapıldı; bütün cache'e değişiklik uygulanmadı. [Güncel analiz](AUTSL_100_onisleme_sonuc_analizi_2026-10-22.md). Aşağıdaki 21 Ekim planı tarihsel üretim tarifidir.

> **22 Ekim durum güncellemesi:** Colab kalite raporları üretildi; Drive Desktop'ta strict cache henüz kısmi görünüyor. Ham videolu bağımsız bulgular ve düzeltme/aktarım kararı için [sonuç analizi](AUTSL_100_onisleme_sonuc_analizi_2026-10-22.md) geçerlidir. Bu sayfanın altındaki komutlar tarihsel üretim planıdır; kalite kapısı henüz tamamen kapanmadı.

**Durum (21 Ekim 2026):** 10 sınıflı `centerrepair_strict` akışı ön işleme v1 olarak kabul edildi. Bu belge yalnız 100 sınıfın train/val ön işlemesini ve kalite denetimini planlar. **100 sınıf modeli eğitilmeyecek; resmî test açılmayacak.**

## Sabit küme ve kapsam

`scripts/prepare_autsl_100_preprocessing.py`, önceki 10 özgün AUTSL ClassId'sini koruyup kalan 216 train sınıfından `Random(20261021)` ile 90 sınıf seçti. Liste val başarısı ya da tahminleriyle seçilmedi. Özgün ID → yeni 0–99 model etiketi, Türkçe/İngilizce adlar, kaynak dosya SHA-256'ları ve sınıf başına sayılar `data/autsl/autsl_100_preprocessing_selection.json` içinde. **12.405 train (31 işaretçi) + 1.963 val (6 işaretçi) = 14.368 video** bekleniyor. Bu küme “zor minimal çiftleri” bilerek seçilmiş bir küme değildir; geniş, önceden sabitlenmiş kalite taramasıdır. Zor hareketler tarama sonucunda bulunacak.

Üretilen config'ler `configs/experiment/autsl_100_preprocess_base16.yaml` ve `configs/experiment/autsl_100_preprocess_centerrepair_strict.yaml`. V1: 16 örnek kare, MediaPipe poz/iki el noktası, 160×160 iki el crop'u, 1,5 ölçekli canlı bilek yedek crop'u ve yalnız kısa boşluklarda **crop merkezinin** ihtiyatlı onarımı. Parmak landmark'ı onarımı v1'de yoktur. Strict config doğrudan `preprocess_tid.py` girdisi değildir; iki aşama zorunlu.

## Colab sırası

Komutlar proje dizini `/content/BitirmeProjesi` ve resmî AUTSL kökü `/content/drive/MyDrive/BitirmeProjesi/AUTSL_Dataset` varsayar. Gerçek konum farklıysa `AUTSL_DATA_DIR` değerini değiştir. Colab'a bu commit/dosyalar, resmî etiketler ve `SignList_ClassId_TR_EN.csv` taşınmalı. `requirements.txt` sürümleri ve kullanılan `.task` MediaPipe modellerinin SHA-256'ları kaydedilmeli; ön işleme hızı için Colab **GPU tokunun** faydası varsayılmamalı, çünkü mevcut çıkarıcı `RunningMode.IMAGE` ile `detect()` çağırıyor.

```bash
cd /content/BitirmeProjesi
export AUTSL_DATA_DIR=/content/drive/MyDrive/BitirmeProjesi/AUTSL_Dataset
python scripts/prepare_autsl_100_preprocessing.py

# Önce her sınıftan train/val birer video: toplam 200-video duman testi.
python scripts/preprocess_tid.py --config configs/experiment/autsl_100_preprocess_base16.yaml --data_dir "$AUTSL_DATA_DIR" --splits train val --video_ids_file data/autsl/autsl_100_preprocessing_pilot_ids.json --no_drive_backup
python scripts/repair_hand_crops.py --config configs/experiment/autsl_100_preprocess_base16.yaml --data_dir "$AUTSL_DATA_DIR" --dest_cache cache/autsl_100_features_centerrepair_strict --splits train val --video_ids_file data/autsl/autsl_100_preprocessing_pilot_ids.json --box_scale 1.5 --center_repair --center_policy configs/experiment/center_repair_strict.json --max_alignment_error 0
python scripts/audit_cache_quality.py --caches cache/autsl_100_features_centerrepair_strict --splits train val --worst 30 --output outputs/autsl_100_pilot_quality.json
# Pilot aşamasında motion denetiminin kapsam hatası vermesi beklenir;
# çünkü henüz bütün 100 sınıf videoları işlenmedi. Tam koşudan sonra çalıştır.
```

Pilotun 100+100 dosyası, sıfır kare hizalama hatası ve görsel panoları kontrol edilmeden tam koşuya geçilmez. Pilot sonrası aynı klasördeki geçerli dosyalar tekrar işlenmeden kalan videolar tamamlanabilir:

```bash
python scripts/preprocess_tid.py --config configs/experiment/autsl_100_preprocess_base16.yaml --data_dir "$AUTSL_DATA_DIR" --splits train val
python scripts/repair_hand_crops.py --config configs/experiment/autsl_100_preprocess_base16.yaml --data_dir "$AUTSL_DATA_DIR" --dest_cache cache/autsl_100_features_centerrepair_strict --splits train val --box_scale 1.5 --center_repair --center_policy configs/experiment/center_repair_strict.json --max_alignment_error 0
python scripts/audit_cache_quality.py --caches cache/autsl_100_features_centerrepair_strict --splits train val --worst 200 --output outputs/autsl_100_full_quality.json
python scripts/audit_autsl_preprocessing_motion.py --config configs/experiment/autsl_100_preprocess_centerrepair_strict.yaml --cache_dir cache/autsl_100_features_centerrepair_strict --data_dir "$AUTSL_DATA_DIR" --output outputs/autsl_100_motion_audit.json --top 300

# Audit sıfır hata ile bittikten sonra strict cache'i kalıcı depoya kopyala.
mkdir -p /content/drive/MyDrive/BitirmeProjesi/cache/autsl_100_features_centerrepair_strict
cp -a cache/autsl_100_features_centerrepair_strict/. /content/drive/MyDrive/BitirmeProjesi/cache/autsl_100_features_centerrepair_strict/
find /content/drive/MyDrive/BitirmeProjesi/cache/autsl_100_features_centerrepair_strict -name '*.npz' | wc -l
# Beklenen: 14368
```

Tam koşu logunda `failed=0`, strict cache'te **12.405 train / 1.963 val** NPZ, eşleşen manifest, split/etiket tutarlılığı ve sıfır beklenmeyen dosya doğrulanacak. Pilot cache'i tam klasöre karışmadığı için ayrı bir sürüm değildir; aynı config/manifestle kesintiden devam eder. Başka MediaPipe sürümü/modeliyle devam edilirse karışık cache oluşabilir: ortam/model hash'lerini önce karşılaştır, farklıysa yeni cache aç. `repair_hand_crops.py` strict cache'i Drive'a otomatik yedeklemez; Colab oturumu bitmeden strict klasörün Drive kopyası ve dosya sayısı doğrulanmalı. `fallback_summary.json` yalnız **son çağrıda yeni işlenen** dosyaları sayar; pilotta işlenip tam çağrıda atlanan dosyaları içermez. Toplam kaliteyi tam cache denetiminden oku. 10 sınıf strict cache ortalama boyutuyla kaba ölçekleme, 100 sınıf strict cache için yaklaşık **16 GiB** verir; gerçek boyut pilotta ölçülmeli. Base ve strict birlikte yaklaşık iki kat alan ister.

## Ne ve nasıl incelenecek?

1. **Bütün NPZ'leri otomatik tara:** Eksik/bozuk dosya, 16 kare, etiket, tespit maskesi, 1–2 / 3–4 / 5–8 / 9+ kare boşluklar, bilek uyuşmazlığı, crop kaynağı, impute sayısı, sınıf ve işaretçi yoğunlaşması. `audit_autsl_preprocessing_motion.py` ayrıca ani bilek ve bileğe göre parmak ucu sıçramalarını **yalnız inceleme adayı** olarak sıralar; hızlı gerçek hareket yanlış pozitif verebilir. Sayılar train/val ayrı raporlanacak.
2. **Görsel kontrol:** Her sınıftan en az bir train örneği, farklı işaretçiler ve en kötü boşluk/bilek/anlık sıçrama örnekleri ham videonun 16 kare üst üste bindirilmiş landmark panoları ve crop panolarıyla incelenecek. `scripts/inspect_preprocessing.py --config configs/experiment/autsl_100_preprocess_centerrepair_strict.yaml --data_dir "$AUTSL_DATA_DIR" --split train --video_id signerX_sampleY_color --frame_counts 16 --cache_dir cache/autsl_100_features_centerrepair_strict --output_dir outputs/autsl_100_panels` komutu tek video için kullanılabilir. Gerekirse aynı ham video daha sık karelerle **yalnız görsel teşhis** için açılacak; 16 kare v1 otomatik değişmeyecek.
3. **Hata taksonomisi:** (a) el gerçekten kadraj dışında/örtülü, (b) hareket bulanıklığı/düşük kontrast, (c) tespit yok ama el görüntüde, (d) tespit var fakat parmak eklemleri yanlış, (e) el/işaretçi eşlemesi ters, (f) crop eli kesiyor, (g) 16 örnek kare hareketin belirleyici anını kaçırıyor, (h) çözücü/kare hizası hatası. Her kararın video ID'si, split'i, sınıfı, işaretçisi, kare indeksi ve görsel kanıtı kaydedilecek; sebep gözlenemiyorsa “belirsiz” yazılacak.
4. **Düzeltme kapısı:** Tekrarlayan, görselle doğrulanmış bir hata türü bulunursa yalnız train'den seçilen küçük, eşleştirilmiş pilotta tek değişken denenir. V1 cache'i ve manifesti korunur. Val üzerinde kişi/video özel ayar yapılmaz; val kalite güvencesi ve daha sonra model seçimi içindir.

`.npz` tespit bayrağı, gerçek parmak doğruluğunun etiketi değildir: yanlış parmak noktası “tespit edildi” görünebilir. Ayrıca 16 NPZ karesi, aradaki hareketin kaçıp kaçmadığını göstermez. **Neden analizi için ham RGB video gerekir.** Yerel `C:\Users\Bünyamin\Desktop\AUTSL_Dataset` kökü mevcut ve proje CSV'lerinin train/sınıf-listesi hash'leriyle eşleşiyor. Bu yüzden kullanıcı strict NPZ'leri getirdiğinde ham videolarla aynı ID üzerinden inceleme yapabiliriz. Tam cache aktarımı güçse ilk teslim: seçim JSON'u, iki manifest, Colab logları, `fallback_summary.json`, kalite JSON'u ve sıralanan şüpheli videoların strict NPZ'leri; tam kapsam kanıtı için sonradan bütün strict cache gerekir.

**Önemli sınır:** 100 sınıf modelini eğitmediğimiz için bu sınıflarda henüz “hatalı tahmin yaptığı videolar” yoktur. Önceki 10 sınıf hataları olası hata türlerini bulmaya yarar. Ön işleme kusuru ile modelin sınıfları karıştırması farklı nedenler olabilir. Kalite kapısı geçildikten sonra eğitim kararı ayrıca verilecek.
