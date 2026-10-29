# Colab: ilk kontrollü ön işleme karşılaştırması

**Yeni model teşhisi:** İşaretçi 11 incelemesi sonrasında hazır pose-only
deneyinin komutları [dayanıklılık raporunda](signer11_dayaniklilik_analizi_2026-10-20.md).
Bu rehberin aşağıdaki seed 2026 komutları isteğe bağlı sonraki dayanıklılık
kontrolüdür; mevcut model geliştirme için zorunlu değildir.

**20 Ekim güncel durum:** Seed 42 ve seed 7 eğitim/değerlendirmeleri alındı.
ZIP'in tam incelemesi [seed 7 sonuç raporunda](colab_seed7_sonuclari_2026-10-20.md).
Seed 7 strict, 10 sınıf model geliştirme baz çizgisi olarak seçildi. Önce
yedi val hatası incelenecek; **seed 2026 çifti zorunlu ön koşul değil**, sonra
dayanıklılık kontrolü için hazır tutuluyor. Komutları bu rehberin sonundadır.
Aşağıdaki ilk seed 42/7 komutları tarihsel kayıttır.

Bu rehber yerel tam cache denetimi tamamlandıktan sonra uygulanır. Eğitim bu
oturumda başlatılmadı. İlk karşılaştırma 10 sınıf, resmî 1.245 train ve 199 val,
16 kare, aynı dondurulmuş MobileNetV3, seed 42 ve 20 epoch kullanır.

## Doğru dosyalar

- Temel: `configs/experiment/autsl_10_base16_screen.yaml` ve `cache/autsl_10_features`.
- Aday: `configs/experiment/autsl_10_centerrepair_screen.yaml` ve **`cache/autsl_10_features_centerrepair_strict`**.
- Aday manifesti: sürüm 3, ölçek 1,50 ve `min_disagreement_ratio=1.0`.
- `cache/autsl_10_features_centerrepair` ilk keşif koşusudur; iki görsel gerileme nedeniyle reddedildi. Eğitimde kullanılmaz.
- `*_pilot` klasörleri 25 video içerir; eğitimde kullanılmaz. `autsl_10_wristcrop_screen.yaml` eski varyanttır; bu karşılaştırmanın ikinci kolu değildir.

İki kolda model, optimizasyon, augmentation ve epoch bütçesi eşit. Hız artırımı
ve erken durdurma bu kısa taramada kapalı. Deney bütün eksik-el kırpım politikasını
karşılaştırır; ölçek, canlı kırpım ve merkez onarımının tek tek katkısını ayırmaz.

## Drive'a aktarılacaklar

Kullanıcı tarafından mevcut Colab proje düzenine aktarılmalı:

1. Güncel `src/`, `scripts/`, `configs/` ve projenin bağımlılık dosyaları.
2. İki cache klasörünün **manifest.json + train/ + val/** içeriği. Test cache'i bu deney için gerekmez. Birlikte yaklaşık 3,18 GiB.
3. `outputs/colab_training_metadata/` (dört resmî CSV ve kaynak checksum bilgisi).
4. `outputs/colab_cache_checksums.json`.
5. Yerel denetim kanıtı olarak `outputs/colab_preflight.json` ve `docs/merkez_onarimi_tam_onisleme_raporu.md`.

CSV paketi yalnızca cache yükleyicisinin resmî örnek/sınıf eşlemesi içindir.
`test_labels.csv` kök klasör kontrolü için bulunur; eğitim ve aşağıdaki
değerlendirme komutları test bölümünü kullanmaz. Cache'lerden eğitim için ham
RGB videolarını Colab'a yeniden taşımak gerekmez; metadata ile yükleme yerelde
doğrulandı. Ön işleme ham videoları gerektirir ve burada tamamlandı.

Bu çalışma Drive'a yükleme veya Colab oturumu açmadı.

## Colab hazırlık ve aktarım doğrulaması

Aşağıdaki örnek `/content/BitirmeProjesi` proje yolunu ve Drive'da
`MyDrive/BitirmeProjesi` düzenini varsayar. Gerçek yollar farklıysa uyarlanmalı.
GPU türü, RAM, boş disk ve Drive alanı ilk koşunun kaydına yazılmalı.

```bash
cd /content/BitirmeProjesi
mkdir -p cache
cp -r /content/drive/MyDrive/BitirmeProjesi/cache/autsl_10_features cache/
cp -r /content/drive/MyDrive/BitirmeProjesi/cache/autsl_10_features_centerrepair_strict cache/
export AUTSL_DATA_DIR=/content/drive/MyDrive/BitirmeProjesi/outputs/colab_training_metadata
python scripts/check_colab_inputs.py --metadata_dir "$AUTSL_DATA_DIR" --checksum_manifest /content/drive/MyDrive/BitirmeProjesi/outputs/colab_cache_checksums.json --output outputs/colab_preflight_uploaded.json
```

Bu kontrol model kurmaz veya eğitim başlatmaz. Her iki cache'in kapsamını,
manifest/yöntem/ölçek/eşik uyumunu, örnek girdi boyutlarını, ortak deney
ayarlarını ve aktarılan dosyaların SHA256 değerlerini doğrular. Eksik ya da
yanlış dosya varsa önce aktarım düzeltilir. Klasörleri tekrar kopyalarken yanlışlıkla
`cache/isim/isim/` şeklinde iç içe yerleştirmemeye dikkat et.

## İki eğitim ve yalnızca val değerlendirmesi

```bash
python scripts/train.py --config configs/experiment/autsl_10_base16_screen.yaml --cache_dir /content/BitirmeProjesi/cache/autsl_10_features
python scripts/train.py --config configs/experiment/autsl_10_centerrepair_screen.yaml --cache_dir /content/BitirmeProjesi/cache/autsl_10_features_centerrepair_strict
python scripts/evaluate.py --config configs/experiment/autsl_10_base16_screen.yaml --cache_dir /content/BitirmeProjesi/cache/autsl_10_features --checkpoint /content/drive/MyDrive/BitirmeProjesi/outputs/autsl_10_base16_screen/checkpoints/best_model.pt --split val --output outputs/base16_val_v2.json
python scripts/evaluate.py --config configs/experiment/autsl_10_centerrepair_screen.yaml --cache_dir /content/BitirmeProjesi/cache/autsl_10_features_centerrepair_strict --checkpoint /content/drive/MyDrive/BitirmeProjesi/outputs/autsl_10_centerrepair_screen/checkpoints/best_model.pt --split val --output outputs/centerrepair_val_v2.json
```

`train.py`, Drive bağlı Colab ortamında deney çıktılarını config adına göre
Drive'a yönlendirir. Başlamadan önce logdaki cache/çıktı yollarını kaydet.
Bu komutlar yeni deneyler içindir; farklı bir deneyin checkpoint'iyle resume
yapılmaz. Oturum kesilirse aynı deneyin `latest_model.pt` dosyasından sürdürülür.

## Birlikte değerlendirilecek sonuçlar

İki deneyin config/manifesti, `checkpoints/history.csv`, `training_curves.png`,
en iyi epoch/val Top-1, val macro-F1, GPU/süre ve preflight JSON'u gönderilecek.
Val 199 video: bir doğru tahmin yaklaşık 0,503 yüzde puan. İlk kısa eğitim
tarama içindir; küçük farkı kesin kazanç saymadan seed 7 ve 2026 ile tekrar
planlanacak. Ön işleme kararı sonuçlara göre verilecek; test nihai seçim sonrasında.

## 20 Ekim 2026: ilk Colab sonucu ve ölçüm düzeltmesi

Kullanıcının paylaştığı iki eğitim ve val değerlendirme logunda resmî 1.245/199
örneğin tamamı yüklenmiş, seed 42, Tesla T4 ve 20 epoch kullanılmış. Aynı
model/optimizasyon/augmentation ayarları korunmuş. Eğitim içindeki en iyi val
doğruluğu temel kolda 174/199 = %87,44, adayda 177/199 = %88,94; ikisi de
20. epoch'ta. Ayrı `best_model.pt` değerlendirmesi temel kolda 173/199 =
%86,93, adayda 178/199 = %89,45 verdi. Ayrı değerlendirmeye göre net fark
**5/199 = 2,51 yüzde puan**. İki ölçüm arasındaki birer örneklik ters yöndeki
fark henüz tekrar çalıştırılarak açıklanmadı.

Not defterindeki tablo `macro_f1` anahtarını okuyordu; JSON `f1_macro` yazdığı
için yanlışlıkla iki 0,0000 gösterdi. Ayrıca önceki `metrics.py`, gerçek macro
F1 yerine macro precision ile macro recall'un harmonik ortalamasını hesaplıyordu.
Eski logdaki 0,8714 ve 0,8954 değerleri bu eski formüle aittir; gerçek macro F1
olarak kullanılmaz. Kod artık sınıf başına F1 ortalamasını hesaplıyor. Bağımsız
değerlendirme de eğitimdeki val gibi CUDA AMP kullanıyor. Bu iki düzeltmeden sonra
**mevcut checkpoint'ler yeniden değerlendirilir; yeniden eğitim gerekmez.**
Notebook ve yukarıdaki komutlar geçmiş sonuçları korumak için `_v2.json` yazar.
Colab'a güncel `scripts/evaluate.py`, `src/evaluation/evaluator.py` ve
`src/evaluation/metrics.py` aktarılmalı. `colab_egitim.ipynb` tablo anahtarını
`f1_macro` olarak düzeltti ve eksik anahtarı 0 diye gizlemiyor.

Eğitim logundaki `Test evaluation skipped: Call setup() before requesting
DataLoaders` uyarısı, eğitim betiğinin sonda test yükleyicisini çağırmasından
gelmiş; resmî test çalışmamış. Güncel `train.py` bu otomatik çağrıyı kaldırdı.
Test yalnızca sonuç seçimi tamamlanınca açık `--split test` komutuyla çalışacak.

Bu tarama eksik el için tüm kırpım politikasını karşılaştırır. Val'de sıfır
merkez değişikliği vardı; 5 merkez düzeltmesi train'deydi. Dolayısıyla bu
doğruluk farkı yalnız merkez onarımına mal edilemez. 199 videoluk tek seed
sonucu umut verici, seçim için yeterli değil. Önce yeni kodla iki val JSON'u
ve mümkünse `history.csv`/grafikler alınır; sonra aynı 20 epoch bütçesiyle
iki kol seed 7 ve 2026'da tekrarlanır. Bu tekrarlar da aday lehineyse tüm
ön işleme politikası sonraki deneylerin adayı olur. `gap2_v2`, 36 kare ve yeni
omurga daha sonra ayrı karşılaştırılır; test kilitli kalır.

## 20 Ekim devamı: altı Drive çıktısının ayrıntılı incelemesi

`C:\Users\Bünyamin\Desktop\Güncel Drive Çıktıları` içindeki altı dosya okundu.
Dosyalar yerelde `outputs/colab_seed42_review` altında açıklayıcı adlarla
korundu. `history.csv` ve `training_curves (1).png` temel koşuya; `history
(1).csv` ve `training_curves (2).png` strict adaya aittir. Eşleme 20. epoch
val doğruluk/kayıpları ve grafikteki son noktalarla kontrol edildi. Klasörde
checkpoint, video bazlı tahmin, cache manifesti veya Colab preflight dosyası
yok; ağırlıkların içeriği ve tekil hata değişimleri burada incelenemedi.

Gönderilen yeni `base16_val.json` artık **174/199 = %87,44 Top-1**,
**0,8718487 gerçek macro-F1** ve **198/199 Top-5** içeriyor. Yeni
`centerrepair_val.json` **177/199 = %88,94 Top-1**, **0,8885203 gerçek macro-F1**
ve **199/199 Top-5** içeriyor. JSON'lardaki F1 değerleri sınıf başına F1
ortalamasıyla uyumlu; ayrı değerlendirme Top-1 değerleri eğitim içi AMP val
sonuçlarıyla eşleşti. Bu, önceki birer video farkının AMP farkıyla uyumlu
olduğunu gösterir; checkpoint dosyaları olmadığından byte düzeyinde kimlik
doğrulanamadı. Düzeltilmiş Top-1 farkı **3/199 = 1,51 yüzde puan**, macro-F1
farkı **0,01667**. Önceki %86,93/%89,45 rakamları eski AMP'siz ölçümdür;
güncel karar tablosuna alınmaz.

Her iki geçmiş 20 epoch içeriyor. Temelin son train/val kaybı 0,5927/0,6990,
adayın 0,5968/0,6155. Adayın val kaybı 0,0835 daha düşük. Her iki koşuda
en düşük val kaybı ve en yüksek val Top-1 **son epoch'ta**; grafiklerde henüz
kalıcı bir val kaybı yükselişi yok. Train verisi artırılmış, val artırılmamış
olduğundan train ve val doğruluklarını doğrudan genelleme aralığı saymıyoruz.
Son epoch'ta temelin train/val doğruluğu %88,07/%87,44, adayın
%88,15/%88,94. Val eğrileri birkaç epoch dalgalanıyor; tek seed ve yalnız
199 video ile kalıcı üstünlük kararı verilmez.

Bu model 16 karede 225 boyutlu poz+iki el noktası, iki 160×160 el kırpımı,
paylaşılan/dondurulmuş MobileNetV3-Small görsel omurgası, eğitilen projeksiyon,
cross-attention füzyon, TCN+Transformer zaman modeli ve 10 sınıflı başlık
kullanıyor. Tam beden RGB görüntüsü doğrudan girişte değil. Önceki logdaki
`get_num_parameters.total=5.969.610`, paylaşılan el omurgasını iki kez
sayıyordu; gerçek benzersiz parametre sayısı **4.894.890**, eğitilen **3.959.434**.
Raporlama fonksiyonu düzeltildi; model ağırlıkları veya önceki tahminler
değişmedi. Tüm ön işleme değişikliği train'de yalnız beş merkez onarımı,
val'de sıfır merkez onarımı içerdiğinden gözlenen kazanç tek başına merkez
onarımının nedensel etkisi değildir.

## Önceki adım: eşleştirilmiş seed 7 pilotu

Seed 7 ve 2026 için dört ayrı config üretildi; her çift yerelde resmî
1.245/199 cache ve eşit model/augmentation/eğitim bütçesiyle preflight geçti.
İlk planda seed 7'den sonra seed 2026 çifti vardı; 20 Ekim karar
düzeltmesiyle seed 2026 zorunlu kapı olmaktan çıkarıldı.
Yeni isimler Drive'daki seed 42 checkpoint'lerini korur. Eski 20 epoch
checkpoint'ini değiştirilmiş epoch sayısıyla resume etmek yerine her seed
sıfırdan başlar.

Colab'a güncel `scripts/evaluate.py`, `src/evaluation/evaluator.py`,
`src/evaluation/metrics.py`, `src/models/hybrid_model.py` ve dört yeni seed
config'i aktar. Önce iki seed 42 checkpoint'inde video bazlı tahminleri al;
bu ek değerlendirme yaklaşık mevcut iki val değerlendirmesi kadar sürer:

```bash
cd /content/BitirmeProjesi
export AUTSL_DATA_DIR=/content/drive/MyDrive/BitirmeProjesi/outputs/colab_training_metadata
python scripts/evaluate.py --config configs/experiment/autsl_10_base16_screen.yaml --cache_dir cache/autsl_10_features --checkpoint /content/drive/MyDrive/BitirmeProjesi/outputs/autsl_10_base16_screen/checkpoints/best_model.pt --split val --output outputs/base16_seed42_val_verified.json --predictions_output outputs/base16_seed42_predictions.json
python scripts/evaluate.py --config configs/experiment/autsl_10_centerrepair_screen.yaml --cache_dir cache/autsl_10_features_centerrepair_strict --checkpoint /content/drive/MyDrive/BitirmeProjesi/outputs/autsl_10_centerrepair_screen/checkpoints/best_model.pt --split val --output outputs/centerrepair_seed42_val_verified.json --predictions_output outputs/centerrepair_seed42_predictions.json
```

Ardından seed 7 çifti için:

```bash
python scripts/check_colab_inputs.py --base_config configs/experiment/autsl_10_base16_seed7.yaml --candidate_config configs/experiment/autsl_10_centerrepair_seed7.yaml --metadata_dir "$AUTSL_DATA_DIR" --output outputs/seed7_preflight_colab.json
python scripts/train.py --config configs/experiment/autsl_10_base16_seed7.yaml --cache_dir cache/autsl_10_features
python scripts/train.py --config configs/experiment/autsl_10_centerrepair_seed7.yaml --cache_dir cache/autsl_10_features_centerrepair_strict
python scripts/evaluate.py --config configs/experiment/autsl_10_base16_seed7.yaml --cache_dir cache/autsl_10_features --checkpoint /content/drive/MyDrive/BitirmeProjesi/outputs/autsl_10_base16_seed7/checkpoints/best_model.pt --split val --output outputs/base16_seed7_val.json --predictions_output outputs/base16_seed7_predictions.json
python scripts/evaluate.py --config configs/experiment/autsl_10_centerrepair_seed7.yaml --cache_dir cache/autsl_10_features_centerrepair_strict --checkpoint /content/drive/MyDrive/BitirmeProjesi/outputs/autsl_10_centerrepair_seed7/checkpoints/best_model.pt --split val --output outputs/centerrepair_seed7_val.json --predictions_output outputs/centerrepair_seed7_predictions.json
```

İki seed 42 tahmin JSON'u, iki seed 7 metrik+tahmin JSON'u ve iki seed 7
`history.csv` dosyası gönderilsin. Böylece ortak doğru/yanlış örnekleri,
işaretçi/sınıf kümelerini ve ikinci seed'in yönünü inceleyebiliriz. Seed 2026
kararı bu incelemenin ardından; resmî test açılmaz.

## 20 Ekim ZIP incelemesi sonrası: seed 2026 eşleştirilmiş çift

Seed 42 ve 7'nin `best_model.pt` değerlendirmelerinde strict sırasıyla
177/199 ve 192/199, base16 ise 174/199 ve 188/199 yaptı. Seed duyarlılığı
büyük ve işaretçi 11 adaya karşı iki kez geriledi. Bu üçüncü çift,
geliştirme başladıktan sonra yapılabilecek **isteğe bağlı dayanıklılık
kontrolüdür**; aynı 16 kare cache'leri kullanır. Yeni 36 kare, `gap2`,
yeni omurga veya 100 sınıf bu koşuya eklenmez. Güncel `scripts/evaluate.py`,
`src/evaluation/metrics.py` ve seed 2026 config'leri Colab proje klasöründe
olmalıdır. Örnek Drive düzeni önceki bölümlerle aynıdır.

```bash
cd /content/BitirmeProjesi
export AUTSL_DATA_DIR=/content/drive/MyDrive/BitirmeProjesi/outputs/colab_training_metadata
python scripts/check_colab_inputs.py --base_config configs/experiment/autsl_10_base16_seed2026.yaml --candidate_config configs/experiment/autsl_10_centerrepair_seed2026.yaml --metadata_dir "$AUTSL_DATA_DIR" --checksum_manifest /content/drive/MyDrive/BitirmeProjesi/outputs/colab_cache_checksums.json --output outputs/seed2026_preflight_colab.json
python scripts/train.py --config configs/experiment/autsl_10_base16_seed2026.yaml --cache_dir cache/autsl_10_features
python scripts/train.py --config configs/experiment/autsl_10_centerrepair_seed2026.yaml --cache_dir cache/autsl_10_features_centerrepair_strict
python scripts/evaluate.py --config configs/experiment/autsl_10_base16_seed2026.yaml --cache_dir cache/autsl_10_features --checkpoint /content/drive/MyDrive/BitirmeProjesi/outputs/autsl_10_base16_seed2026/checkpoints/best_model.pt --split val --output outputs/base16_seed2026_val.json --predictions_output outputs/base16_seed2026_predictions.json
python scripts/evaluate.py --config configs/experiment/autsl_10_centerrepair_seed2026.yaml --cache_dir cache/autsl_10_features_centerrepair_strict --checkpoint /content/drive/MyDrive/BitirmeProjesi/outputs/autsl_10_centerrepair_seed2026/checkpoints/best_model.pt --split val --output outputs/centerrepair_seed2026_val.json --predictions_output outputs/centerrepair_seed2026_predictions.json
```

Eğitimden önce preflight çıktısının başarılı olduğunu kontrol et; önceki
checkpoint'leri üzerine yazma ve `--resume` kullanma. İki val JSON'u, iki
tahmin JSON'u, iki `history.csv`, iki eğitim eğrisi, preflight JSON'u ve GPU/
süre kaydı birlikte paylaşılmalı. Tahmin JSON'larında checkpoint SHA256
bulunur. İstenirse iki dosya `scripts/analyze_val_pairs.py` ile yalnız resmi
val etiketlerine karşı eşleştirilir:

```bash
python scripts/analyze_val_pairs.py --base outputs/base16_seed2026_predictions.json --candidate outputs/centerrepair_seed2026_predictions.json --official_labels "$AUTSL_DATA_DIR/val_labels.csv" --output outputs/seed2026_pair_analysis.json
```

`test` bu turda çalıştırılmaz. Güncel karar kapıları
[yol haritasında](AUTSL_226_yol_haritasi.md).
