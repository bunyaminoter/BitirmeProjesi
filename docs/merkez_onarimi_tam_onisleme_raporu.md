# Merkez onarımı ve tam 10 sınıf ön işleme sonucu

Tarih: 19 Ekim 2026. Son karar: **yerel ön işleme tamamlandı; kontrollü kısa Colab karşılaştırmasına hazır.** Bu, tanıma doğruluğunun arttığına dair bir eğitim sonucu değildir.

## Kullanılacak son sürüm

- Cache: **`cache/autsl_10_features_centerrepair_strict`**.
- Kapsam: resmî 10 sınıfın **1.245 train + 199 val = 1.444 videosu**. 100/226 sınıf çıkarımı yapılmadı.
- Girdi: videoya yayılan 16 örnek, 160×160 el görüntüleri; aynı 33+21+21 nokta ve özgün sınıf kimlikleri.
- Manifest: `pipeline_version=2`, `derivation.method=pose_wrist_fallback_crop`, `version=3`, `box_scale=1.5`.
- Politika: `configs/experiment/center_repair_strict.json`.
- Tam config: `configs/experiment/autsl_10_centerrepair.yaml`.
- İlk kısa eğitim config'i: **`configs/experiment/autsl_10_centerrepair_screen.yaml`**.
- Boyut: yaklaşık **1,647 GiB**. Mevcut temel train+val cache'i yaklaşık 1,533 GiB; ikisi birlikte yaklaşık 3,18 GiB.

Model eğitimi Colab'da yapılacak. Bu oturumda yeni AUTSL modeli eğitilmedi, Drive'a yükleme/Colab çalıştırması yapılmadı.

## Onarımın kapsamı

Sadece elin kayıp olduğu **en fazla iki örneklik iç boşluklar** değerlendirilir. İki uçta da aynı elin tespiti ve poz bulunmalı; noktalar sonlu/görüntü içinde, kutu en az 8 piksel olmalı. El bileği kendi poz bileğine diğer bilekten daha yakın ve normalize koordinatta 0,12 mesafeden küçük olmalı.

Uç kutu merkezleri arasındaki hareket bir temel medyan el kutusu kenarını aşmamalı. Bu uçlardan doğrusal kestirilen merkez ile mevcut poz merkezinin farkı **en az bir tam temel kutu kenarı** ise merkez değiştirilir. Poz yoksa aynı güvenilir uç kuralı altında merkez kestirimi kullanılabilir. Başlangıç, son ve uzun boşluklara yayılmaz. Kararlı poz merkezleri korunur.

Tam eşikler:

```json
{
  "method": "short_gap_bbox_linear_if_disagree",
  "max_gap": 2,
  "max_wrist_distance": 0.12,
  "max_anchor_step_ratio": 1.0,
  "min_disagreement_ratio": 1.0
}
```

Bu, **kırpım merkezi** kestirimidir; parmak veya eklem noktası doldurmaz. Modelin landmark girdileri ve MediaPipe tespit maskeleri temel cache'le bire bir aynı kaldı. Bütün klip eğitimde ve video tahmininde mevcut olduğundan iki uç kullanılabilir; bu yöntem geleceği görmeyen canlı kare takibi olarak sunulmamalı.

Yeni kaynak kaydı: `*_crop_source`: 0 tespitli, 1 pozdan canlı kırpım, 2 değişmeden kalan görüntü, **3 komşu tespitlerden onarılan merkez**. Sürüm 3 ayrıca `*_crop_center` (kaynak görüntü pikseli), `*_crop_box` (gerçek tam sayı x/y/kenar) ve `*_crop_center_imputed` maskesini saklar. Bu metadata yalnızca kaynak 1/3 için merkez/kutu içerir; diğer satırlarda sıfırdır. Kaynaklar gerçek el tespitinden ayrı tutulur.

## İki deneme ve düzeltme kararı

1. İlk politika uyuşmazlık eşiğini 0,75 kutu kenarı kullanıyordu. Sabit 25 videoda iki bilinen hatayı düzeltti; diğer 158 kayıp örneğin görüntüsü aynı kaldı. Pilot tam koşuya kabul edildi.
2. İlk tam 1.444 video üretildi ve ham kare denetimi geçti. **7** merkez değişikliğinin hepsi görsel incelendi. `signer38_sample58_color` sol örnek 11 ve `signer40_sample243_color` sol örnek 12'de hızlı/kavisli hareket nedeniyle yeni kırpım kötüleşti. Doğrusal hareket varsayımı iki karede uygun değildi. Bu tam aday eğitim için **reddedildi**.
3. Müdahale eşiği 1,0 kutuya yükseltildi. Diğer eşikler değişmedi. Aynı 25 videoda ayrı sıkı pilot üretildi: bütün NPZ alanları kabul edilen ilk pilotla bire bir aynı; iki olumlu düzeltme korundu.
4. Sıkı politika ayrı tam cache'e çıkarıldı. **5** merkez değişikliğinin tümü görsel incelendi: eller yeniden kadraja giriyor. İki gerileme örneğinde merkez artık değişmiyor. Bu yeni aday kısa model karşılaştırması için kabul edildi.

İlk aday `cache/autsl_10_features_centerrepair` ve QA klasörü `outputs/centerrepair_full_qa` korunuyor; eğitimde kullanılmaz. İlk politika `configs/experiment/center_repair_exploratory.json` içinde. Son adayın adı `_strict` ile biter. Aynı türetme sürümündeki eşik farkı manifest/config karşılaştırmasında denetlenir.

Bu eşik train kalite gözlemleriyle seçildi. İki iyi kare veya bir yüksek kapsam oranı bütün veri için doğruluk garantisi sayılmadı; bu nedenle ilk tam koşuda bulunan gerilemeler kayda geçirilip ikinci tam koşu yapıldı.

## Tam ham kare denetimi

`scripts/audit_center_repair.py`, son cache'in **bütün 1.444 videosunu** ham RGB kaynağından tekrar okudu:

- **23.104 video örneği**, iki el için 46.208 el örneği.
- **36.587 tespitli kırpım** yeniden kesildi; temel cache ve ham kareyle piksel farkı sıfır.
- **9.619 canlı yedek kırpım** gerçek kutu sınırlarından yeniden kesildi; piksel farkı sıfır. Train 8.639, val 980.
- **5 merkez onarımı**: train'de 4 sağ + 1 sol; val'de 0. Bunlar 1,50 ölçekli poz varyantına göre değişen beş görüntüdür; onarımın etkisi geniş kapsamlıymış gibi sunulmamalı.
- Etiket, kare sayısı, poz/el koordinatları ve tespit maskeleri temel cache'le bire bir aynı. Tespitli görüntüler değişmedi.
- **2 çözülemeyen örnek**: train `signer33_sample505_color` sağ örnek 2 eski kırpımı taşıyor; `signer33_sample571_color` sol örnek 0 ilk tespitten önce varsayılan görüntüde. Bu nedenle mutlak sıfır donma iddiası yok.
- Kaynak 1/3 görüntülerinde tümü siyah veya önceki kırpımla tamamen aynı görüntü: 0. Bu yalnızca görüntü/provenans kontrolüdür; doğru elin kadrajda olduğunu tek başına kanıtlamaz.
- Gözlenen el kareleri üzerinde varsayımsal 1,50 poz kutusu kapsamı: **35.162/36.561 = %96,17**. Bu bir vekil ölçümdür; kayıp karelerin gerçek parmak doğruluğu veya model val doğruluğu değildir.
- Bu ortalama bütün işaretçileri garanti etmez: en düşük vekil kapsam `signer9` için %75,22; ardından `signer32` %91,68 ve `signer3` %91,93. Bu gruplar olası hata yoğunlaşması olarak korunur; kısa eğitim sonrasında kişi/sınıf hata analizinde izlenecek. Kapsamı düşük karelerin tamamının elle doğrulandığı iddia edilmiyor.
- **89** yedekte geometrik dış görüntü dolgusu %25'i aşıyor; tamamı train'de. En büyük oran %61,81. En yüksek 12 örnek ayrıca görsel incelendi: dinlenen el görüntünün alt sınırında/dışında; kaynakta görünmeyen el büyütmeyle geri getirilemez. `padding_top12.png` ve seçim JSON'u kaydedildi. 89 örneğin tamamının elle etiketlendiği iddia edilmiyor.
- **2.795** kırpımda diğer tespitli elin merkezi de kutuya giriyor. Temas eden/çaprazlanan iki el işaretlerinde bu otomatik kimlik hatası değildir. Kayıt inceleme yardımıdır.

Son beş merkez değişikliği:

- `train/signer33_sample186_color`, sağ, örnek 2 ve 3.
- `train/signer33_sample505_color`, sağ, örnek 5 ve 6.
- `train/signer41_sample269_color`, sol, örnek 10.

Örnek indeksleri sıfır tabanlı. Görsel kontrol bütün bu değişiklikleri ve çözülemeyen iki örneği kapsar. Değişmeyen tüm 9.614 yedek görüntü için insan anotasyonlu el doğruluğu çıkarılmadı. Bulanıklık, örtülme, uzun kayıplar ve görüntü dışında kalan eller devam eden sınırlamalar.

Kanıt:

- `outputs/centerrepair_strict_full_qa/audit.json`: tüm ham kare/piksel/geometri doğrulaması, sınıf ve işaretçi grupları.
- Aynı klasörde `visual_review.json`: `accepted_for_short_colab_comparison`.
- `centers_001.png`: tüm beş merkez değişikliği ve iki çözülemeyen örnek.
- `padding_top12.png` ve `padding_top12_cases.json`: dolgu odaklı inceleme.
- `outputs/centerrepair_strict_full_quality.json`: tespit/eksik seri/taşınan görüntü sayıları.

## Kod ve doğrulama

`wrist_fallback_crop.py` kaynak/geometri/kısa merkez onarımını ortak hesaplar. `repair_hand_crops.py` manifesti, politika JSON'unu ve kaynak/çıktı ayrımını denetleyerek ayrı cache yazar. Tam koşuda resmî örnek listesinin tamamını ister; kaynak cache eksikse sessizce alt kümeyle bitirmez.

Config, cache yükleyicisi ve `inference_pipeline.py` aynı merkez politikası/sürüm/ölçeği kullanır. Yanlış eşik veya manifestle yükleme durur. Ham `signer33_sample505_color` videosu ve cache'lenmiş detector çıktılarıyla, ortak tahmin hazırlayıcısının girdileri cache yükleyicisinin normalize girdileriyle **bire bir aynı** bulundu. Bu kontrol detector tekrar üretilebilirliğini veya sınıflandırma başarısını ölçmez. Kanıt: `outputs/centerrepair_strict_tensor_parity.json`.

**63/63 test geçti.** Kısa/uzun boşluk, başlangıç/son, yanlış el/uç, görüntü dışı nokta, büyük uç hareketi, küçük uyuşmazlıkta müdahale etmeme, girdi eşitliği ve manifest uyumu test edildi. Python derleme ve diff boşluk kontrolü başarılı.

## Tekrar üretim

Proje kökünde, ham AUTSL kökü yerel varsayılan yolda veya `AUTSL_DATA_DIR` ile erişilebilir olmalı:

```powershell
python scripts/repair_hand_crops.py --dest_cache cache/autsl_10_features_centerrepair_strict_pilot --box_scale 1.5 --center_repair --center_policy configs/experiment/center_repair_strict.json --video_ids_file configs/experiment/autsl_10_wristcrop_pilot25.json --max_alignment_error 0
python scripts/audit_center_repair.py --candidate cache/autsl_10_features_centerrepair_strict_pilot --selection configs/experiment/autsl_10_wristcrop_pilot25.json --output_dir outputs/centerrepair_strict_pilot_qa
python scripts/repair_hand_crops.py --dest_cache cache/autsl_10_features_centerrepair_strict --box_scale 1.5 --center_repair --center_policy configs/experiment/center_repair_strict.json --max_alignment_error 0
python scripts/audit_center_repair.py --candidate cache/autsl_10_features_centerrepair_strict --output_dir outputs/centerrepair_strict_full_qa
python scripts/audit_cache_quality.py --caches cache/autsl_10_features_centerrepair_strict --output outputs/centerrepair_strict_full_quality.json
python -m pytest -q
```

Var olan dosyalar atlanır; üreticinin `fallback_summary.json` dosyası son çağrının yeni işlenen/atlanan sayılarını içerir. Tam kapsam için auditor sonucu esas alınır. Yeniden sıfırdan üretimde farklı yeni hedef kullanılır. Mevcut kanıtlar silinmez.

Günlükler: `outputs/logs/centerrepair_*`. Keşif ve sıkı pilot/tam koşular ayrı dosyalarda. Test sonucu `centerrepair_strict_tests.log`; aktarım hazırlığı `centerrepair_colab_preflight.log`.

## Colab'a hazır olanlar ve sonraki karar

`outputs/colab_training_metadata/` içine dört resmî CSV, kaynakla aynı SHA256 değerleriyle kopyalandı. Bu işlem test videolarını veya test doğruluğunu kullanmadı; yükleyicinin beklediği metadata dosyaları hazırlandı. Ham RGB videolar olmadan metadata + cache ile train/val yükleme yerelde doğrulandı.

`outputs/colab_cache_checksums.json` iki kolun train/val dosyaları ve manifestleri için **2.890 SHA256 kaydı** içerir. `scripts/check_colab_inputs.py` bunların tamamını doğruladı; iki kolda 1.245/199 örnek, eşit model/augmentation/optimizasyon bütçesi, seed 42 ve 20 epoch doğrulandı. Sonuç: `outputs/colab_preflight.json`. Bu kontrol model kurmaz veya eğitim başlatmaz.

Sonraki adım: **Colab'da temel 16 kare ile `centerrepair_strict` ön işleme paketinin eşit bütçeli kısa karşılaştırması.** Deney bütün kırpım politikasını karşılaştırır; bu tek koşu canlı yedek, ölçek ve merkez onarımının ayrı katkısını ayırmaz. Merkez onarımı val'de hiç tetiklenmedi; etkisini olduğundan büyük yorumlamamak gerekir.

Aktarım listesi, kontrol ve eğitim/val komutları: `docs/colab_onisleme_karsilastirmasi.md`. Sonuçlar gelince val eğrileri/Top-1/macro-F1 ve süreye göre birlikte karar verilecek. Test kilitli kalır. 100/226 sınıfa ve yeni omurgaya bu kısa karşılaştırmadan sonra geçilir.
