# İşaretçi 11 dayanıklılık incelemesi ve canlı uygulama deney planı

## 20 Ekim pose-only sonucu — önceki deney planının güncellemesi

Planlanan pose-only seed 7 deneyi tamamlandı: resmî 10 sınıf val'de
**198/199 (%99,50)**, işaretçi 11'de **30/30**. Yedi hibrit hatanın tamamı
düzeldi; tek yeni hata işaretçi 16'da. Güncel karar ve veri denetimi
[pose-only raporunda](poseonly_seed7_sonuclari_2026-10-20.md). Aşağıdaki
“Önerilen en kısa deney sırası” bölümünün ilk adımı artık tarihsel kayıttır;
sıradaki deney aynı bütçede seed 42 pose-only tekrarıdır. Sonuç RGB kolunun
bu 10 sınıf koşusunda yararlı olmadığını gösterir; kürk/kontrastın tek neden
olduğunu veya bütün 226 sınıfta RGB'nin yararsız kalacağını kanıtlamaz.

**Tarih:** 20 Ekim 2026  
**Kapsam:** Resmî AUTSL val bölümündeki seçilmiş 10 sınıf, seed 7 `centerrepair_strict` referansı. Resmî test açılmadı. Colab eğitimi bu incelemede yapılmadı.

## Sonuç ve sınır

Seed 7 strict model **192/199 (%96,48)**; işaretçi 11'de **26/30 (%86,67)**, diğer beş işaretçide birlikte **166/169 (%98,22)** doğru. Yedi hatanın dördü işaretçi 11'e ait. Kullanıcının kalın tüylü yelek, koyu kıyafet, mor koltuk, oturma açısı ve gölgelenme gözlemleri ham videoda görülebiliyor. Bu koşulların el izleme ve sınıflandırmayı zorlaştırması makul bir **hipotez**; aynı işaretçinin kimliği, hareket tarzı ve sahnesi birbirinden ayrılmadığından kürk dokusunun tek başına neden olduğu kanıtlanmış değil. İşaretçi 11 val'de kalır; etiketler veya örnekler değiştirilmez.

İki seed'de strict base16'yı net +3/+4 videoyla geçti; sonraki geliştirme için uygun baz çizgisidir. Bu, yöntemin istatistiksel üstünlüğünü **kesinleştirmez**: eşleştirilmiş videoların bir kısmında aday geriledi, işaretçi 11 iki seed'de de kötüleşti ve keşifsel exact McNemar p değerleri 0,736/0,289. Ayrıca val'de merkez onarımı yapılan kare **sıfır**; gözlenen farkı yalnız “merkez onarımı”na atfetmek doğru değil. Bütün strict kırpım politikası karşılaştırılıyor. Sayımların kaynağı [seed 7 raporu](colab_seed7_sonuclari_2026-10-20.md).

## Ham video ve cache denetimi

`scripts/audit_val_error_quality.py`, seed 7/42 strict tahminlerini aynı resmî 199 val cache dosyasıyla eşleştirdi. Makinece okunabilir kayıtlar `outputs/colab_seed7_review/seed7_strict_quality_audit.json` ve `seed42_strict_quality_audit.json`. `scripts/inspect_preprocessing.py` ile yedi yanlış videonun 16 kare ham/landmark ve el kırpımı panoları `outputs/colab_seed7_review/error_panels/` altında üretildi. Bunlar teşhis görselleridir; yeni model sonucu değildir.

| Grup | Doğru/video | Tespit edilen el-kare | Yedek kırpım |
| --- | ---: | ---: | ---: |
| İşaretçi 11 | 26/30 | 814/960 = %84,79 | 146/960 = %15,21 |
| Diğer işaretçiler | 166/169 | 4574/5408 = %84,58 | 834/5408 = %15,42 |
| İşaretçi 11'in dört yanlış videosu | 0/4 | 107/128 = %83,59 | 21/128 = %16,41 |
| İşaretçi 11'in 26 doğru videosu | 26/26 | 707/832 = %84,98 | 125/832 = %15,02 |

İşaretçi 11'in **tespit oranı diğerlerinden düşük değil**; işaretçi 16'da yalnız 325/640 el-kare tespit edilirken seed 7 strict 20/20 videoyu doğru sınıflandırdı. Tespit boşluklarını azaltmak tek başına kök çözüm sayılmamalı. Sabit karelerde açık renk el, koyu kıyafetten görsel olarak ayrılıyor; bazı hızlı karelerde el ve kürk çevresi bulanık/örtüşüyor. “Tespit edildi” bilgisi parmak noktalarının doğru yerleştirildiğini kanıtlamaz; bu ölçüm ince konum hatasını, hareket bulanıklığını ve modelin sahne görünümüne bağlanmasını ayıramaz.

Seed 7 strict'in yedi yanlış videosu:

| Video | Gerçek → tahmin | Güven | Tespit/32 | Gözlem |
| --- | --- | ---: | ---: | --- |
| `signer11_sample342_color` | kitap → teşekkür | 0,76 | 30 | Hızlı harekette bulanıklık; iki bilekte birer kare uyuşmazlık. |
| `signer11_sample447_color` | okul → öğretmen | 0,69 | 26 | Hareketli elde 6 yedek kırpım; diğer el çoğunlukla sabit. |
| `signer11_sample570_color` | kitap → aile | 0,95 | 24 | 8 yedek kırpım; sol bilekte üç kare uyuşmazlık ve bulanık/örtüşen el. |
| `signer11_sample600_color` | okul → öğretmen | 0,68 | 27 | `447` ile benzer hareket/sahne; 5 yedek kırpım; ayrı video, özdeş cache değil. |
| `signer1_sample866_color` | kitap → kardeş | 0,95 | 26 | 6 yedek kırpım. |
| `signer1_sample1012_color` | aile → arkadaş | 0,98 | 32 | Her iki el 16 karenin tamamında tespitli; fallback çözümü beklenmez. |
| `signer35_sample60_color` | teşekkür → ev | 0,89 | 24 | 8 yedek kırpım; sol bilekte üç kare uyuşmazlık. |

Güven yalnız hata JSON'undaki model `top1_confidence` değeridir; olasılıkların kalibrasyonu doğrulanmış değil. 0,95 güven eşiği uygulansa bile kabul edilen 132 videoda **3 yanlış** kalıyor; yalnız eşik koymak canlı sistem güvenilirliğini çözmez. %100 Top-5, 10 sınıftan beşini listelemek demektir; gerçek zamanlı otomatik çeviri başarısı değildir.

## Mekanizma ve literatür

Google'ın MediaPipe açıklaması el algılamada örtüşme ve düşük kontrastın zorluğunu, gövde/kol bağlamının yer belirlemede yardımcı olduğunu belirtir; bu kullanıcının hipotezini biyolojik/görüntü işleme açısından makul kılar. Kaynak: [Google Research, MediaPipe Hands](https://research.google/blog/on-device-real-time-hand-tracking-with-mediapipe/). Ancak bizim val sayımlarımız işaretçi 11'de **genel tespit kaybı** göstermiyor. Sorun, tespitli fakat bulanık/yanlış parmak noktaları, el kırpımı dışındaki kol/yüz/gövde bağlamı veya eğitim dağılımına uymayan hareket/poz kombinasyonu olabilir.

AUTSL yarışma incelemesi, benzer işaretlerin küçük el konumu/hareket farklarıyla ayrıldığını ve güçlü sistemlerin iskelet, RGB/optik akış ve beden/yüz/el ipuçlarını birleştirdiğini bildirir. [ChaLearn sonuç makalesi](https://arxiv.org/html/2105.05066); [SAM-SLR yöntem makalesi](https://arxiv.org/abs/2103.08833). Bu literatür, mevcut dondurulmuş MobileNetV3 el kırpımı + landmark sistemine **tamamlayıcı** bir akış araştırmayı destekler; aynı puanın burada elde edileceğini garanti etmez.

## Önerilen en kısa deney sırası

1. **Önce görüntü bağımlılığını ayır:** `configs/experiment/autsl_10_poseonly_seed7.yaml` hazırlandı. Aynı strict cache, 16 kare, seed 7 ve 20 epoch; tek ana mimari değişiklik `model.use_rgb: false`. El RGB encoder/fusion atlanır, landmark ve zaman modeli eğitilir. Bu yol RGB dokuya bağımlı değildir, **MediaPipe landmarklarına hâlâ bağımlıdır**. İşaretçi 11'e yardımcı olup olmadığını ölçen **teşhis deneyi**; tek başına performans artışı iddiası değildir. Yerelde 1.245 train/199 val yükleme ve `(16,10)` ileri geçiş denetlendi. Colab'da ayrı çıktı klasörüyle eğit, yalnız val tahminlerini kaydet; yedi hata ID'si, işaretçi 11 ve toplam Macro-F1/Top-1 üzerinden seed 7 hibritle karşılaştır. Aynı cache kullanılır; yeni ön işleme gerekmiyor.
2. **Pose-only tamamlayıcıysa:** Güncel `src/evaluation/evaluator.py` val tahminlerine `class_probabilities` alanıyla on sınıfın tüm skorlarını ekler. ZIP'teki eski hibrit tahmininde bu alan yok; mevcut seed 7 hibrit checkpoint'ini **yalnız val'de yeniden değerlendirerek** skorları al. Ağırlıkları yalnız val'de seçilen geç füzyon için ayrı deneme yap; hibritin toplam doğru sayısını, işaretçi 11'i ve yeni bozulan videoları birlikte raporla. Canlıda iki modelin hesap/latans bedelini ölç. Pose-only zayıfsa ve özellikle işaretçi 11'i de düzeltmiyorsa bu yolu zorlamayalım.
3. **RGB görünüm sorunu devam ediyorsa:** Train-only, video boyunca tutarlı ve ölçülü kontrast/renk/gri ton veya arka plan çeşitliliği politikasını **tek ayrı deney** olarak sınayalım. Val ve test pikselleri aynı kalır; işaretçi 11'in video veya kırpımı eğitimde kullanılmaz. Fotometrik değişim mevcut kötü parmak koordinatlarını geri getirmez, bu yüzden sonuç baştan varsayılmaz. Landmark uyuşmazlığı baskın çıkarsa pose-guided yeniden tespit ya da farklı el izleyici, küçük train/val pilotunda ham video görsel kalite kapısından geçirilir.
4. **Hâlâ sınıf benzerliği baskınsa:** Pozdan türetilen üst beden RGB kırpımıyla hafif, önceden eğitilmiş ek zaman akışı deneyelim. Bu, “okul/öğretmen” gibi el şekli benzer ama konum/kol/yüz bağlamı farklı olabilecek işaretler için doğrudan hipotezdir. Ayrı cache, ayrı eğitim ve mobil/web gecikme ölçümü ister. 100/226 sınıfa ölçeklemede bu akışın yararı tekrar sınanır.

İlk Colab teşhis komutları (mevcut `/content/BitirmeProjesi` ve Drive düzeni varsayımı):

```bash
cd /content/BitirmeProjesi
export AUTSL_DATA_DIR=/content/drive/MyDrive/BitirmeProjesi/outputs/colab_training_metadata
python scripts/check_colab_inputs.py --base_config configs/experiment/autsl_10_base16_seed7.yaml --candidate_config configs/experiment/autsl_10_centerrepair_seed7.yaml --metadata_dir "$AUTSL_DATA_DIR" --checksum_manifest /content/drive/MyDrive/BitirmeProjesi/outputs/colab_cache_checksums.json --output outputs/poseonly_seed7_cache_preflight.json
python scripts/train.py --config configs/experiment/autsl_10_poseonly_seed7.yaml --cache_dir cache/autsl_10_features_centerrepair_strict
python scripts/evaluate.py --config configs/experiment/autsl_10_poseonly_seed7.yaml --cache_dir cache/autsl_10_features_centerrepair_strict --checkpoint /content/drive/MyDrive/BitirmeProjesi/outputs/autsl_10_poseonly_seed7/checkpoints/best_model.pt --split val --output outputs/poseonly_seed7_val.json --predictions_output outputs/poseonly_seed7_predictions.json
python scripts/analyze_val_pairs.py --base outputs/centerrepair_seed7_predictions.json --candidate outputs/poseonly_seed7_predictions.json --official_labels "$AUTSL_DATA_DIR/val_labels.csv" --output outputs/hybrid_vs_poseonly_seed7_pairs.json
```

Pose-only bazı hibrit hatalarını düzeltirse, aynı güncel değerlendirme koduyla
korunan hibrit checkpoint'inden tüm sınıf skorları alınır:

```bash
python scripts/evaluate.py --config configs/experiment/autsl_10_centerrepair_seed7.yaml --cache_dir cache/autsl_10_features_centerrepair_strict --checkpoint /content/drive/MyDrive/BitirmeProjesi/outputs/autsl_10_centerrepair_seed7/checkpoints/best_model.pt --split val --output outputs/centerrepair_seed7_val_scores.json --predictions_output outputs/centerrepair_seed7_predictions_scores.json
```

Bu yalnız yeniden değerlendirmedir; mevcut modelin ağırlıklarını veya eski
çıktıları değiştirmez. Yeni `class_probabilities` alanı `top1_confidence`
ile tutarlı olacak şekilde hedefli testten geçti.

Öncesinde güncel config/kodun ve iki cache manifestinin Colab kopyası denetlenmeli. `train.py` AUTSL `fit` aşamasında test yükleyicisini kurmuyor; resmî test yalnız nihai tarif sabitlenince açılır. Bu deneyi mevcut seed 7 checkpoint'inden **resume etme**; mimari farklıdır.

## Canlı sürüm için ek kanıt kapıları

Mevcut eğitim **önceden kesilmiş tek işaret videolarını** sınıflandırıyor. Canlı iletişimde başlangıç/bitiş belirleme, işaretler arası geçiş, bilinmeyen işaret, farklı telefon/kamera/ışık ve gecikme ayrıca ölçülmeli. Seed 7 checkpoint'i korunacak adaydır; gerçek kullanıcıda güvenilirlik ilanı için son tarifin resmî testte ve yeni kişilerle kaydedilmiş, eğitim/val dışında tutulan canlı video grubunda doğrulanması gerekir. Hata halinde seçenek sunma veya “emin değilim” kararı için kalibre edilmiş olasılık ve kapsama–hata eğrisi de ölçülmeli; val'deki tek 0,95 eşik yeterli değil.

Kaynak ZIP seed 7 `best_model.pt` ağırlığını içermiyor; tahmin JSON'unda
checkpoint SHA256 kimliği var. Canlı uygulamaya bağlama ve modelin yerelde
yeniden değerlendirilmesi için bu ağırlık dosyası ayrıca gerekecek.
