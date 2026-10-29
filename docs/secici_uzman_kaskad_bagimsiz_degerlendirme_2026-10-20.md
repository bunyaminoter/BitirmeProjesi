# Seçici RGB uzmanı için bağımsız fizibilite incelemesi

> **21 Ekim güncellemesi:** Çalışmanın şu anki tek aktif adımı ve veri sınırı
> [güncel durum belgesindedir](MEVCUT_DURUM_VE_TEK_YOL_2026-10-21.md).

**Tarih:** 20 Ekim 2026  
**Karar durumu:** Mimari henüz onaylanmadı. Bu belge, Gemini'nin iki aşamalı önerisini uygulama ve bilimsel kanıt bakımından denetler; yeni bir 100/226 sınıf eğitimi veya telefon ölçümü içermez.

## Kısa karar

Pose-only modelini her klipte, RGB uzmanını yalnız seçilmiş kliplerde çalıştırmak **teknik olarak mümkün**. Ancak “%85–90 pose ile çözülür”, “pose 20 ms”, “ConvNeXt 400–800 ms”, “ConvNeXt-Tiny en uygun mobil omurga” ve “seçici sistem doğruluğu artırır” iddiaları bu proje için **ölçülmüş sonuç değil**. Şimdilik kesin ürün mimarisi seçilmemeli. İlk karar kapısı, 100 sınıfta pose-only hatalarını ve aynı videolarda RGB'nin *net* kurtarma gücünü ölçmek; nihai karar kapısı tam 226 sınıf ve hedef telefon üzerinde doğruluk–gecikme eğrisi olmalı.

**Ek literatür kontrolü:** [2026 RMA-MoE çalışması](https://www.sciencedirect.com/science/article/pii/S0167865526002795) seçici iskelet/RGB/derinlik sistemini gerçekten sınamış; ana ViSL120 deneyinde %58,5 yalnız iskelet çıkışı ve yaklaşık 2,3× ortalama uzman hesap tasarrufu, yeniden eğitilen üç uzmanla AUTSL'de %95,51 bildirmiştir. Bu seçici tasarımın mümkünlüğünü güçlendirir, fakat Gemini'nin %85–90 tek başına pose oranını veya bizim iki uzmanlı 16 kare mobil uygulamamızı doğrulamaz. Bütçe sınırıyla uygulanacak [iki koşuluk plan](100_sinif_iki_kosu_butce_plani_2026-10-20.md) ayrıca yazıldı.

## Öneride doğru olan ve kanıtsız kalan

1. **Mümkün olan işlem:** Pose sınıf skorlarına göre bir kural RGB kolunu çağırabilir. RGB hesaplanmadan geçilen kliplerde bu kolun hesap maliyeti tasarruf edilir. Tam RGB'yi her klipte çıkarıp sonunda sıfırla çarpan bir “gate” tasarruf sağlamaz. Girdi videolarının ve pose/hand landmarklarının elde edilme maliyeti ise devam eder.
2. **El CNN iş yükü:** 16 kare × 2 el = 32 kırpım doğrudur. Fakat mevcut [`HybridASLModel.forward`](../src/models/hybrid_model.py) soldaki 16 kırpımı tek batch, sağdaki 16 kırpımı ikinci batch olarak encoder'a verir; ifade edilen “32 ayrı forward pass” kod açısından doğru değildir. Aritmetik iş 32 kırpımlık olsa da çağrı sayısı, bellek kullanımı ve gecikme batch'e ve cihaz backend'ine bağlıdır.
3. **20 ms pose iddiası:** Mevcut [`inference_pipeline.py`](../src/preprocessing/inference_pipeline.py) her karede MediaPipe pose ve hand çalıştırır; pose-only durumda bile iki el kırpımını, strict fallback işlemlerini ve normalize RGB tensörlerini hazırlar. [`src/api/main.py`](../src/api/main.py) bir video yükleme sunucusudur; gerçek mobil cihaz ölçümü yoktur. Yalnız küçük pose ağının ileri geçiş süresi ölçülse bile bu, uygulamanın uçtan uca gecikmesi değildir. [Google'ın Hand Landmarker görev karşılaştırmasında](https://developers.google.com/edge/mediapipe/solutions/vision/hand_landmarker) Pixel 6'da tam el tespiti/landmark görevi için ortalama **17,12 ms CPU / 12,27 ms GPU** verilir; bu **bizim kombine MediaPipe + 16 kare + pose modeli süremiz değildir**, yalnız landmark çıkarımının sıfır maliyetli sayılamayacağını gösterir.
4. **400–800 ms RGB ve %10–15 tetikleme:** Donanım, precision, runtime, çözünürlük, batching, quantization ve video hazırlama belirtilmediği için proje ölçümü sayılamaz. Tetikleme oranı `q`, eşikten ve 226 sınıftaki sınıf/işaretçi dağılımından çıkar; baştan sabitlenemez. 100 sınıftaki oran 226'ya aktarılmaz.
5. **Omurga kıyası:** [TorchVision ConvNeXt-Tiny kartı](https://docs.pytorch.org/vision/main/models/generated/torchvision.models.convnext_tiny.html) **4,46 GFLOPs @224×224**, 28,59 M parametre ve 109,1 MB ağırlık dosyası verir. [EfficientNetV2-S kartı](https://docs.pytorch.org/vision/main/models/generated/torchvision.models.efficientnet_v2_s.html) **8,37 GFLOPs @384×384**, 21,46 M parametre verir. Bu iki GFLOPs değeri *farklı çözünürlüktedir*; 160×160 el kırpımında doğrudan 2× hız farkı veya mobil gecikme sonucu çıkarılamaz. “ViT esintisi parmak eklemlerini daha iyi yakalar” da eldeki AUTSL verisinde sınanmamış bir hipotezdir. ImageNet Top-1, işaret dili veya mobil ürün sıralaması değildir.

## Projenin mevcut kanıtı: 10 sınıf eşleşmiş tahminler

`outputs/colab_poseonly_seed7_review/poseonly_seed7_predictions.json` ile `outputs/colab_seed7_review/centerrepair_seed7_predictions.json` dosyalarını **aynı 199 video kimliği** üzerinden eşleştirip pose softmax üst iki olasılık farkına göre sıraladım. Pose-only **198/199**, eski MobileNetV3-Small hibrit **192/199**. Pose-only'nin tek yanlışı `signer16_sample317_color`: üst iki olasılık farkı **0,57177**; eski hibrit bu videoyu doğru biliyor.

Salt küçük margin kuralıyla en düşük marginli **20/199 (%10,05)** videoyu eski hibrite yönlendirirsek, o tek yanlış kurtuluyor fakat hibrit pose'un doğru bildiği **bir** videoyu bozuyor: sonuç **198/199**. İlk **30/199 (%15,08)** yönlendirmede **bir kurtarma, iki bozma** var: sonuç **197/199**. Bu, mevcut tahminlerdeki **retrospektif sert model değişimi** hesabıdır; yeni ConvNeXt'in davranışı değildir, eşik bu val üzerinde ayarlanıp bağımsız doğrulama yapılmadı ve 10 sınıf tek hatayla istatistiksel olarak çok küçüktür. Fakat “RGB'yi çağırmak otomatik iyileştirir” varsayımını çürütmek için somut bir uyarıdır. Daha iyi uzman/füzyon modeli bu tabloyu değiştirebilir.

AUTSL'de [çoklu akış sonuçları](https://openaccess.thecvf.com/content/CVPR2021W/ChaLearn/papers/Jiang_Skeleton_Aware_Multi-Modal_Sign_Language_Recognition_CVPRW_2021_paper.pdf) ve [el görüntüsü + pose-flow çalışması](https://openaccess.thecvf.com/content/CVPR2021W/ChaLearn/html/De_Coster_Isolated_Sign_Recognition_From_RGB_Video_Using_Pose_Flow_and_CVPRW_2021_paper.html) görüntü ile iskeletin tamamlayıcı *olabileceğine* dayanak sağlar; bu yayınlar bizim seçici kuralın, ConvNeXt omurgasının veya mobil sürenin başarılı olacağını kanıtlamaz. 10 sınıf başarısı 226 sınıf sonuç değildir.

## Doğruluk için gerekli koşul

`A = pose`, `B = RGB uzmanı`, `G = RGB çağrılan videolar` ve “çağrılınca son karar B'den gelir” ise:

`cascade doğru sayısı = A doğru sayısı + çağrılan A-yanlış/B-doğru − çağrılan A-doğru/B-yanlış`.

Bu eşitlikte uzman **A'nın yanlışlarını bulmalı**, bunları **doğruya çevirmeli** ve doğru A çıktısını **az bozmamalı**. Pose bir işareti yüksek güvenle yanlış söylüyorsa düşük margin kuralı o videoyu kaçırır. Önceden tanımlı karışıklık çiftleri yardımcı olabilir; ancak gerçek sınıf top-2 içinde değilse, modelin karışıklıkları zamanla/işaretçiyle değişirse veya yalnız train/val'in tekrar tekrar incelenmesiyle liste seçilirse yine başarısız olabilir. Güven kalibrasyonu tek başına hata tespitini garanti etmez: [Learning to Cascade](https://ojs.aaai.org/index.php/AAAI/article/view/16900) ve [Rethinking Confidence Calibration for Failure Prediction](https://arxiv.org/abs/2303.02970).

**Doğru ölçüler:** aynı videolarda her iki modelin tam 100/226 sınıf logitleri; pose hata sayısı, RGB'nin kurtarabileceği hata sayısı (oracle tamamlayıcılık üst sınırı), kuralın bu hataları yakalama oranı, çağrılan doğru pose tahminlerinden bozulanlar, net Top-1/Macro-F1, zor çift ve işaretçi başına sonuç. Yalnız `confidence` eşiğine ve genel confusion matrix toplamına bakmak yetmez. Liste/kapı/eşik *train içi* veya ayrılmış kalibrasyon bölümünde seçilmeli; resmî val üzerinde sürekli ayarlama val'i eğitim setine dönüştürür. Resmî test nihai tarif sabitlenince bir kez kullanılmalı.

## Gerçek zaman ve mobil maliyet hesabı

`T_uçtan_uça = T_işaretin tamamlanması/segmentasyon + T_kare edinme/decode + T_landmark/ön işleme + T_pose + G·(T_RGB_hazırlama + T_RGB_model + T_birleştirme) + T_sunucu/ağ (varsa)`.

**Yalnız varsayımsal örnek:** Pose ek süresi 20 ms ve her uzman çağrısı buna *ek* 400–800 ms olsaydı, `q=0,10–0,15` için ortalama **60–140 ms** yalnız bu iki model terimi olurdu. Uzman çağrılan klipte aynı terimler **420–820 ms** olur. Bunlar ölçüm veya vaat değildir; MediaPipe, video toplama, aktarım ve bellek ilk yükleme dışarıdadır. `q > %5` iken P95 gecikme genelde ağır kola yaklaşır; ortalama iyi olsa da zor işaretlerde kullanıcının beklediği süre uzun kalabilir. Uzmanı bellekte tutmak RAM kullanır, gerektiğinde yüklemek ilk tetiklemeyi geciktirir. Sık ağır çağrılar enerji ve ısı yaratabilir.

AUTSL izole işaret videosu üzerinden 16 kare seçmek, canlı akışta işaretin **ne zaman başlayıp bittiğini** çözmez. [`sample_video_frames`](../src/preprocessing/inference_pipeline.py) kayıtlı videonun toplam uzunluğunu bilerek `linspace` ile kare seçiyor. Canlı çeviride hareket bölütleme, bilinmeyen işaret/boş hareketi reddetme ve sürekli video üstünde yanlış tetiklemeler ayrıca geliştirilmelidir. Modelin 20 ms ileri geçişi, işaretin başından sonucun gösterilmesine kadar 20 ms anlamına gelemez.

[ExecuTorch backend kılavuzu](https://docs.pytorch.org/executorch/stable/backends-overview.html) cihaz/backend seçimi ve desteklenmeyen operatörlerin CPU'ya düşmesini vurgular; [quantization kılavuzu](https://docs.pytorch.org/executorch/stable/quantization-overview.html) quantize modelin doğruluğunun yeniden ölçülmesini önerir. Bu projede telefon üzerinde export, quantization, operatör desteği, P50/P95 veya ısınma testi henüz yok. Mevcut `src/api/main.py` telefona yerleşik çıkarım değil, sunucu üzerinden çıkarımdır.

## Riskler ve deneyle ayırma yolu

| Risk / olası eksik | Nasıl anlaşılır? | Gerekirse yön |
| --- | --- | --- |
| Pose hataları yüksek güvenli; gate kaçırır | Gate oranı–pose hata yakalama eğrisi, signer bazında | Top-k, kalite sinyali veya her videoda RGB; gate'i zorlama |
| RGB kurtardığından fazlasını bozar | Eşleşmiş rescue/harm sayıları | Geç füzyon, yalnız çift uzmanı, pose-only |
| İki model aynı hataları yapar | Oracle tamamlayıcılık çok düşük | Daha büyük model yerine veri/landmark/hareket kalitesi |
| MediaPipe ve kırpım sürekli maliyettir | Aşamaları ayrı profille | Pose-only yolunda kullanılmayan RGB hazırlığını atla; çıkarımı optimize et |
| ConvNeXt mobilde yavaş veya belleğe sığmaz | Aynı 160×160 ve hedef cihazda model/termal profil | Daha küçük RGB encoder, az RGB kare, distillation/quantization; yeniden doğrula |
| 100 sınıf kapısı 226'ya taşınmaz | Aynı raporu tam 226 val'de yeniden üret | 226 için yeniden kalibre et, 100'ü yalnız stres pilotu say |
| Web/sunucu ve sürekli video gecikmesi gizlenir | Kamera → ekrandaki sonuç P50/P95; bellek/ısı | Ürün mimarisini cihaz/ağ koşuluna göre seç |

## Önerilen karar sırası

1. **100 sınıf kümesini önceden sabitle:** Resmî train etiketleri ve görsel işaret tanımlarıyla yaygın/seyrek ve benzer işaretleri içersin; val hatalarına bakarak sınıf seçilmesin. Resmî train/val ayrımı korunsun. Ayrı strict 16 kare cache ve manifest kalite denetimi yapılsın.
2. **100 pose-only bazını kur:** Val'de sınıf ve işaretçi bazlı confusion, tam softmax/logit, video ID, landmark kalitesi ve zor çiftleri sakla. Bu, RGB gerekip gerekmediğini seçmek için ilk veri olur; otomatik confusion listesi tek başına “minimal çift” kanıtı değildir.
3. **RGB ek değerini önce küçük eşleşmiş deneyle sına:** Aynı 100 sınıf train/val videolarında mevcut hafif RGB referansı ve güçlü RGB+pose araştırma adayından en az biri için tam tahmin üret. ConvNeXt-Tiny config'i hazır ama eğitim sonucu yok. RGB'nin pose'un kaç yanlışını düzelttiğine ve kaç doğruyu bozduğuna bak. Adayın 160×160, aynı örnekleme ve aynı bütçe koşullarını kaydet.
4. **Kapıları karşılaştır:** Pose-only, her videoda RGB/hibrit, sabit geç füzyon ve seçici gate'i aynı videolarda değerlendir. `q` için sabit %10–15 dayatma; doğru–maliyet Pareto eğrisi çiz. Gate seçimini ayrılmış kalibrasyon verisinde yap, resmî val'i raporla.
5. **226 ve cihaz doğrulaması:** Tam 226 strict cache sonrası modelleri yeniden eğit ve aynı analizleri tekrarla. Hedef Android/iOS cihazı ve çalışma biçimi (tam cihaz mı, sunucu mu) netleşince MediaPipe, crop, pose, RGB, bellek, P50/P95, ısı/enerji ve gerçek kamera gecikmesini ölç. **Seçici mimariye yalnız net doğruluk kazancı ve kabul edilebilir zor-klip gecikmesi birlikte görülürse geç.**

**Şimdiki proje kararı:** 100 sınıf pose-only stres pilotuna ilerlemek makul. ConvNeXt-Tiny güçlü RGB araştırma adayı olarak kalır. Cascade, %10–15 tetikleme ve 20 ms ürün vaadi bu aşamada onaylanmış mimari değildir. Bu rapor tamamlandı; bu oturumda eğitim, cache üretimi veya telefon benchmark'ı yapılmadı.
