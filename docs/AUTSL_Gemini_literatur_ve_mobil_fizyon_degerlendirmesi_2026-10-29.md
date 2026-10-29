# Gemini literatür notunun bağımsız değerlendirmesi

**Tarih:** 29 Ekim 2026  
**Kapsam:** Kullanıcının paylaştığı “Bu sorun, bilgisayarla görme ve işaret dili tanıma…” metnindeki iddialar; mevcut 100 sınıf resmî val sonuçları, proje kodu, 101 ortak hata incelemesi ve birincil yayınlar. Bu rapor yeni model eğitimi veya mobil hız ölçümü içermez.

## Kısa karar

Gemini'nin **geç füzyon** fikri doğru yönde: bizim önceden sabit 0,5/0,5 olasılık ortalamamız zaten 100 sınıf val'de **1.784/1.963 = %90,88** verdi. Bu, yalnız aynı 100 sınıf izole val kümesinin sonucudur. **Güven/tespit maskesi ve modalite dropout** araştırılabilir, fakat mevcut 174 bozulan videonun nedenini kanıtlamaz, eğitimsiz tak-çıkar çözüm değildir. **CorrNet/VAC, PoseC3D ve SignBERT** ciddi araştırma çizgileridir; mevcut AUTSL/mobil akışımıza doğrudan taşınan doğruluk veya hız garantisi yoktur. Şu anda 226 önbellek ve kalite kapısını tamamlamak daha düşük riskli.

## Bizim sonuçlarımız neyi gösteriyor?

- Pose-only **1.678/1.963 (%85,48)**; ConvNeXt-Tiny+pose **1.688/1.963 (%85,99)**. Hibrit, pose'un 184 hatasını düzeltti ve 174 doğrusunu bozdu. Aynı anda ikisinin de yanlış olduğu **101** video var. [Eşleşmiş sonuç raporu](AUTSL_100_RGB_pose_egitim_sonucu_2026-10-29.md).
- **1.862/1.963 = %94,85** iki modelin Top-1 doğrularının *oracle* birleşimidir: hangi modelin doğru bildiğini önceden bilen imkânsız bir yönlendirici için sayı. Ulaşılabilir mobil sonuç değildir. İki modelin de Top-1 yanıldığı 101 videonun üçü olasılık ortalamasıyla kurtarıldığından, bu sayı **her türlü skor füzyonu için matematiksel üst sınır** da değildir.
- Hibritin 174 bozduğu videoların nedeni “el crop'u pose'u zehirledi” diye belirlenemez. İki modelin ağırlıkları, kapasitesi, öğrenme dinamiği ve kalibrasyonu farklıdır. Hibritte train doğruluğu yaklaşık %100, val %86; aşırı uyum da olasıdır. Görsel hata panoları [ayrı incelemede](AUTSL_100_ortak_101_hata_ve_226_onbellek_kapisi_2026-10-29.md).
- Ek ucuz tanı: yerel strict NPZ'si bulunan **174 bozulma** ile **180 kurtarmada** `both_missing_frames` skorunun bozulma/kurtarma ayrımı ROC-AUC'si **0,528**; `fallback_crops` sayısının **0,538**. Rastgele ayrım 0,5'tir. Bozulanların **122/174**'ünde iki elin birlikte kaybolduğu kare yok. Bu göstergeler temel alınıp basit bir “el yoksa RGB'yi kapat” router'ına güvenilmemeli. Söz konusu AUC'ler yalnız bu val videosu ve hazır cache üzerinde keşifsel tanıdır, yeni performans ölçüsü değildir; 184 kurtarmanın dördünde yerel NPZ eksik olduğu için 180'i kapsar. [Video bazlı CSV](../outputs/autsl_100_shared_errors_2026-10-29/videos.csv), [yeniden üretim betiği](../scripts/audit_rgb_quality_router.py) ve [sayısal çıktı](../outputs/autsl_100_shared_errors_2026-10-29/rgb_quality_router_audit.json).

## Yayın iddiaları ve uygulanabilirlik

### 1. Geç füzyon ve çok akış

**Desteklenen fikir.** [SAM-SLR](https://openaccess.thecvf.com/content/CVPR2021W/ChaLearn/html/Jiang_Skeleton_Aware_Multi-Modal_Sign_Language_Recognition_CVPRW_2021_paper.html), AUTSL'de iskelet, RGB, akış ve başka modalitelerin skorlarını birleştirir. Bu takım **Northeastern University** kaynaklıdır. Metindeki “University of Vigo – MS-G3D şampiyonu” ifadesi iki ayrı yayını karıştırıyor: [Vigo MS-G3D çalışması](https://openaccess.thecvf.com/content/CVPR2021W/ChaLearn/html/Vazquez-Enriquez_Isolated_Sign_Language_Recognition_With_Multi-Scale_Spatial-Temporal_Graph_Convolutional_Networks_CVPRW_2021_paper.html) ayrı bir AUTSL araştırmasıdır ve S3D ile ek füzyonun sınırlı kazancını da bildirir.

**Bizde:** 0,5/0,5 *olasılık* ortalaması uygulandı. Gemini'nin “logitleri topla veya ağırlıklandır, %88–91 olabilir” öngörüsü artık hipotez değil: seçilmiş 100 sınıf val için gerçek ölçüm **%90,88**. Logit toplamı, olasılık ortalamasıyla aynı kural değildir; eğitimli modellerin logit ölçekleri farklı olabilir. Yeni ağırlığı aynı val üzerinde aramak iyimser sonuç verir. 226 için iki 226 modelin aynı 4.418 val tahmini beklenir.

**Mobil eksi:** İki ayrı model ve ConvNeXt'in 16 kare × 2 el görsel işleyişi çalışır. Sabit geç füzyon bu işi azaltmaz. Önce gerçek hedef cihazda kamera/örnekleme, MediaPipe, crop, her ağ, toplam gecikme, bellek ve ısınma ayrı ölçülmelidir. Colab A100 eğitim adım süresi telefondaki gecikme değildir.

### 2. Güven maskesi ve modalite dropout

**Makul ama doğrulanmamış aday.** Eksik modalitede eğitim dayanıklılığına ilişkin [ActionMAE araştırması](https://ojs.aaai.org/index.php/AAAI/article/view/25378), modalite düşürme ve rekonstrüksiyonun bazı aksiyon tanıma kümelerinde yararlı olabildiğini gösterir; AUTSL'de bizim crop+pose ağımıza %20 dropout veya 0,3 güven eşiği vermez.

**Kod gerçekliği:** Cache'de `left_hand_detected` ve `right_hand_detected` boolean alanları var. [`CachedTIDDataset`](../src/data/tid_datamodule.py) bu alanları model batch'ine taşımıyor; [`HybridASLModel`](../src/models/hybrid_model.py) el özniteliklerini her karede hesaplıyor, [`CrossAttentionFusion`](../src/models/fusion/cross_attention_fusion.py) bu üç özniteliğe dış tespit maskesi uygulamıyor. Dolayısıyla Gemini'nin “maskesiz akış” tanısı doğru. Ancak mevcut cache'de **kalibre edilmiş kare başı MediaPipe güven skoru** yok; boolean ve crop üretimindeki 0/0,5/1 heuristiği güven skoru yerine konmamalı.

**Uygulama riski:** İleri beslemede yalnız attention sonrasında maske koymak ConvNeXt maliyetini azaltmaz. RGB kolunu gerçekten atlamak için kararın CNN'den önce verilmesi ve modelin aynı maske politikasıyla eğitilmesi gerekir. “El yoksa karar tamamen pose'a geçer” ancak bağımsız pose modeline yönlendirmeyle garantilenebilir; hibritin kendi pose temsili bağımsız pose-only model değildir. El detektörü el şeklini yanlış çıkardığı halde `detected=true` de olabilir. Sabit %20 dropout/%0,3 eşik gerekçesizdir; ayrıntılı ablasyon ve sağlam videoda bozma denetimi ister. Bu bir **sonraki model deneyi** olabilir, mevcut cache'e sessizce eklenmez.

### 3. Tam kare, CorrNet/VAC ve el kırpımı

**Metindeki mutlak yargı desteklenmiyor.** [CorrNet](https://openaccess.thecvf.com/content/CVPR2023/html/Hu_Continuous_Sign_Language_Recognition_With_Correlation_Network_CVPR_2023_paper.html) komşu kare öznitelikleri arasındaki korelasyonla hareket izlerini öğrenir; çalışması *sürekli* işaret tanımadır. Ayrı optik akış hesapladığı veya örtülen parmakları geri getirdiği söylenemez. [VAC](https://openaccess.thecvf.com/content/ICCV2021/html/Min_Visual_Alignment_Constraint_for_Continuous_Sign_Language_Recognition_ICCV_2021_paper.html) sürekli işarette CTC hizalama ve görsel öznitelik eğitimi sorunu içindir; “tam kare + deformable attention” tarifi değildir.

AUTSL için karşı örnek: [De Coster ve arkadaşları](https://openaccess.thecvf.com/content/CVPR2021W/ChaLearn/html/De_Coster_Isolated_Sign_Recognition_From_RGB_Video_Using_Pose_Flow_and_CVPRW_2021_paper.html), gürültülü el keypoint kutusu yerine **dirsek/bilek tabanlı sağlam el kırpımı** ve pose flow kullanır. Yani literatür “el crop'unu terk edin” demiyor. Bizde crop konumunun bağlam kaybı pose koordinatlarıyla kısmen gideriliyor. Tam kare/üst gövde RGB yeni bir aday olabilir ama daha büyük görüntü, farklı model, yeniden eğitim ve cihaz ölçümü gerektirir; örtülmüş parmak bilgisini fiziksel olarak geri getirmez.

### 4. PoseC3D / ısı haritaları

[PoseC3D (PoseConv3D)](https://openaccess.thecvf.com/content/CVPR2022/html/Duan_Revisiting_Skeleton-Based_Action_Recognition_CVPR_2022_paper.html), koordinatları 2B eklem ısı haritalarına dönüştürüp 3B CNN ile aksiyon tanır; pozu kestirme gürültüsüne karşı dayanıklılık bildirir. **Doğru araştırma yönü**, ama AUTSL'de bizim girişimizle veya telefonda üstünlüğü kanıtlanmış değil. Boş ısı haritası kayıp eklemi açıkça temsil edebilir; “hiç aktivasyon/gradyan üretmez” ifadesi doğru değildir (evrişim komşuları ve bias bulunur). Isı haritası kayıp elin hareketini yaratmaz. Koordinat dizisine göre çok daha büyük uzamsal-zamansal tensör ve 3B CNN maliyeti vardır.

### 5. SignBERT ve anatomik ön eğitim

[SignBERT](https://openaccess.thecvf.com/content/ICCV2021/html/Hu_SignBERT_Pre-Training_of_Hand-Model-Aware_Representation_for_Sign_Language_Recognition_ICCV_2021_paper.html), el önbilgisiyle maskeli el pozu temsili ön eğitir; NMFs-CSL, SLR500, MSASL ve WLASL üzerinde değerlendirilir. Bu, bizim ön işlemde eksik el noktalarını kesin anatomik doğrulukla dolduracak hazır bir modül değildir. Görünmeyen parmak pozu tekil olarak çıkarılamaz; ön eğitim verisi, model ve ek hesap gerekir. Şu anki düşük bütçeli 226 kapısından önce öncelik değil.

## Karar sırası ve mobil kapı

1. **Şimdi:** 226 base16 resmî train/val cache'i tamamlama ve kalite denetimi. 100 sınıf `centerrepair_strict` v1 ve 0,5/0,5 ensemble sonucu dondurulur; yeni genel crop/dedektör kuralı eklenmez.
2. **226 eğitimi:** pose-only baz; sonra bütçe uygunsa aynı 226 train/val, 16 kare ve uyumlu ön işlemle RGB+pose. Tek modeller ve önceden sabit 0,5/0,5 füzyon val'de Top-1, Macro-F1, kurtarılan/bozulan videolar, işaretçiler ve yanlışken güven üzerinden karşılaştırılır. Resmî test tasarım kararına kapalıdır.
3. **Mobil karar:** Gerçek hedef telefonda MediaPipe dâhil uçtan uca p50/p95 gecikme, bellek, sürekli kullanımda ısınma/batarya, model boyutu ölçülür. Ağır ensemble bu kapıyı geçmezse pose-only ürün başlangıcı; RGB kazanımı büyük ve anlamlıysa daha küçük RGB uzmanı, ağırlık paylaşımı veya distillation ayrı geliştirme hedefi. Bu seçeneklerin doğruluğu bugün bilinmiyor.
4. **Sonraki araştırma:** 226 hataları RGB güvenilirliğiyle sistematik ilişkili çıkarsa, kayıp-el booleanını batch'e ekleyen ve aynı maske/dropout politikasıyla **yeniden eğitilen** hibrit tek kontrollü deney. Basit kayıp sayısı router'ı mevcut 100 sınıf tanıda zayıf olduğu için birincil seçim değil. CorrNet/PoseC3D/SignBERT mimari değişimi ve ek eğitim bütçesi olarak ayrı değerlendirilir.

**Kesinleşmeyenler:** 226 sınıf doğruluğu, model birleşiminin 226 kazancı, gerçek mobil gecikme ve güven maskesinin faydası ölçülmeden yüzde veya milisaniye vaat edilmez.
