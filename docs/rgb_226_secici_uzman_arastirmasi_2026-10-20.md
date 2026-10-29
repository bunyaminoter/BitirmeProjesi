# AUTSL 226 sınıfta RGB uzmanı ve seçici kullanım planı

**Tarih:** 20 Ekim 2026. Bu belge mimari araştırma ve kod hazırlığıdır; yeni eğitim sonucu değildir. 10 sınıflı pose-only %99,50, 226 sınıf için RGB'nin gereksiz olduğunu göstermez.

**Sonraki bağımsız denetim:** Gemini'nin %10–15 seçici ConvNeXt mimarisi ve
mobil gecikme iddiaları için [fizibilite raporu](secici_uzman_kaskad_bagimsiz_degerlendirme_2026-10-20.md)
hazırlandı. Buradaki ConvNeXt adayı, onaylanmış ürün omurgası değildir.

## Sorun ve literatür

AUTSL'nin 226 sınıflı işaretçiden bağımsız protokolünde benzer işaretler özellikle zor; yarışma incelemesi güçlü yöntemlerin el/gövde/yüz ipuçlarını, RGB ve iskeleti birleştirdiğini bildirir: [ChaLearn 2021](https://openaccess.thecvf.com/content/CVPR2021W/ChaLearn/html/Sincan_ChaLearn_LAP_Large_Scale_Signer_Independent_Isolated_Sign_Language_Recognition_CVPRW_2021_paper.html). [SAM-SLR](https://arxiv.org/html/2103.08833v5) RGB, iskelet ve akışı ayrı akışlar olarak eğitip son sınıf skorlarını birleştiriyor. Bildirilen %98,42, **tam 226 sınıf resmî testinde çoklu akış/ensemble ile val videolarını da eğitime katan** sonucun değeridir; val'i eğitime katmadan test ensemble sonucu **%97,51**. Bizim tek modele veya train-only protokolümüze doğrudan hedef sayı değildir. [El kırpımı + poz akışı çalışması](https://openaccess.thecvf.com/content/CVPR2021W/ChaLearn/html/De_Coster_Isolated_Sign_Recognition_From_RGB_Video_Using_Pose_Flow_and_CVPRW_2021_paper.html) el görüntüsünün şekli, poz hareketinin benzer el şekillerini ayırmak için tamamlayıcı olabileceğini AUTSL'de gösteriyor.

Bizim 10 sınıf hibritte RGB kolu **ImageNet ön eğitimli fakat dondurulmuş MobileNetV3-Small**, yalnızca iki 160×160 el kırpımı ve erken cross-attention füzyonu kullanıyor. Seed 7 pose-only'nin 198/199, hibritin 192/199 olması, bu **spesifik** RGB kullanımının 10 sınıfta yardımcı olmadığını gösterir. RGB'nin tüm 226 sınıfta yararsız olduğuna dair kanıt değildir. Üstelik “pose-only” 21'er el/parmak noktası içerir; parmak görüntüsü eksikliği değil, görüntü dokusu/şekil ayrıntısı eksikliği tartışılıyor.

## Omurga seçimi

Torchvision'un ImageNet-1K model kartlarındaki sayılar farklı çözünürlük ve veri görevi içindir; AUTSL doğruluğu veya bizim 160×160 crop hızımız değildir:

| Omurga | ImageNet Top-1 | Parametre | GFLOPs, model kartı çözünürlüğünde | Buradaki rol |
| --- | ---: | ---: | ---: | --- |
| [MobileNetV3-Small](https://docs.pytorch.org/vision/main/models/generated/torchvision.models.mobilenet_v3_small.html) | %67,67 | 2,54 M | 0,06 @224 | Mevcut hafif, dondurulmuş referans |
| [MobileNetV3-Large](https://docs.pytorch.org/vision/main/models/generated/torchvision.models.mobilenet_v3_large.html) | %75,27 (V2) | 5,48 M | 0,22 @224 | Mobil cihaz için sonraki küçük aday |
| [ConvNeXt-Tiny](https://docs.pytorch.org/vision/main/models/generated/torchvision.models.convnext_tiny.html) | %82,52 | 28,59 M | 4,46 @224 | **İlk güçlü RGB araştırma adayı** |
| [EfficientNetV2-S](https://docs.pytorch.org/vision/main/models/generated/torchvision.models.efficientnet_v2_s.html) | %84,23 | 21,46 M | 8,37 @384 | Daha yüksek çözünürlük/bütçe karşılaştırması |

“En iyi” RGB omurgası bu tablodan belirlenemez. El ayrıntısı, temporal model, crop çözünürlüğü, eğitim verisi ve gecikme AUTSL sonucunu belirler. İlk somut aday olarak **ConvNeXt-Tiny** seçildi: mevcut torchvision akışına ek bağımlılık gerektirmeden giriyor; MobileNetV3-Small'den belirgin güçlü bir görüntü temsili sunuyor. Koddaki ConvNeXt özellik çıkarımına ön eğitimli son LayerNorm da dahil edildi. `efficientnet_v2_s` ikinci karşılaştırma seçeneği olarak kaydedildi. İki el × 16 kare nedeniyle ConvNeXt bile Colab ve canlı cihaz için pahalı olabilir; gerçek T4 belleği/süresi ve telefon/web gecikmesi ölçülmeden ürün modeli seçilmeyecek.

Yeni `configs/experiment/autsl_226_rgb_convnext_tiny.yaml`, 226 sınıf, 16 kare, **eğitilebilir** ConvNeXt-Tiny, paylaşılan iki el encoder'ı, poz akışı, cross-attention, AMP ve bellek için batch 2 × 8 accumulation ile bir **araştırma config'i**. Parametre sayısı yerelde 31.892.674. Gerekli `cache/autsl_226_features_centerrepair_strict` henüz yok; mevcut 10 sınıf cache'iyle çalıştırılamaz. 160×160 mevcut crop kalitesi başlangıç karşılaştırmasıdır; 224×224 isteği yalnız ham kareden ayrı cache/pilot ile yapılmalı, 160'tan yeniden boyutlandırma parmak bilgisi eklemez. Başlangıç ayarı T4 bellek garantisi değildir.

Yerel doğrulama: config 226 sınıf ve strict türetme ayarlarıyla yüklendi;
ConvNeXt-Tiny / EfficientNetV2-S özellik çıkarımı şekil testleri geçti,
ConvNeXt'li hibrit model kuruldu. Tam proje testleri **69/69 geçti**.

## RGB'yi yalnız gerektiğinde çalıştırmak mümkün mü?

**Evet, iki aşamalı çıkarımla mümkün:**

1. Her klipte 226 sınıflı poz+el-landmark modeli çalışır.
2. Poz çıktısı önceden belirlenmiş belirsizlik/benzer sınıf kuralını tetiklerse RGB uzmanı çalışır. İki modelin 226 sınıf skorları, val'de sabitlenmiş bir geç füzyon veya uzman seçimi kuralıyla birleştirilir.
3. Tetikleme yoksa RGB encoder hiç çalışmaz; bu gerçek hesap tasarrufu sağlar. Her örnekte RGB hesaplayıp yalnız ağırlığını sıfırlayan “gated fusion” **hesap tasarrufu sağlamaz**.

Bu bir garanti değil: poz modeli **yüksek güvenle yanlış** tahmin ettiğinde basit confidence/margin eşiği RGB'yi çağırmaz. Yakın sınıf çiftleri için ayrı tetikleme veya ikiden fazla aday sınıfın skoru kullanılabilir; yanlış tetiklemeler ve kaçan kurtarılabilir hatalar birlikte sayılmalı. Softmax güvenleri kalibre edilmeden eşik “gerçek hata olasılığı” değildir. Cascade güven/hesap maliyeti ilişkisi için [Learning to Cascade](https://ojs.aaai.org/index.php/AAAI/article/view/16900). Yaşama alınmadan önce signer bazında hata, RGB çağrı yüzdesi ve uçtan uca gecikme ölçülmeli.

## Deney sırası ve karar kapıları

1. **10 sınıfı burada bırak:** Seed 42 pose-only config'i korunuyor fakat kullanıcı tercihiyle zorunlu sonraki koşu değil. Tekrarlanan kolay 10 sınıf val iyileştirmeleri 226 hedefinin yerine geçmez.
2. **100 sınıfı stres testi yap:** Sınıf kümesi, train etiketleri ve işaret anlamları üzerinden önceden sabitlensin; özellikle benzer el şekli/hareketli çiftler bulunsun. Resmî train/val işaretçi ayrımı korunarak ayrı cache ve manifest hazırlanır. Poz-only baz çizgisi eğitilir; hatalar sınıf çifti, signer ve landmark kalitesine ayrılır. 100'deki doğruluk 226'ya taşınmaz.
3. **226 sınıfı ana ölçüm yap:** Tam resmî train/val için aynı landmark/crop sözleşmeli strict cache oluşturulup QA yapılır. Önce poz-only 226; sonra ConvNeXt RGB+poz adayını ayrı çıktı/aynı sınıf sırasıyla eğit. Gerekirse MobileNetV3-Large ve EfficientNetV2-S yalnız ölçülen kalite/hız ihtiyacına göre karşılaştırılır. Mevcut `autsl_226_mobilenet_frozen.yaml` geçmiş bir referans config'tir; 10 sınıftaki zayıf RGB koluna dayanarak nihai omurga sayılmaz.
4. **RGB'nin katma değerini video bazında ölç:** Poz-only / RGB+poz / sabit geç füzyon / seçici cascade için aynı resmî val videolarında Top-1, Macro-F1, signer dağılımı, zor sınıf çiftleri, RGB'nin kurtardığı ve bozduğu videolar, tetikleme oranı ve P50/P95 gecikme. Yalnız RGB'nin anlamlı ek kazancı varsa cascade eşiği ve füzyon ayarlanır. Basit kural işe yaramazsa her örnekte iki akış veya pose-only seçilebilir. Resmî test model/kapı seçimi bittikten sonra bir kez açılır.

**Uygulama sınırı:** Şu an seçici yönlendiriciyi “hazır model” diye kodlamak erken; 100/226 sınıfın iki modelden eşleşmiş skorları olmadan doğru tetikleme kuralı ve kazanç ölçülemez. Kodda güçlü RGB adayı hazırlandı, eğitimi ve yönlendirme seçimi veriler geldikçe yapılacak.
