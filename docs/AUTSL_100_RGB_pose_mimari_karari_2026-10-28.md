# AUTSL 100 sınıf RGB ve pose mimari kararı

> **29 Ekim sonuç notu:** Bu plandaki tam eğitim tamamlandı. Tek hibrit modelin sınırlı net kazancı ve sabit olasılık füzyonunun güçlü kazanımı için [eğitim sonucu ve yeni karar](AUTSL_100_RGB_pose_egitim_sonucu_2026-10-29.md). Aşağıdaki "tam eğitim henüz başlamadı" ifadesi plan yazıldığı andaki durumdur.

**28 Ekim 2026 — seçilen deney:** Mevcut 100 sınıf pose-only bazına karşı, aynı `centerrepair_strict` v1 train/val videolarında **ImageNet ön eğitimli, tamamen eğitilebilir ConvNeXt-Tiny el kolu + mevcut pose kolu** eğitilecek. Bu, şu an için bir **araştırma modeli** kararıdır; mobil üretim modelinin ConvNeXt olacağı veya RGB'nin kesin artış sağlayacağı kararı değildir. [Seçilen config](../configs/experiment/autsl_100_rgb_convnext_tiny_seed7.yaml) hazır, yerel tek mikro batch ileri/geri kontrolü geçti. **Tam eğitim henüz başlamadı.**

## Önceki kararlar ve neden değişti

20 Ekim [RGB omurga araştırması](rgb_226_secici_uzman_arastirmasi_2026-10-20.md) ConvNeXt-Tiny'yi ilk güçlü **araştırma adayı** olarak seçmişti; telefon hızı ve gerçek AUTSL sonucu ölçülmeden ürün omurgası saymamıştı. 28 Ekim'deki ilk 100 sınıf koşusu pose-only **1.678/1.963 = %85,48 Top-1; 0,8524 Macro-F1** verdi. Sonraki 35 video/36 train referansı incelemesi, en sık üç karışıklığın yalnız eksik el yüzünden olmadığını gösterdi; [görsel ve geometrik kanıt](AUTSL_100_hata_tanisi_ve_226_karari_2026-10-28.md).

Önceki 10 sınıf seed 7 hibriti **192/199 (%96,48)** ve pose-only **198/199 (%99,50)** idi. Hibritteki RGB kolu **dondurulmuş MobileNetV3-Small** olduğundan, bu sonuç eğitilebilir RGB'yi veya 100 sınıfı elemez. Son turda bütçe için `MobileNetV3-Large` taslağı hazırlanmıştı. Kullanıcının gerekli Colab tokenlarını alabileceğini netleştirmesi ve tek 100 sınıf koşusunda görünüm bilgisini güçlü biçimde sınama hedefiyle **ConvNeXt-Tiny seçildi**. MobileNetV3-Large dosyası maliyet odaklı yedektir, çalıştırılacak config değildir.

## Birincil kaynakların söylediği

- [De Coster ve arkadaşları, AUTSL hand crop + pose flow](https://openaccess.thecvf.com/content/CVPR2021W/ChaLearn/papers/De_Coster_Isolated_Sign_Recognition_From_RGB_Video_Using_Pose_Flow_and_CVPRW_2021_paper.pdf): 16 kare ve iki el için paylaşılan ImageNet ön eğitimli ResNet-34 kullanır. Tam kare VTN **%82,03**, el kırpımı VTN-HC **%90,13**, buna pose flow eklenen VTN-PF **%91,51** AUTSL val doğruluğu bildirir. Bu, el görünümü ve konum/hareket ipuçlarının tamamlayıcı olabileceğine **doğrudan AUTSL kanıtıdır**; aynı sınıf seçimi, cache, omurga veya mimari değildir. ResNet-34 bu yüzden gerçek bir alternatif, fakat çalışmanın başarısı yalnız omurgaya atfedilemez.
- [SAM-SLR](https://arxiv.org/html/2103.08833): Tam AUTSL val'de keypoint akışı **%95,45**, RGB akışı **%94,77**; iskelet akışlarının ensemble'ı **%96,11**, RGB/akış eklenen ensemble **%96,96** bildirir. Güçlü keypoint varken görünüm yine tamamlayıcı olabilir. Onların RGB kolu **256×256 vücut kırpımı, 32 ardışık kare, ek ön eğitim ve çoklu model** kullanır. Bizim 160×160 iki el crop'u ve 16 yayılmış karemiz için bu yüzdeler tahmin değildir. %98,42 test sonucu val'i eğitime katan daha büyük ensemble'a aittir.
- Resmî TorchVision model kartlarında [MobileNetV3-Large V1](https://docs.pytorch.org/vision/main/models/generated/torchvision.models.mobilenet_v3_large.html) **5,48 milyon parametre / 0,22 GFLOPs / %74,04 ImageNet Top-1**, [ResNet-34](https://docs.pytorch.org/vision/main/models/generated/torchvision.models.resnet34.html) **21,80 milyon / 3,66 GFLOPs / %73,31**, [ConvNeXt-Tiny](https://docs.pytorch.org/vision/main/models/generated/torchvision.models.convnext_tiny.html) **28,59 milyon / 4,46 GFLOPs / %82,52** bildirir; bu FLOPs değerleri **224×224** içindir. [EfficientNetV2-S](https://docs.pytorch.org/vision/main/models/generated/torchvision.models.efficientnet_v2_s.html) **%84,23** ImageNet Top-1'i **384×384** ön işlem ve **8,37 GFLOPs** ile bildirir; doğrudan 160×160 el crop'u veya mobil hız üstünlüğü değildir. ImageNet sıralaması AUTSL sıralaması olarak okunmaz.

**Omurga değerlendirmesi:** ResNet-34'ün lehine AUTSL el kırpımı makalesi vardır, fakat o makale ResNet-34'ü ConvNeXt ile doğrudan karşılaştırmaz. ConvNeXt-Tiny'yi seçmemin nedeni, tek pahalı koşuda yetersiz RGB kapasitesinden doğan yanlış negatif karar riskini azaltmak, mevcut kodda desteklenmesi ve yerel 160×160 eğitim adımının çalışmasıdır. MobileNetV3-Large mobil ürün için anlamlı bir yedektir, fakat eski dondurulmuş Small başarısızlığı sonrası **tek araştırma koşusunu en hafif omurgaya harcamak** RGB'nin potansiyelini zayıf sınayabilir. ConvNeXt'in daha doğru olacağını **kanıtlamadık**; bu bilinçli, test edilebilir mimari hipotezidir.

## Yerel karşılaştırma ve gerçek ön kontrol

Aynı **2 video × 16 kare × 2 el × 160×160** girişinde, AMP ve gradyan checkpointing ile RTX 5050 Laptop GPU üzerinde kısa **rastgele ağırlıklı hesap ölçümü** yapıldı. MobileNetV3-Large için toplam hibrit **7,06 M parametre**, kararlı mikro adım medyanı yaklaşık **0,10 sn**, tepe ayrılmış GPU belleği **0,50 GiB**; ResNet-34 için **25,26 M / 0,10 sn / 0,70 GiB**; ConvNeXt-Tiny için **31,86 M / 0,14 sn / 1,48 GiB**. [Ham yerel ölçüm](../outputs/autsl_100_rgb_architecture_review/benchmark_local.json). Bu kısa sentetik ölçüm, Colab T4/A100, DataLoader, depolama, tam epoch ve telefon gecikmesi değildir; tek tek omurgaların AUTSL doğruluğunu da ölçmez.

Seçilen ConvNeXt config'iyle **gerçek strict cache** ve ön eğitimli ağırlık üzerinde tek mikro batch ileri/geri geçişi de başarılı oldu: **12.405 train + 1.963 val** NPZ kapsamı, sınıf sırası, işaretçi ayrılığı ve `[2,100]` logit şekli doğrulandı; tepe ayrılmış bellek **1,48 GiB**, ısınma sonrası mikro adım medyanı bu çalıştırmada **0,188 sn**. [Yerel ön kontrol JSON](../outputs/autsl_100_rgb_architecture_review/convnext_preflight_local.json). İlk adım CUDA/optimizatör ısınmasıyla yaklaşık 10 saniye sürdü; epoch hesabına bu ilk adım tek başına çarpılmamalı. Colab'da aynı [ön kontrol betiği](../scripts/preflight_100_rgb_pose.py) yeniden çalıştırılmalı.

## Tek 100 sınıf koşusunun kesin mimarisi

1. **Girdi ve protokol:** Seçili 100 sınıfın aynı sırası; 12.405 resmî train, 1.963 resmî val; 16 eşit zamanlı örnek kare; `centerrepair_strict` v1 cache; resmî test kapalı. Yeni 36 kare, yeni el onarımı veya val'i train'e ekleme yok.
2. **RGB kolu:** Her karede iki **160×160 el crop'u** aynı ImageNet ön eğitimli ConvNeXt-Tiny ağırlıklarıyla işlenir. Omurga **eğitilebilir**; sol/sağ el için ağırlık paylaşımı parametre sayısını ikiye katlamaz. Her elden 256 boyutlu özellik projeksiyonu alınır. Crop kareleri mevcut strict cache'ten gelir; bu kol tam gövde RGB'sini görmez.
3. **Pose kolu:** Önceki modelin aynı 33 gövde + 21 sol el + 21 sağ el = **75 nokta, 225 koordinat** girdisi; aynı landmark MLP. Mevcut model kare başına geçerli noktaları merkezleyip ölçekliyor, sonra TCN/Transformer ile zamana yayıyor. Bu koşuda pose akışı korunur; aynı anda yeni pose flow/anchor özelliği eklenmez.
4. **Birleştirme ve eğitim:** Mevcut **kare düzeyinde cross-attention** ile pose/sol RGB/sağ RGB birleştirilir, aynı TCN/Transformer ve 100 sınıf başlığı kullanılır. Seed **7**, en fazla **60 epoch**, en iyi val checkpoint'i, doğruluk ilerlemezse **12 epoch** erken durdurma, AdamW **1e-4**, AMP, A100 üzerinde **fiziksel batch 16 × biriktirme 1 = etkin batch 16**, gradient checkpointing açık. Pose-only 29. epoch'ta en iyi sonucu gördüğü için daha yavaş yakınsayabilecek eğitilebilir RGB kolunu 35'te zorunlu kesmiyoruz; bu üst sınır tam 60 epoch koşulacağı anlamına gelmez. Colab GPU ön kontrolünde bellek/süre görülür; A100 yoksa [T4/L4 profilindeki](../configs/experiment/autsl_100_rgb_convnext_tiny_seed7_t4_l4.yaml) **8 × 2 = 16** kullanılır. Bu model pose-only ile aynı veri/etiket/seed/16 kareyi paylaşır, fakat RGB kolu ve füzyon nedeniyle eşit parametreli tek değişkenli ablation değildir.
5. **Çıktı sözleşmesi:** Config/kod sürümü, cache manifesti, eğitim eğrisi, en iyi/son/devam checkpoint'i, 1.963 resmî val videosunun 100 sınıf olasılıkları ve video ID'leri saklanır. Pose-only checkpoint'i ve tahmin JSON'u değişmez.

**Bu deneyin çözebileceği hata:** Crop'ta MediaPipe noktalarının kaçırdığı parmak biçimi, avuç yönü veya görünüm ayrıntısı varsa model yeni bilgi öğrenebilir. `şeker→Atatürk` ve `okul→beklemek` bunun olası adaylarıdır; crop bulanıklığı nedeniyle garanti yoktur. `onlar→kalem` için ayırıcı el konumu pose girdisinde zaten var; RGB crop konumu merkezlediği için yalnız daha büyük CNN'in bu çifti kesin düzelteceğini söyleyemeyiz. Signer16'daki uzun tespit boşlukları veya yanlış crop da CNN kapasitesiyle otomatik düzelmez.

## 28 Ekim batch ve Colab GPU kararı

Önceki **batch 2 × biriktirme 8** ayarı yerel RTX 5050 üzerinde bellek riski düşük bir ön kontrol ve olası düşük bellekli çalıştırma içindi; doğruluğu en yüksek yapan keşfedilmiş hiperparametre değildi. O ön kontrol yalnız 2 örneğin geçişini ölçtü; 16 örneğin Colab belleği ve hızını kanıtlamaz. [PyTorch AMP belgeleri](https://docs.pytorch.org/docs/stable/notes/amp_examples.html), etkin batch'in mikro batch × biriktirme adımı olduğunu açıklar. Pose MLP'de BatchNorm bulunduğu için 2 × 8 ve 16 × 1 matematiksel olarak tamamen aynı eğitim değildir; fiziksel batch 16'nın daha kararlı istatistik sağlaması olasıdır, doğruluk artışı garanti değildir.

**Tercih:** Colab'da A100 varsa [seçilen config](../configs/experiment/autsl_100_rgb_convnext_tiny_seed7.yaml) ile **16 × 1**. A100 yoksa T4 veya L4 için [ayrı config](../configs/experiment/autsl_100_rgb_convnext_tiny_seed7_t4_l4.yaml) ile **8 × 2**; preflight OOM verirse tam eğitime başlamadan **4 × 4** profiline inilir. Her iki hazır profilde AMP ve gradient checkpointing açık kalır; [PyTorch checkpointing açıklamasına](https://docs.pytorch.org/docs/main/utils/checkpoint.html) göre bellek kazanımı geri geçişte ek hesaplama gerektirir. A100'de checkpointing'i kapatma fikri yalnız preflight ile bellek ve adım süresi karşılaştırıldıktan sonra, tam koşu başlamadan değerlendirilebilir. Aynı eğitim başladıktan sonra batch profilini değiştirmeyiz; yeniden başlatmada checkpoint ve config aynı kalır.

Batch'i küçültmek tek başına ezberlemeyi önleyen yöntem değildir. Bu koşuda işaretçi ayrık resmî train/val, mevcut hafif RGB/pose artırmaları, AdamW weight decay **0,01**, classifier dropout **0,4**, label smoothing **0,05**, en iyi val checkpoint'i ve patience **12** korunur. Train doğruluğu yükselip val kaybı kötüleşirse en iyi epoch seçilir; yalnız batch'i daha da küçültmek ilk müdahale değildir. GPU seçimi bir doğruluk hiperparametresi değildir. [NVIDIA](https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/nvidia-a100-datasheet-us-nvidia-1758950-r4-web.pdf) A100 için 40/80 GB, [T4 için](https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/tesla-t4/t4-tensor-core-product-brief.pdf) 16 GB, [L4 için](https://www.nvidia.com/en-gb/data-center/l4/) 24 GB bellek bildirir. Gerçek Colab maliyeti/erişimi ve oturum limitleri değişir; [Google Colab SSS](https://research.google.com/colaboratory/faq.html) sabit GPU veya kullanım kotası garantisi vermez. Bu nedenle GPU başına compute unit verimliliği ancak gerçek preflight ve Colab'ın gösterdiği tüketimle ölçülür.

**Çalıştırma kapısı:** A100 notebook'u en az 32 GiB GPU görmezse tam eğitime geçmez. Preflight 16 örnekle gerçek ileri/geri geçişi, ayrılan tepe belleği ve ısınmış adım süresini raporlar. Tepe bellek toplam GPU belleğine fazla yakınsa veya OOM olursa, tam koşudan önce T4/L4 profilindeki 8 × 2'ye geçilir; koşuların çıktıları ayrı klasörlerdedir. Daha büyük batch ya da daha çok epoch'un daha yüksek val doğruluğu sağlayacağı önceden söylenemez.

Colab ekranındaki **Yüksek RAM** anahtarı CPU belleğini artırır, GPU VRAM'ini değil; ilk koşuda kapalı kalabilir. Cache yerel `/content` SSD'ye kopyalanarak eğitimde Drive'dan binlerce küçük NPZ okuma beklemesi azaltılır. A100 preflight tahmini yalnız saf GPU adımı içindir; gerçek epoch süresini ilk tam epoch sonunda ölç. T4/L4 zorunluysa A100 notebook'u yerine aşağıdaki config ile aynı preflight ve train komutlarını çalıştır; kendi `outputs/autsl_100_rgb_convnext_tiny_seed7_t4_l4/` dizinine yazar:

```bash
python scripts/preflight_100_rgb_pose.py --config configs/experiment/autsl_100_rgb_convnext_tiny_seed7_t4_l4.yaml --cache_dir /content/BitirmeProjesi/cache/autsl_100_features_centerrepair_strict
python scripts/train.py --config configs/experiment/autsl_100_rgb_convnext_tiny_seed7_t4_l4.yaml --cache_dir /content/BitirmeProjesi/cache/autsl_100_features_centerrepair_strict
```

## Eğitim sonrası karar ölçüsü

**Sadece iki doğruluk yüzdesini yan yana koymayacağız.** Aynı 1.963 val videosunda Top-1 ve Macro-F1 ile şu eşleşmiş sayılar hesaplanacak: pose yanlış → hibrit doğru (**kurtarma**), pose doğru → hibrit yanlış (**bozma**), ikisi de yanlış/doğru. Üç tekrarlı çift, 6 val işaretçi, düşük ve yüksek güvenli hatalar ayrı görülecek. Farkın belirsizliği eşleşmiş video bootstrap aralığı veya exact McNemar ile verilecek; alt küme/işaretçi sayıları küçükken kesin üstünlük dili kullanılmayacak.

Hibrit net kazanç sağlarsa 226 sınıf için güçlü RGB adayı olur. Hibrit tek başına zayıf ama farklı pose hatalarını düzeltiyorsa, **yeni eğitim yapmadan** iki modelin kaydedilmiş skorlarıyla sabit ağırlıklı geç füzyon ayrıca incelenir; val'e göre çok sayıda ağırlık/eşik denemesi resmî test için yanlılık yaratır. Hibrit ne genel ne zor çiftlerde katkı verirse 226 için pose/temsil ve tam gövde bağlamı öncelik alır. Bu deney yalnız **100 sınıfta bu ConvNeXt el crop + pose mimarisinin** katkısını ölçer; bütün RGB yöntemlerini kesin elemez veya 226 sonucunu garanti etmez.

## Mobil uygulama kararı ayrı

ConvNeXt-Tiny'nin 100 sınıfta kazanması, her klipte 32 el görüntüsü işlemesini mobilde kabul edeceğimiz anlamına gelmez. Önce gerçek cihazda uçtan uca gecikme, güç ve ısınma ölçülür. Ardından gerekirse küçük RGB omurgaya distilasyon veya seçici uzman düşünülür. **%10–15 uzman çağrısı** ve **20 ms pose gecikmesi** bu projede ölçülmüş değerler değildir. Pose'un yüksek güvenle yaptığı hatalar bulunduğundan yalnız düşük margin tetiklemesi de önceden kabul edilmez.

**Şimdiki işlem:** Seçilen config ve preflight dosyasını Colab projesine taşı; yerel Colab cache yolunda ön kontrolü çalıştır; bellek/süre uygunsa tek tam 100 sınıf eğitimini başlat. Tam eğitimi bu bilgisayarda çalıştırmadım. Colab çalışma dizini `/content/BitirmeProjesi`, `AUTSL_DATA_DIR=/content/BitirmeProjesi/data/autsl` ve strict cache'in Colab yerel SSD kopyası hazır olduğunda sıra:

```bash
python scripts/preflight_100_rgb_pose.py \
  --config configs/experiment/autsl_100_rgb_convnext_tiny_seed7.yaml \
  --cache_dir /content/BitirmeProjesi/cache/autsl_100_features_centerrepair_strict \
  --output /content/drive/MyDrive/BitirmeProjesi/outputs/autsl_100_rgb_architecture_review/convnext_preflight_colab.json

python scripts/train.py \
  --config configs/experiment/autsl_100_rgb_convnext_tiny_seed7.yaml \
  --cache_dir /content/BitirmeProjesi/cache/autsl_100_features_centerrepair_strict

python scripts/evaluate.py \
  --config configs/experiment/autsl_100_rgb_convnext_tiny_seed7.yaml \
  --checkpoint /content/drive/MyDrive/BitirmeProjesi/outputs/autsl_100_rgb_convnext_tiny_seed7/checkpoints/best_model.pt \
  --cache_dir /content/BitirmeProjesi/cache/autsl_100_features_centerrepair_strict \
  --split val \
  --output /content/drive/MyDrive/BitirmeProjesi/outputs/autsl_100_rgb_convnext_tiny_seed7/val_metrics.json \
  --predictions_output /content/drive/MyDrive/BitirmeProjesi/outputs/autsl_100_rgb_convnext_tiny_seed7/val_predictions.json
```

Colab preflight çıktısı GPU adını, belleği ve ısınma sonrası mikro adım süresini verir. Eğitim kesilirse aynı koşu `--resume /content/drive/MyDrive/BitirmeProjesi/outputs/autsl_100_rgb_convnext_tiny_seed7/checkpoints/latest_model.pt` ile sürdürülür; yeni deney başlatılmaz. En iyi checkpoint'ten bütün val tahminleri ve olasılıkları saklanır.
