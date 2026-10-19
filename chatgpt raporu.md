# AUTSL model doğruluğu için proje inceleme ve deney raporu

**Tarih:** 19 Ekim 2026  
**Kapsam:** Proje kodu, `BitirmeProjesi/outputs` deney çıktıları, masaüstündeki `AUTSL_Dataset`, önceki `literatür araştırması.docx` ve doğrulanabilir AUTSL yayınları. Bu rapor yeni bir model eğitiminin sonucu değildir.

## Kısa sonuç

Masaüstündeki veri, ChaLearn 2021 AUTSL kıyaslamasında kullanılan 36.302 RGB videoluk, 226 sınıflı, işaretçiden bağımsız bölünmeyle **sayı, dosya adı ve sınıf kapsamı bakımından eşleşiyor**. Ancak çıkarılmış klasörlerde derinlik verisi yalnızca eğitim bölümünde bulunuyor. Mevcut %85,31 sonuç, bu 226 sınıflı resmi testte değil, **resmi AUTSL eğitim bölümünden seçilmiş 10 sınıfın 177 videoluk özel testinde** alınmış. Dolayısıyla literatürdeki %95–98 sonuçlarıyla doğrudan karşılaştırılamaz.

En yüksek doğruluğa gitmek için önce tam AUTSL veri okuyucusu ve güvenilir ölçüm hattı kurulmalı. Sonra güçlü bir RGB video modeli ile iskelet modelinin ayrı sonuçları ölçülmeli; yalnızca doğrulama kümesinde yararlı oldukları görülürse geç füzyon denenmeli. MobileNetV3-Small'in tek başına değiştirilmesi önemli olabilir, fakat mevcut kurulumda en büyük sınırlamalar 16 kareye indirgeme, yalnızca el kırpıntılarını kullanma, ImageNet omurgasını dondurma ve tam AUTSL'ye henüz uygun olmayan veri yükleyicisidir. %95+ ulaşılabilir bir araştırma hedefidir; donanım kredisi veya mimari benzerliği bunu garanti etmez.

## 1. Masaüstündeki veri kümesi

`C:\Users\Bünyamin\Desktop\AUTSL_Dataset` klasörünün etiket dosyaları ile video dosyalarını karşılaştırdım:

| Bölüm | Etiketli RGB video | İşaretçi | Sınıf | Sınıf başına örnek aralığı | Çıkarılmış derinlik videosu |
|---|---:|---:|---:|---:|---:|
| train | 28.142 | 31 | 226 | 90–127 | 28.142 |
| val | 4.418 | 6 | 226 | 12–20 | 0 |
| test | 3.742 | 6 | 226 | 10–17 | 0 |
| **Toplam** | **36.302** | **43** | **226** |  | **28.142** |

Her etiket satırının karşılık gelen `_color.mp4` dosyası var; her bölümde video kimlikleri tekil ve bölümler arasında çakışma yok. `SignList_ClassId_TR_EN.csv` sınıf adlarını içeriyor. Saptanan işaretçi kümeleri de birbirinden ayrı: resmi **val** işaretçileri 1, 11, 16, 18, 25, 35; resmi **test** işaretçileri 6, 14, 27, 30, 34, 39. Önceki masaüstü literatür belgesinde val ve test işaretçi listeleri karışmış; veri dosyalarını esas almak gerekir. ChaLearn makalesinin kullandığı 28.142 / 4.418 / 3.742 sayıları bu klasörle eşleşiyor [2]. Bu, dosya envanteri eşleşmesidir; videoların içerik karmaları veya tüm dosyaların çözülüp çözülemediği henüz doğrulanmadı.

**Modalite sınırı:** Resmi RGB yarışma yolunu mevcut çıkarılmış dosyalarla çalışabiliriz. Resmi RGB+D yolunu val/test derinlik dosyaları sağlanmadan aynı koşullarda çalışamayız. Derinlik modelini yalnızca train'de öğrenip testte RGB ile karşılaştırmak yanlış bir kıyas olur.

## 2. Önceki eğitimlerin gerçek anlamı

`outputs/tid_10_mobilenet_cv/cv_summary.json` içindeki beş katlı deneyin bulguları:

| Ölçüm | Sonuç |
|---|---:|
| 5 kat doğrulama ortalaması | %90,12 ± 4,84 yüzde puan |
| Beş modelin tek tek özel test ortalaması | %83,73 ± 3,79 yüzde puan |
| Beş model topluluğunun özel test doğruluğu | **%85,31 = 151 / 177** |
| Özel testte işaretçi sayısı | 6 |
| Sınıf sayısı | 10 |

Katlar arasında büyük fark var. Örneğin fold 1 doğrulamada %97,78 iken aynı özel testte %84,18; fold 4 doğrulamada %93,69 iken özel testte %76,84. Bu, küçük işaretçi gruplarına duyarlılığı gösteriyor. 177 video için %85,31'in yaklaşık %95 Wilson belirsizlik aralığı **%79,35–%89,77**; aralık yalnızca bu örneklem için istatistiksel belirsizliği anlatır, başka işaretçilere genellemeyi garanti etmez.

`data/tid_10_classes/labels.csv` içindeki 1.245 örneğin **tamamı resmi AUTSL train bölümünden** seçilmiş. Özel ayrım 1.068 train, 87 val ve 90 test örneği içeriyor; CV betiği val+test'i 177 videoluk özel test olarak birleştiriyor. Bölümleme kendi içinde işaretçiden bağımsız ve faydalı bir deney; resmi 226 sınıflı AUTSL test protokolü değildir. Tam veri train bölümünde bulunan `signer36`–`signer42` örnekleri, bu 10 sınıflı pakette val/test'e taşınmış.

`outputs/tid_10_mobilenet/checkpoints/history.csv` için en iyi val doğruluğu epoch 50'de %86,21; son epoch'ta eğitim doğruluğu %99,43 ve val doğruluğu %83,91. İncelenen fold 3 ve 4 grafiklerinde eğitim kaybı düşerken doğrulama kaybı daha erken duruluyor veya yükseliyor. Bu, mevcut temsil ve küçük eğitim kümesinde genelleme açığı olduğunu destekler; yalnızca “model zayıf” demek için yeterli değil.

Diğer `outputs` klasörleri ağırlıkla **ASL Citizen** deneylerine ait; AUTSL sonucu olarak sunulmamalı. `asl_citizen_100_overfit` geçmişindeki en iyi %98,75 val değeri özellikle yanıltıcıdır: ilgili yapılandırma `overfit_batches: 5` diyor ve `Trainer.evaluate` bu modda gerçek val yükleyicisi yerine eğitimde önbelleğe alınan aynı 5 batch'i değerlendiriyor. Bu, 80 örneklik hata ayıklama denemesidir. `asl_citizen_100_optimized` geçmişinde en iyi val yaklaşık %1,95; `asl_citizen_baseline` en iyi val %64. Bu klasörlerde bazı checkpoint veya grafikler mevcut olsa da, hepsinde tam geçmiş ve bağımsız test özeti yok. Sayısal sonuç için CSV ve JSON bulunan deneyleri esas aldım.

## 3. Kodun tam AUTSL'ye geçişteki engelleri

1. **Veri okuyucu uyumsuzluğu.** `src/data/datasets/tid_dataset.py` birleştirilmiş ve başlıklı `labels.csv` bekliyor (`split`, `tr_label`, `class_id`, `video_filename`). Masaüstündeki resmi `train_labels.csv`, `val_labels.csv`, `test_labels.csv` ise başlıksız iki sütundan oluşuyor (`video_id,label`). Yalnızca dosya yolunu değiştirerek 226 sınıf eğitimi başlamaz. `top_n_classes` parametresi de tanımlı olmasına rağmen yükleme içinde filtreleme yapmıyor. Yeni 226 sınıf yükleyicisi ya da açık dönüşüm adımı gerekli.
2. **Ön işleme yalnızca 10 sınıf paketine göre yazılmış.** `scripts/preprocess_tid.py` mevcut yapılandırma ile `train` ve `val` dolaşıyor, `test` işlenmiyor. Resmi tam AUTSL için üç bölümün açık yolları ve cache kapsam denetimi gerekir. Cache bulunmayan örnekler `CachedTIDDataset` içinde sessizce dışlanıyor; her bölüm için beklenen ve işlenen sayı eşit olmalı.
3. **Görsel bağlamın kaybı.** Mevcut hibrit model iki adet 160×160 el kırpıntısı ve MediaPipe'ın 33 poz + 42 el noktasını görüyor. Üst beden, iki elin birbirine göre konumu, yüz ve çevresel bağlamın görüntüsü modele doğrudan verilmediği için bazı sınıflar arasında bilgi kaybı olabilir. `num_frames: 16` videonun hareket akışını seyrek örnekliyor. 16/32/64 kare deneyi gereklidir; daha fazla karenin mutlaka daha iyi olacağı varsayılmamalı.
4. **Dondurulmuş küçük omurga.** `tid_10_mobilenet.yaml` MobileNetV3-Small'i ImageNet ağırlıklarıyla başlatıp tamamını donduruyor. Eğitim yalnızca projeksiyon ve sonraki katmanları uyarlıyor. Tam 28.142 videoluk train ile aşamalı çözme ve daha güçlü görsel omurga karşılaştırılmalı. Colab kredisi kullanılabilir GPU türünü, belleğini veya kesintisiz süresini garanti etmez; deneyler gerçek oturum donanımına göre ölçeklenmeli.
5. **Ölçüm disiplini.** `train_cv.py` her fold sonunda özel test performansını yazıyor. Bunun çok sayıda denemede model seçimi için kullanılması testin fiilen doğrulama kümesine dönüşmesine yol açar. Tam AUTSL'de yalnızca `val` ile model, epoch ve füzyon ağırlıkları seçilmeli; resmi `test` en sonunda bir kez kullanılmalı. Her koşu için sınıf haritası, split listesi, ön işleme sürümü, seed, donanım ve checkpoint kaydedilmeli.

## 4. Literatürde doğrulanmış kıyas noktaları

- AUTSL'nin ilk makalesi, **rastgele train/test bölünmesinde %95,95** rapor ediyor; aynı makaledeki **işaretçiden bağımsız benchmark en iyi başlangıcı %62,02**. Bu iki rakamın farklı protokollerle geldiği açıkça belirtiliyor [1]. %95,95'i resmi signer-independent test hedefi gibi kullanmamak gerekir.
- ChaLearn 2021 resmi AUTSL yarışması 226 etiket, 36.302 video ve işaretçiden bağımsız değerlendirme kullanıyor [2]. SAM-SLR yarışma çalışması **RGB yolunda %98,42**, **RGB+D yolunda %98,53** rapor ediyor [3]. Yöntem, iskelet ve farklı görsel/akış kollarını birleştiriyor; tek başına MobileNetV3 veya tek ST-GCN sonucu olarak yorumlanamaz. Yazarların resmi kodunda RGB, optical flow, iskelet ve derinlik türevleri ayrı veri hazırlama adımlarına sahip [4].
- Daha erişilebilir ara hedef için, 2023 tarihli deformable 3D CNN + iskelet çalışması AUTSL'de **RGB görünüm kolu tek başına %95,39**, iskelet kolu tek başına **%95,32**, ikisinin füzyonu **%96,67** rapor ediyor. Görünüm modeli üst beden kırpımı ve Kinetics/işaret dili ön eğitimi kullanıyor; bu sayılar da sıfırdan eğitilmiş küçük bir CNN ile eşdeğer değil [5].
- 2022 topluluk çalışması AUTSL testinde **%95,48** rapor ediyor ve çoklu modellerin olasılıklarını birleştiriyor [6]. Topluluk yalnızca üyeler farklı ve yeterli kalitedeyse yararlı; mevcut 10 sınıf örneğinde beş modelin ortalamasından %83,73'ten %85,31'e geçiş, %95'e otomatik sıçrama olmadığını gösteriyor.

Masaüstündeki önceki literatür belgesinde “aynı seviyeye kesinlikle ulaşılır”, “cache ile 50 kat hızlanır” ve “5-fold ensemble %95–98'e taşır” ifadeleri deney veya uygun bir kıyasla kanıtlanmıyor. Ayrıca rapordaki “Logos %98,60” iddiası için özgün makalede AUTSL resmi test koşulunu doğrulayacak açık bir kaynak bulamadım. Bu iddiaları hedef veya hipotez olarak tutmak doğru, kesin sonuç gibi sunmak doğru değil. Belgedeki “bizim model ST-GCN + EfficientNet-CBAM” anlatımı da şu anki **TİD 10 sınıf yapılandırmasıyla** örtüşmüyor: burada landmark MLP, dondurulmuş MobileNetV3-Small ve `use_cbam: false` var.

## 5. Önerilen deney sırası

**Aşama A — Ölçüm hattını doğrula.** Resmi `train/val/test` CSV'lerini sınıf kimliklerini değiştirmeden okuyabilen 226 sınıf yükleyici oluştur. Üç bölümün 28.142 / 4.418 / 3.742 RGB videosunu, işaretçi kümelerinin ayrıklığını, bozuk video sayısını ve her sınıfın indeksini otomatik doğrula. Ön işlemden sonra MediaPipe el/poz bulunma oranlarını kişi ve sınıf bazında ölç; kayıp özellikleri sessiz sıfır doldurma yerine raporla. 10 sınıf deneyinin checkpoint'lerini 226 sınıf metriği diye yeniden kullanma.

**Aşama B — Tek model temel çizgileri.** Önce resmi train ile, resmi val üzerinde 226 sınıflı basit bir RGB video modeli kur. Kullanılabilir pretrained ağırlık ve Colab GPU'suna göre üst beden kırpımlı R(2+1)D-18 veya benzer kısa video omurgası makul bir başlangıçtır. Ayrı bir MediaPipe iskelet modeli (joint/bone/motion akışları) kur. Her biri için 16/32/64 kare, giriş çözünürlüğü, ön eğitim ve omurga çözme ablation'larını bir değişkeni bir seferde değiştirerek yap. Gerekirse daha az GPU kullanan kare CNN + zamansal model, fakat tüm gövde görüntüsünü içeren bir temel çizgi kullan.

**Aşama C — Güçlendirme.** En iyi iki bağımsız modelin val tahminlerini kaydet; per-class karışıklık matrisi, signer bazında doğruluk, Top-1/Top-5, macro-F1 ve MediaPipe başarısızlığına göre performansı incele. Hatalar birbirini tamamlıyorsa füzyon ağırlığını yalnızca val üzerinde seç. Ardından resmi testte tek geçiş yap. 95+ hedefi için önce resmi testte güvenilir **%90+ tek model**, sonra **%95+** çoklu model hedefi koy; eşikler planlama hedefidir, tahmin değildir.

**Aşama D — Uygulama uyumu.** Mobil uygulama gerçek zamanlı çalışacaksa en yüksek doğruluklu çoklu modeli doğrudan telefona koymak zor olabilir. Ayrı olarak gecikme, bellek, kare örnekleme ve kamera verisiyle doğruluğu ölç. Gerekirse sunucu modeli, küçük mobil model veya distilasyon karşılaştır. AUTSL test doğruluğu, canlı kamera başarısının yerine geçmez.

## 6. Tekrar üretilebilir sonuç için teslim ölçütü

Sonraki eğitim raporunda en az şu bilgiler bulunmalı: kullanılan resmi split ve 226 sınıf haritası; veri/özellik çıkarımında başarılı video sayıları; model ve ön eğitim ağırlıkları; kare sayısı ve çözünürlük; donanım ve gerçek eğitim süresi; val seçme kuralı; en iyi epoch; en az üç seed için val sonuçları; testte tek model ve varsa ensemble Top-1, Top-5, macro-F1; işaretçi ve sınıf bazında hatalar. Karşılaştırma tablosunda 10 sınıflı özel test ile 226 sınıflı resmi testi ayrı tut.

## Kaynaklar

1. [Sincan ve Keles, AUTSL veri kümesi ve başlangıç modelleri](https://arxiv.org/abs/2008.00932) — rastgele bölünme ile işaretçiden bağımsız başlangıç arasındaki ayrım.
2. [ChaLearn LAP 2021 yarışma tasarımı ve sonuçları](https://arxiv.org/abs/2105.05066) — resmi 36.302 video, 226 sınıf ve değerlendirme protokolü.
3. [Jiang ve diğerleri, Skeleton Aware Multi modal Sign Language Recognition](https://arxiv.org/abs/2103.08833) — RGB %98,42 ve RGB+D %98,53 yarışma sonuçları.
4. [SAM-SLR resmi kodu ve veri hazırlama talimatı](https://github.com/jackyjsy/CVPR21Chal-SLR) — kullanılan kollar ve hazırlık adımları.
5. [Deformable 3D convolution ve iskelet füzyonu çalışması](https://sl-redu.e-ce.uth.gr/papers/ICASSP_2023.pdf) — tek kol ve füzyon ablation'ları.
6. [One Model is Not Enough](https://pmc.ncbi.nlm.nih.gov/articles/PMC9269724/) — AUTSL topluluk sonuçları.

**Yerel kanıt dosyaları:** `BitirmeProjesi/outputs/tid_10_mobilenet_cv/cv_summary.json`, `BitirmeProjesi/outputs/**/checkpoints/history.csv`, `BitirmeProjesi/data/tid_10_classes/labels.csv`, `BitirmeProjesi/configs/experiment/tid_10_mobilenet.yaml`, `BitirmeProjesi/scripts/train_cv.py`, `BitirmeProjesi/src/data/datasets/tid_dataset.py`, `BitirmeProjesi/src/data/tid_datamodule.py`, `BitirmeProjesi/src/training/trainer.py`; masaüstündeki `AUTSL_Dataset` ve `literatür araştırması.docx`.
