# AUTSL 226 sınıf model doğruluğu yol haritası

**Başlangıç:** 19 Ekim 2026  
**Amaç:** Resmi AUTSL işaretçiden bağımsız ayrımında, yeniden üretilebilir ve giderek artan Top-1 doğruluk elde etmek. Nihai yüksek doğruluk hedefi %95+; bu eşik garanti edilen sonuç değildir.

## Çalışma kuralı

Resmi `train` ile ağırlık öğrenilir, `val` ile epoch ve deney seçilir. Resmi `test` etiketleri ve sonucu, tasarım kararları tamamlanana kadar model seçmek için kullanılmaz. Eski 10 sınıflı %85,31 deney ayrı bir protokoldür; yeni 226 sınıf tablolarına katılmaz. Her deney tek bir ana değişkeni sınar ve önceki başarılı checkpoint korunur. Kullanılan config, ön işleme manifesti, seed ve veri kapsamı sonuçla birlikte saklanır.

## Başlangıç — Resmî 10 sınıflı doğrulama

İlk çalışma, önceki on işareti resmî AUTSL bölümlerinden alır: `train=1.245`, `val=199`, `test=167`. Kullanılacak dosya `configs/experiment/autsl_10_mobilenet.yaml`; eski özel CSV, 5-fold CV ve `val+test` birleşimi kaldırılmıştır. Ön işleme üç bölüm için ayrı cache üretir. Eğitim yalnızca train cache'ini, erken durdurma ve seçim yalnızca val cache'ini okur. Test cache'i eğitim aşamasında yüklenmez.

İlk karar sırası: veri denetimi → üç split ön işleme → MobileNetV3 temel çizgisi → yalnızca val üzerinden hata analizi. Test, bu kararlar tamamlandığında bir kez çalıştırılır. 10 sınıfta güvenilir şekilde çalışan akış, sınıf listesi değiştirilerek 100 ve 226 sınıfa genişletilir.

**19 Ekim düzeltmesi:** İlk üç videoluk teknik pilotta MediaPipe el etiketlerinin poz bileklerine göre ters kaydedildiği saptandı. Etiket eşlemesi düzeltildi ve o üç `.npz` ile eski pilot raporu kaldırıldı. Yeni cache manifesti `pipeline_version=2` ve seçilen özgün sınıf kimliklerini içerir; eski cache ile eğitim durur. Model başlatılamazsa boş özellik yazmak yerine ön işleme hata verir. Hız artırımı sonrası attention maskesi güncellenir; kayıp eklemlere rastgele gürültü eklenmez. El kırpımına uygulanan, fakat eklem koordinatlarına uygulanmayan dönme ve kaydırma ilk temel çizgide kapatıldı.

**10 sınıf pilotu:** Yerelde 38 train, 36 val ve 20 test videosunun cache'i üretildi. Train'de sol/sağ el bulunma oranı %83,39/%85,03; val'de %84,38/%92,53. Train ve val bilek uyumu %96–97 civarında. Hiçbir dosya veya özellik okuma hatası görülmedi. İlk eşit aralıklı örnekleme her sınıfı kapsamıyordu; `--pilot_size` artık sınıf dengeli seçim yapıyor ve 20 videoda her sınıftan iki örnek seçiyor. 94 pilot dosyası 102 MiB; aynı ortalama ile tam 1.611 dosya yaklaşık 1,7 GiB olur. Gerçek boyut değişebilir. Test verisinden model kararı verilmeyecek.

**Sıradaki adım:** Önce her sınıf ve birkaç işaretçiden görsel el kırpımı ve iskelet incele; yalnızca tespit oranı doğru kırpımı garanti etmez. 16 ve 36 kareyi aynı videolar üzerinde model eğitmeden görsel olarak karşılaştır. Sonra küçük, ayrı cache'li 36 kare pilotunda tespit kalitesini ölç. Kalite sorunları çözülünce 1.611 videonun ana cache'ini tamamla ve `--require_cache` ile eksiksizliği doğrula. Nihai kare sayısı kararı yine aynı modelin resmî val karşılaştırmasıyla verilir.

### Ön işleme kalite kapısı ve literatürden deneyler

`scripts/inspect_preprocessing.py`, aynı resmî train veya val videosundan 16/36 kare panoları üretir. Kare sayısı cache manifestiyle eşleşirse kol, bilek ve 21 el noktası çizilir; el kırpımları ayrıca gösterilir. Eşleşmeyen kare sayısı **yalnızca RGB** gösterir; o sayıdaki tespit başarısını ölçtüğünü iddia etmez. Varsayılan seçim sınıf başına bir önbellekli örnek kullanır. Kararlar için test bölümü açılmaz.

```powershell
python scripts/inspect_preprocessing.py --split val --max_videos 10
python scripts/inspect_preprocessing.py --split train --max_videos 10
```

Kontrol listesi: (1) etkin hareket 16 karede kayboluyor mu; (2) baş/son nötr pozlar çok yer kaplıyor mu; (3) poz dirsek ve bilekleri gerçek uzuvlara oturuyor mu; (4) her elin 21 noktası doğru ele ve parmaklara oturuyor mu; (5) kırpım parmak uçlarını kesiyor mu veya fazla arka plan taşıyor mu; (6) eksik tespit tek kare mi, uzun seri mi; (7) işaretçi, sınıf, ışık, hareket bulanıklığı veya el çakışmasında sorun yoğunlaşıyor mu. `*_inspection.json` yalnızca sayısal taramadır; insanın görsel kontrolünün yerine geçmez. Tek elli işarette diğer elin bulunmaması otomatik hata değildir.

İlk pilotta seçilen `aile` val videosunda kollar ve parmaklar genel olarak izleniyor; 16 karede bir elde 13/16, ötekinde 15/16 tespit var. 36 RGB karede çok sayıda birbirine yakın ve başlangıç/son bekleme karesi görülüyor. Sınıf başına bir örnekle genişletilen 10 videoluk taramada `kardeş` örneğinde her iki el 6/16 karede bulundu; eller yüz önünde çakışırken art arda 5 ve 9 karelik kayıp görüldü. Bu, öncelikli inceleme örneğidir. Bu gözlemler **küçük pilot** içindir; 16 kareyi tüm veri için yeterli veya 36 kareyi gereksiz kılmaz.
Train'den farklı işaretçiler seçilen 10 videoluk taramada da en uzun el kaybı 5 kareye ulaştı (`okul`, `teşekkür`). Bu yüzden yalnızca ortalama tespit oranına bakmak yeterli değil; zor videoların panoları ve eksik seri uzunluğu birlikte incelenmeli.

**Eşleştirilmiş kalite pilotu (19 Ekim 2026):** `configs/experiment/autsl_10_preprocessing_pilot.json` içindeki aynı 10 train + 10 val videosu üç ayrı önbellekte incelendi: mevcut 16 kare, 16 kare + en çok 2 karelik el noktası onarımı, 36 kare. Resmî test kullanılmadı. 36 kare çıkarımı 20/20 videoda tamamlandı, okuma hatası yok. Karşılaştırma: `outputs/preprocessing_pilot_comparison.json`.

- 16 karede 320 örnek karede poz bulunma %100; sol/sağ el bulunma %79,37/%84,38, iki taraf ortalaması %81,87.
- 36 karede 720 örnek karede poz bulunma %100; sol/sağ el bulunma %80,00/%84,17, iki taraf ortalaması %82,08. Bu fark küçük ve örnekleme farklı olduğundan tespit kalitesinde belirgin kazanım kanıtı değildir.
- 36 karelik 20 video cache'i 51,5 MiB; aynı ortalama ile 1.444 train+val videosu yaklaşık 3,6 GiB, test de eklenirse 1.611 video yaklaşık 4,1 GiB olabilir. Bu yalnızca pilot dosya boyutundan ölçekleme tahminidir.
- 16 kare onarımında gözlenen tespit maskesi değiştirilmedi. Kısa ve güvenilir iki uçla çevrili 45 sol + 21 sağ el karesi doğrusal tahmin edildi. Tahmin edilen noktaların hiçbiri görüntü dışında değil; en uzak tahmini bileğin poz bileğine mesafesi normalize koordinatta 0,0569. Bu geometrik kontrol, parmak şeklinin gerçek olduğu anlamına gelmez.
- Zor `kardeş` val videosunda 16 karede sol/sağ 6/16, 36 karede 12/36 ve 15/36 bulundu; 36 karedeki en uzun boşluklar sol 21, sağ 14 kare. Kısa boşluk onarımı bunu çözmez. El kırpımları bu süre boyunca eski görüntüyü tekrarlıyor. `öğretmen` val videosunda dinlenen elin uzun kaybı da sürüyor.
- Bazı `aile` karelerinde el ve poz bileklerinin yakınlık testi uyuşmuyor; iki el gövde önünde çaprazlandığından bu yalnızca **şüpheli kimlik/poz eşleşmesi** işaretidir, tek başına kesin el etiketi hatası değildir. El kırpımı ile poz iskeleti birlikte görsel incelenmeli.

**Pilot kararı:** Kısa boşluk onarımı güvenli sınırlarla çalışan bir model adayıdır; 36 kare hareketin daha sık örneklenmesini sağlar ama el kaybını çözmez. Her iki aday da sadece 20 video içerdiği için bu cache'lerden val doğruluğu kıyaslamak anlamsızdır. Sonraki kontrollü eğitim için aynı 1.245 train + 199 val videosunun her üç cache'i tamamlanmalı: temel 16, bundan türetilen kısa boşluk onarımı ve ayrı 36. Test cache'i eğitim kararları için gerekmiyor. Önce tam cache'in kalite dağılımı ve depolama süresi yeniden ölçülmeli. 36 kare seçilirse `max_seq_len=36` içeren `configs/experiment/autsl_10_36.yaml` kullanılacak; onarım için `configs/experiment/autsl_10_gap2.yaml` ayrı çıktı yazar.

```powershell
# Mevcut 16 kare cache'i değiştirmeden aynı 20 videonun kısa boşluk pilotu:
python scripts/repair_hand_gaps.py --video_ids_file configs/experiment/autsl_10_preprocessing_pilot.json
# Aynı videoların 36 kare MediaPipe pilotu:
python scripts/preprocess_tid.py --config configs/experiment/autsl_10_preprocess_36.yaml --splits train val --video_ids_file configs/experiment/autsl_10_preprocessing_pilot.json
python scripts/compare_preprocessing_pilots.py
```

`sample_frames` artık okunamayan kare yerine yapay siyah kare eklemiyor; videoyu hata olarak bildiriyor. Bu kontrol, tam cache oluşturulmadan önce eklendi. Ayrı cache manifestleri 16 ve 36 karelerin yanlışlıkla karışmasını engeller. Onarımın `*_hand_imputed` maskesi tahmini noktaları gerçek MediaPipe tespitlerinden ayırır. Kırpım görüntüleri bu pilotta değiştirilmedi; özellikle uzun kayıplar için poz bileği merkezli güncel kare kırpımı ileride ayrı bir deney olmalı.

**İkinci kalite turu — Tam 10 sınıf 4 önbelleğin kesin sonuçları (19 Ekim 2026):**
İlk turdaki pilot bulguları üzerine 1.245 train ve 199 val videosunun tamamı (1.444 video) dört ayrı önbellekte tamamlandı ve denetlendi:
1. **Temel 16 kare (`cache/autsl_10_features`):** Train'de iki el ortalama tespit %78,31 (sol %78,13 / sağ %78,49); Val'de %84,61 (sol %82,69 / sağ %86,53). Bilek uyumu Train %95,5–%97,2; Val %96,0–%97,5. Tespit edilemeyen karelerde eski kırpımın donarak taşınma (stale) oranı Train'de sol %20,52, sağ %18,95; Val'de sol %16,27, sağ %11,75'tir. Süre: 19,6 dakika.
2. **Kısa boşluk onarımlı 16 kare (`cache/autsl_10_features_gap2`):** Sadece güvenilir uçlar arasındaki <=2 karelik boşluklar onarıldı. Train'de 1.914 sol + 1.470 sağ (3.384 kare), Val'de 303 sol + 218 sağ (521 kare) başarıyla dolduruldu. Onarılmayan uzun eksiklik oranı Val'de sol %7,8, sağ %6,6'ya indi.
3. **Bilek merkezli canlı yedek kırpım (`cache/autsl_10_features_wristcrop`):** İlk turun tespit ettiği asıl açık sorun (donuk kare tekrarı) çözüldü. Elin kaybolduğu ama poz bileğinin görüldüğü anlarda, donmuş eski kırpım yerine güncel kareden poz bileği merkezli ve video içi medyan el ölçeğinde kareler kesildi (`src/preprocessing/wrist_fallback_crop.py`). Train'de 8.639 kare, Val'de 980 kare canlı kırpımla değiştirildi; donuk (stale) kare oranı %0,0'a indi. Hizalama denetiminde 1.444 videonun tamamında tespit edilen kırpımlarla 0,0 piksel hata sağlandı.
4. **Temel 36 kare (`cache/autsl_10_features_36`):** 1.444 videonun tamamında çıkarıldı (44.820 train + 7.164 val karesi, 56,8 dakika). İki el ortalama tespit oranı Train'de %77,78 (sol %77,52 / sağ %78,04); Val'de %84,07 (sol %81,84 / sağ %86,31). 16 kareye göre tespit oranı Train'de -%0,53, Val'de -%0,54 daha düşüktür; çünkü sık örnekleme hızlı hareketlerdeki hareket bulanıklığı (motion blur) karelerine denk gelerek tespiti zorlaştırmaktadır. Donuk (stale) kırpım oranı Train'de sol %21,51, sağ %20,12'dir. Boyut ~3,6 GiB'dir (226 sınıfta ~90 GiB olur).
5. **Zor işaretçi ve sınıflar:** Val setindeki en yüksek kayıplar `signer16` (ör. `signer16_sample393_color` sağ el 0/16 tespit) ve `signer35` (ör. `signer35_sample539_color` kardeş, yüz önü kayıpları) üzerinde yoğunlaşıyor. Sınıf bazında en düşük tespit %65,9 sağ el ile Sınıf 5 (`doktor`) videosundadır.
Rapor: `outputs/full_cache_quality_all4.json`, görsel panolar: `outputs/preprocessing_wristcrop_qa/`.

Literatürden öncelikli, tek değişkenli geliştirme adayları:

1. **Örnekleme ve etkin bölüm.** AUTSL temel makalesi bazı videolarda başlangıç/son nötr karelerin yanıltıcı olabildiğini gösteriyor. Önce 16 ile 36'nın hareketi kapsama düzeyini görsel karşılaştır; gerekirse poz/el hareketinden etkin aralık bulup baş/sonu sınırlı kırpma varyantı üret. Elin durduğu anlamlı işaretleri yanlışlıkla kesmemek için tam klibi koruyan karşılaştırma şart.
2. **Eksik ve hatalı el noktaları.** 2024 MediaPipe ön işleme çalışması, kısa eksik serilerde çevre karelerden interpolasyon ve el için ayrı dayanak normalizasyonu deniyor. Bizim cache'de eksik nokta sıfır, görüntü kırpımı ise son görülen kareden taşınıyor; bu iki akış zaman zaman uyuşmuyor. Önce eksik serileri ve görselleri ölç; sonra kısa boşluk interpolasyonu + tespit maskesi varyantını karşılaştır. Uzun boşluklara veya gerçekten görünmeyen ele sahte parmak şekli üretme.
3. **El kırpımı.** Mevcut kırpım sadece tespit edilen 21 noktadan ve %20 boşluktan hesaplanıyor. Parmağın kesildiği veya iki elin karıştığı örnekleri say. Sorun yaygınsa poz bileğinden yedek kırpım ve takipli/kareler arası yumuşatılmış kutu varyantı dene; rastgele birkaç iyi örneğe dayanarak tüm veri hattını değiştirme.
4. **Göreli koordinatlar.** Şimdiki `LandmarkEncoder` tüm 75 noktayı tek merkez ve tek ölçekle normalize ediyor. Literatürde üst gövde ve eller için ayrı dayanaklar kullanılıyor; el şekli ile konumun dengesi değişebilir. Cache'i yeniden çıkarmadan model girişinde ayrı el normalizasyonu ve bilek konumu bilgisini koruyan varyant sınanabilir.
5. **MediaPipe çalışma modu.** Kod `IMAGE` modunda seyrek seçilen kareleri birbirinden bağımsız işler. Resmî MediaPipe `VIDEO` modu zamansal takip yapar; ancak atlanan kareler ve zaman damgalarıyla ayrı pilotta ölçülmeli. Tespit oranı artışı tek başına doğruluk kanıtı değildir.

Bu kalite kapısında her varyant için tam model eğitimi gerekmez: görsel pano, tespit maskesi/eksik seri, bilek uyumu, kırpım taşması ve karelerin kapsadığı hareket önce eler. Sağlam kalan değişiklikler aynı 10 sınıf, aynı resmî val, aynı model ve seed ile kısa karşılaştırma eğitimine girer. Nihai karar tam eğitim ve gerekirse tekrar seed'lerle verilir. 2024 AUTSL çalışmasında tüm ön işleme birleşimi 90,40'tan 91,15'e çıkmıştı; kazanımın büyüklüğü bu projeye doğrudan taşınamaz.

SAM-SLR'deki 98,42% RGB test sonucu birden çok akışın birleşiminden ve resmî train+val ile son ince ayardan gelir; tek başına kare seçiminin veya MediaPipe ön işlemesinin etkisi değildir. Aynı makalede RGB/akış kolu eğitimde 32 **ardışık** kare, değerlendirmede beş klip kullanırken iskelet özelliği kolu 60 **eşit aralıklı** kare kullanır. Dolayısıyla makaledeki “32 kare”, bizim videoya yayılan 36 kare seçimiyle aynı deney değildir.

Kaynaklar: [AUTSL temel makalesi](https://arxiv.org/abs/2008.00932), [SAM-SLR](https://arxiv.org/pdf/2103.08833), [MediaPipe nokta onarma ve dayanak normalizasyonu](https://aclanthology.org/2024.signlang-1.36.pdf), [MediaPipe Hand Landmarker belgesi](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker/python).

## Aşama 0 — Tam AUTSL veri yolunu açma

**Durum: Veri yolu doğrulandı; 10 sınıf özellik pilotu çalıştı.** Resmi başlıksız `*_labels.csv` dosyaları doğrudan okunuyor; 226 sınıfta etiket 0–225 korunuyor ve RGB dosyaları doğru bölümden bulunuyor. `preprocess_tid.py` `train`, `val`, `test` işleyebiliyor; bölüm bazlı cache ve yapılandırma manifesti yazıyor. Tam AUTSL eğitim yükleyicisi eksik cache gördüğünde duruyor. `audit_autsl.py` etiket/video sayısı, sınıf kapsamı ve işaretçi ayrıklığını denetliyor. Eski 5-fold CV betiği kaldırıldı.

**Doğrulama:** `outputs/autsl_audit.json` için 28.142 train, 4.418 val, 3.742 test, her birinde 226 sınıf, sıfır eksik RGB ve sıfır split çakışması. Önceki üç videoluk `.npz` yanlış el eşlemesi nedeniyle kaldırıldı; düzeltilmiş 10 sınıf pilotu üretildi. Tam 36.302 video henüz işlenmedi; model henüz eğitilmedi.

## Aşama 1 — Ön işleme pilotu ve veri kalitesi

**10 sınıf temel çizgisi değerlendirildikten sonraki iş.** Colab oturumunda kullanılabilir GPU, RAM, yerel disk ve Drive boşluğunu kaydet. Önce bu projedeki güncel `src/`, `scripts/` ve `configs/` klasörlerini Colab'ın kullandığı Drive proje klasörüne aktar; eski `colab_egitim.ipynb` 10 sınıflı CV içindir ve bu akışı çalıştırmaz. Resmi AUTSL kök klasörünü de Colab'a erişilebilir bir yere yerleştir. `AUTSL_DATA_DIR` değişkeni bu klasörü göstermeli; içinde `train/`, `val/`, `test/`, üç `*_labels.csv` ve `SignList_ClassId_TR_EN.csv` bulunmalı. Veri aktarımı ve depolama durumu görülmeden tam ön işleme başlatma.

İlk pilotu her bölümden 20 video ile çalıştır:

```bash
export AUTSL_DATA_DIR=/content/drive/MyDrive/BitirmeProjesi/AUTSL_Dataset
python scripts/audit_autsl.py --data_dir "$AUTSL_DATA_DIR" --output outputs/autsl_audit.json
python scripts/preprocess_tid.py --config configs/experiment/autsl_226_mobilenet_frozen.yaml --data_dir "$AUTSL_DATA_DIR" --splits train val test --pilot_size 20 --no_drive_backup
python scripts/audit_autsl.py --data_dir "$AUTSL_DATA_DIR" --cache_dir cache/autsl_226_features --inspect_features --output outputs/autsl_pilot_audit.json
```

`--pilot_size`, her bölümde sınıf dengeli ve sınıf içinde yayılmış örnekleri seçer. Pilot raporuna ayrıca gerçek toplam süre, `.npz` toplam boyutu, başarısız video sayısı, MediaPipe poz/sol el/sağ el bulunma oranı ve görsel örnekler eklenecek. Önceki üç dosyalık boyut tahmini yaklaşık **43 GB** idi; gerçek toplam, video içeriğine ve sıkıştırmaya göre değişir. Bu yüzden Drive kapasitesi ile ön işleme süresini 226 sınıf pilotunda yeniden ölçeceğiz.

**Bana gönderilecekler:** `autsl_audit.json`, `autsl_pilot_audit.json`, 20'şer video için süre ve toplam cache boyutu, Colab GPU adı/RAM/boş disk. Bunlarla tam ön işleme yeri ve yöntemi kararlaştırılacak.

**Geçiş ölçütü:** Etiket/video denetimi hatasız; pilotta üç bölümün de cache üretmesi; manifestin `num_frames=16`, `num_classes=226`, `hand_image_size=[160,160]` olması; depolama tahmininin karşılanması. MediaPipe başarısızlıkları sayılmalı ve yoğunlaşmış sınıf/işaretçi varsa önce incelenmeli.

## Aşama 2 — Tam özellik çıkarımı ve kapsam denetimi

Pilot kabul edilirse aynı config ile 36.302 videonun özellikleri çıkarılacak. Kesintiden devam için var olan cache atlanabilir. `--overwrite` yalnızca bir ön işleme hatası düzeltildiğinde kullanılmalı; kare sayısı veya kırpım boyutu değişirse **yeni cache klasörü** açılmalı. Tam koşu sonrası:

```bash
python scripts/preprocess_tid.py --config configs/experiment/autsl_226_mobilenet_frozen.yaml --data_dir "$AUTSL_DATA_DIR" --splits train val test
python scripts/audit_autsl.py --data_dir "$AUTSL_DATA_DIR" --cache_dir cache/autsl_226_features --require_cache --output outputs/autsl_full_audit.json
```

Colab'da varsayılan ön işleme cache'i `/content/BitirmeProjesi/cache/autsl_226_features` altında tutulup işlem sonunda Drive'a yedeklenir. Denetimde `--cache_dir` gerçek kullanılan yolu göstermeli. Yerel disk oturum bitince silinebilir; yedek tamamlanmadan oturum kapatılmamalı.

**Bana gönderilecekler:** `autsl_full_audit.json`, ön işleme günlüğündeki `processed/skipped/failed` sayıları, cache manifesti, toplam boyut ve kalite özeti. Eksik cache varsa model eğitimi başlamaz.

## Aşama 3 — İlk 226 sınıf temel çizgisi

İlk eğitim, mevcut 16 karelik dondurulmuş MobileNetV3 hibrit mimarisini resmi 226 sınıfa taşır. Amaç en yüksek sonuç değil; yeni veri hattını ve ölçümü doğrulamaktır. Kullanılacak dosya `configs/experiment/autsl_226_mobilenet_frozen.yaml`.

```bash
python scripts/train.py --config configs/experiment/autsl_226_mobilenet_frozen.yaml --cache_dir /content/drive/MyDrive/BitirmeProjesi/cache/autsl_226_features
python scripts/evaluate.py --config configs/experiment/autsl_226_mobilenet_frozen.yaml --checkpoint /content/drive/MyDrive/BitirmeProjesi/outputs/autsl_226_mobilenet_frozen/checkpoints/best_model.pt --split val --cache_dir /content/drive/MyDrive/BitirmeProjesi/cache/autsl_226_features --output outputs/autsl_226_mobilenet_val.json
```

Cache yerel Colab diskinde ise aynı komutlarda `--cache_dir` orayı göstermeli. Eğitimden önce tek batch ileri geçiş, sınıf sayısı 226, son katman boyutu 226 ve kayıp değerinin sonlu olduğu kontrol edilecek. `best_model.pt` yalnızca val ile seçilecek. İlk pilot için kısa epoch/az örnek denemesi yapılabilir; nihai temel çizgi sayılmayacak.

**Bana gönderilecekler:** kullanılan config, cache manifesti, train/val eğrisi veya `history.csv`, en iyi epoch, en iyi val Top-1 ve val kaybı, `autsl_226_mobilenet_val.json`, eğitim süresi/GPU. Sonuca göre bir sonraki deneyi seçeceğiz.

## Aşama 4 — Tek değişkenli iyileştirmeler

Kararları val sonuçlarıyla sırasıyla alacağız:

1. **Omurgayı açma.** Aynı 16 kare, aynı veri ve aynı modelde MobileNetV3-Small omurgasını aşamalı fine tuning ile karşılaştır. Öğrenme oranını omurga için daha düşük tut; train/val açığına ve süreye bak.
2. **Zaman çözünürlüğü.** 16 ve 32 kareyi yeni cache'lerle karşılaştır. 64 kareyi ancak 32'nin faydası ve GPU belleği izin verirse dene. Kare sayısı değişince eski cache'i kullanma.
3. **Görsel bağlam.** El kırpıntılarına ek olarak üst beden RGB video kolu dene. Önerilen araştırma temeli, ön eğitimli kısa video omurgasıdır; gerçek Colab GPU'suna göre R(2+1)D-18 veya daha hafif alternatif seçilecek.
4. **İskelet kolu.** Poz ve iki el için normalize edilmiş joint, bone ve motion girdilerini ayrı bir iskelet modelinde dene. MediaPipe bulunma oranı düşükse önce çıkarımı iyileştir.
5. **Füzyon.** RGB ve iskelet modellerinin val hataları birbirini tamamlıyorsa, val üzerinde seçilen ağırlıklarla geç füzyon uygula. İki zayıf modelin birleşimini sırf ensemble olduğu için tutma.

Her karşılaştırmada aynı resmi val seti, aynı sınıf listesi, Top-1, Top-5, macro-F1, işaretçi başına doğruluk, en çok karışan sınıf çiftleri, eğitim süresi ve bellek raporlanacak. Yararlı değişiklikler üç seed ile tekrar edilecek; küçük farklar tek koşuya dayanarak kabul edilmeyecek. İlk makul ara hedef resmi val üzerinde güçlü tek model ve istikrarlı genellemedir; %95+ yalnızca bu kanıt biriktiğinde hedeflenir.

## Aşama 5 — Kilitli test ve mobil uygulama

Mimari, ön işleme, epoch seçimi ve füzyon ağırlığı val üzerinde sabitlendikten sonra resmi test **bir kez** çalıştırılacak. Tek model ve varsa ensemble için Top-1, Top-5, macro-F1, 226 sınıf ve 6 test işaretçisi bazında sonuç kaydedilecek. Test sonucuna göre model ayarı yapılırsa o sonuç artık nihai tarafsız test olarak sunulamaz; yeni bir bağımsız değerlendirme gerekir.

Mobil uygulama için ayrı bir ölçüm yapılacak: gerçek kamera videolarında gecikme, bellek, çevrimdışı/çevrimiçi çalıştırma ve farklı kullanıcılarla başarı. AUTSL test doğruluğu bu uygulama ölçümünün yerine geçmez.

## Dosyalar

- Araştırma raporu: `chatgpt raporu.md`
- İlk 226 sınıf config: `configs/experiment/autsl_226_mobilenet_frozen.yaml`
- Veri denetimi: `scripts/audit_autsl.py`
- Ön işleme: `scripts/preprocess_tid.py`
- Eğitim: `scripts/train.py`
- Val/test değerlendirme: `scripts/evaluate.py`
- Mevcut yerel denetim sonucu: `outputs/autsl_audit.json`

**Şimdiki ortak karar noktası:** Aşama 1 pilotunun günlük ve boyut sonuçlarını paylaş; tam ön işlemenin Colab yerel diskinde mi, Drive üzerinde mi, yoksa parçalara bölünerek mi yapılacağını bu ölçümlere göre seçeceğiz.
