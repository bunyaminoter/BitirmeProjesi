# AUTSL model doğruluğu için proje inceleme ve deney raporu

## 22 Ekim 2026: motion2 eğitim sonucu ve kesin ön işleme seçimi

Colab'dan Drive'a gelen yeni `autsl_10_zero_hand_roi_motion2_seed7` checkpoint'i aynı 199 resmî 10 sınıf val videosunda yerelde tekrar değerlendirildi. **Motion2 176/199 = %88,44; strict v1 192/199 = %96,48**; net −16 doğru video ve −8,04 yüzde puan. Motion2'nin doğrudan değiştirdiği 9 val videosunun 2'si yeni hata, geri kalan 190 videoda net 14 hata artışı var. Aynı 20 epoch eğitim bütçesinde hareket konumlarını geri eklemek model doğruluğunu yükseltmedi. **Karar:** motion2'yi 100 sınıfa genişletme; `centerrepair_strict` v1'i sabitle. Drive'daki 100 sınıf strict cache bugün 12.405 train + 1.963 val ve sıfır bayt eksik olarak yeniden sayıldı. Sıradaki tek tam eğitim, bu cache ile 100 sınıf **pose-only** baz çizgisi; ikinci 100 sınıf koşusu ilk modelin hata analizine bağlı. Resmî test kapalı. [Checkpoint, tahmin ve yol haritası ayrıntıları](docs/AUTSL_10_motion2_egitim_karari_2026-10-22.md). Aşağıdaki motion2 hazırlık bölümü eğitim öncesi tarihsel kayıttır.

## 22 Ekim 2026: seçilen ve Colab'a hazırlanan ön işleme özelliği

Karar verildi: iki elin de kayıp olduğu karelerde **üst gövde ROI el tespiti**, fakat yalnız en az **iki bitişik örnek karede** devam eden yeni tespitler (`motion2`). Bu dar sürüm, 40 zor train videosunda 67 etkin el konumu karesi ekledi ve 8 sağlam kontrolü değiştirmedi. Tam 10 sınıfta 1.245 train + 199 val ayrı cache üretildi; 269 train/26 val kare ve 369/42 el yuvası eklendi, kaynakta görülen eller aynen korundu. Eski seed 7 checkpoint'i strict ve motion2 val'de **192/199 (%96,48)**; yeni ağırlıklar **henüz eğitilmedi**. Tek kareli geniş ROI sürümü 191/199'a düşürdüğü için seçilmedi. Seçilen 1.444 NPZ, config ve Colab eğitim betiği Drive'a taşındı ve cache SHA-256 ile doğrulandı. Şimdi tek adım, [Colab betiğini](scripts/colab_train_zero_hand_roi_motion2.sh) çalıştırıp yeni modeli aynı resmî val videolarında karşılaştırmaktır. [Ayrıntılı sonuç ve karar](docs/AUTSL_10_sifir_el_roi_motion2_sonuc_2026-10-22.md).

## 22 Ekim 2026: zaman + ROI + kol izi birleştirme kararı

40 seçilmiş train videosunda 194 yoğun VIDEO ve bu yöntemle eşleşmeyen 130 üst gövde ROI bilek adayı görsel olarak incelendi. Etkin el üzerinde görünen yeni konum X=VIDEO ile 120/640 (%18,75), Y=ROI ile 130/640 (%20,31), birlikte 163/640 (%25,47) farklı karede önerildi. Bu **aday erişim oranıdır**, doğrulanmış ön işleme veya model doğruluğu artışı değildir. Üç karelik iz ve iki algılayıcı uzlaşması 71 aday seçti, fakat 5 pasif ve 2 nötr el yine geçti. Kol izinin anatomik taraf ataması mevcut cache ile eşleşen 456 bileğin 74'ünde uyuşmadı. ROI, `signer17_sample52` yüz önü etkin 9 karenin 7'sinde el buldu; tekrar eden iki video toplam 1/18. 640 karede tam anatomik insan etiketi ve yeni 21 noktanın doğruluğu yok. Bu yüzden cache ve model ağırlıkları korunuyor, eğitim henüz açılmadı. [Ayrıntılı karar, koşullu yüzdeler ve sıradaki pilot](docs/AUTSL_100_zamansal_roi_birlesim_karar_2026-10-22.md).

## 22 Ekim 2026: yüz önündeki el ve 40 videoluk kimlik kapısı

Mevcut ön işleme FaceLandmarker'ı çalıştırmıyor; yüz noktasının el noktasıyla veri alanı olarak karışması yok. Yüz önündeki görünür elin kaçması ve sol/sağ kimliğinin değişmesi gerçek problem. 40 seçilmiş train videosunda üst gövde ROI ve bütün kaynak kareleri izleyen VIDEO modu ayrı ayrı denetlendi. VIDEO, 16 örnek karede 576→764 tespitli el yuvasına çıktı fakat 8 sağlam kontrolde 5 kayıp kare yarattı. Katı ekleme filtresinin kabul ettiği 27 adayın kare düzeyi görsel etiketinde yalnız 6'sı aktif el, 4'ü pasif, 15'i nötr, 2'si belirsiz; iki aday sağlam kontrolleri de değiştirecekti. Kural `signer38` yüz önündeki gerçek eli reddediyor; `signer17` yüz önünde kesişen iki eli orta bölümde bulamıyor. Bu yüzden cache değiştirilmedi, 10/100 sınıf modeli eğitilmedi. 640 karelik insan etiketlemesi henüz tamamlanmadı. [Yüz önü ve el kimliği raporu](docs/AUTSL_100_yuz_onu_el_kimligi_pilotu_2026-10-22.md) karar ve sonraki kapıyı kaydeder.

## 22 Ekim 2026: Drive tamamlandı; strict piksel ve aktif el pilotu

Önceki eksik aktarım tamamlandı: base16 ve strict cache'lerin her birinde 12.405 train + 1.963 val dosya adı beklenen resmî 100 sınıf listesiyle birebir eşleşti, sıfır bayt dosya yok. Strict için 28 train/val videosu ham kareden yeniden hesaplandı; 878 el karesinde piksel ve geometri farkı sıfır, beş gerçek merkez onarımı karesi dahil. Sonradan yerelde üretilmiş 6.732 strict NPZ'nin tüm ZIP üyeleri CRC taramasından hatasız geçti. Bu, son Drive kopyasındaki 14.368 dosyanın tamamı için bağımsız CRC yapıldığı anlamına gelmez. Yedi train videosunda üst gövde ROI tanı pilotu, `signer38_sample21_color` videosunda 11 ardışık aktif el adayını geri buldu; diğer ağır örneklerde sonuç karışık. V1 cache değiştirilmedi, 100 sınıf modeli eğitilmedi. Kanıt ve karar: [100 sınıf bağımsız inceleme](docs/AUTSL_100_onisleme_sonuc_analizi_2026-10-22.md).

## 22 Ekim 2026: 100 sınıf ön işleme bağımsız inceleme

Drive Desktop üzerinden base16 12.405 train + 1.963 val NPZ'ye erişildi. Colab kalite/hareket raporu strict için aynı kapsamı bildirse de yerel Drive strict klasöründe yalnız 7.619 train NPZ ve val klasörü görünüyor; eşitleme doğrulanmadan yerel strict tam kabul edilmeyecek. Ham videolarla seçilmiş ağır örnekler incelendi. El kaçırma nedenleri küçük/düşük kontrast el, iki elin çakışması, aktif/pasif el ayrımı, hareket/kadraj ve kısmen yanlış pozitif bilek bayrağı olarak ayrıldı. Poz bileği ROI ve VIDEO modu küçük pilotları bütün cache'e uygulanacak güvenli genel çözüm göstermedi. 100 sınıf modeli eğitilmedi. Tüm sayılar, video ID'leri, panolar ve sonraki kapı: [ayrıntılı analiz](docs/AUTSL_100_onisleme_sonuc_analizi_2026-10-22.md).

## 21 Ekim 2026: 100 sınıf ön işleme hazırlığı, eğitim beklemede

10 sınıf `centerrepair_strict` ön işleme v1 olarak kabul edildi. Önceki 10
ClassId ve yalnız train etiketlerinden sabit seed `20261021` ile seçilen
90 ClassId, 100 sınıflık kalite kümesini oluşturuyor: **12.405 train + 1.963
val** video. Seçim/etiket eşlemesi, 200 videoluk Colab pilotu, iki aşamalı
base16→strict komutları ve ham video ile görsel inceleme yöntemi
[100 sınıf kalite kapısı](docs/AUTSL_100_onisleme_kalite_kapisi_2026-10-21.md)
belgesinde. Tüm NPZ'leri kapsayan hareket anomalisi sıralama betiği eklendi;
10 sınıf strict cache üzerinde 1.245 train + 199 val dosyası eksiksiz ve
geçerli bulundu. Bayraklar gerçek parmak hatası kanıtı değildir; ham RGB ile
doğrulanacak. 100 sınıf cache henüz üretilmedi, model eğitimi yapılmadı.

## 21 Ekim 2026: ön işleme ve veri protokolü tek sayfada

Kullanıcının 5-fold/train+val/test ve ön işleme durumu sorusu üzerine
[güncel tek yol belgesi](docs/MEVCUT_DURUM_VE_TEK_YOL_2026-10-21.md)
hazırlandı. Resmî etiketler 28.142 train (31 işaretçi), 4.418 val (6),
3.742 test (6). Train+val son model eğitimi 32.560 video ve %15,70 ek
video sağlar; SAM-SLR bu protokolü son aşamada kullanmıştır. 5-fold'un
her modeli ise yaklaşık 26.048 video görür, bu standart train'den azdır;
beş tam eğitim ve mobilde ensemble çıkarımı bütçeye uygun değildir.
Test videoları eğitime alınırsa artık resmî test sonucu elde edilemez.

16 kare `centerrepair_strict` v1, 10 sınıfta tam train/val ham kare ve
cache-tahmin eşliğiyle doğrulandı; 100/226 sınıf için henüz üretilmedi.
Merkez onarımı val'de sıfır kareyi değiştirdi; pose-only %99,50, RGB
kırpım onarımının başarısını ölçmez. Yeni tek aktif sıra: 100 sınıf listesini
train bilgisiyle sabitle, v1 train/val cache ve kalite denetimini yap, sonra
yalnız bir 100 sınıf pose-only modeli eğit. Bu oturumda eğitim/cache üretimi
veya test değerlendirmesi yapılmadı.

## 20 Ekim 2026: iki 100 sınıf koşusuyla bütçe kararı

Kullanıcının en fazla iki tam 100 sınıf eğitimi sınırı için
[ayrıntılı karar planı](docs/100_sinif_iki_kosu_butce_plani_2026-10-20.md)
oluşturuldu. Yerel resmî etiket sayıları 28.142 train, 4.418 val, 3.742
testtir. Birincil AUTSL ve SAM-SLR sonuçları ile 2026 seçici uzman makalesi
yeniden incelendi. İlk koşu 16 kare strict pose-only baz çizgisi; ikincisi
ilk koşu hatalarının niteliğine göre seçilecek veya 226 sınıfa saklanacak.
Öncesinde sınıf/manifest dondurma, train/val cache kalite denetimi,
pose-only gereksiz RGB veri yolunu kaldırma ve tek batch maliyet ölçümü var.
Bu oturumda yeni model eğitimi yapılmadı.

## 20 Ekim 2026: Gemini kaskad önerisinin bağımsız denetimi

Kullanıcının mimari kararı için [ayrıntılı fizibilite raporu](docs/secici_uzman_kaskad_bagimsiz_degerlendirme_2026-10-20.md)
hazırlandı. İki aşamalı yönlendirme teknik olarak mümkün, ancak pose 20 ms,
ConvNeXt 400–800 ms, %10–15 RGB çağrısı ve otomatik doğruluk kazancı bu proje
için ölçülmüş değil. Kodda pose-only yolunun dahi MediaPipe, el crop ve RGB
tensörü hazırladığı; uygulamanın telefonda yerleşik model değil FastAPI sunucusu
olduğu doğrulandı. 199 videoluk eşleşmiş 10 sınıf val üzerinde %10,05
düşük-margin yönlendirme bir pose hatasını kurtarırken bir doğruyu bozdu;
%15,08'de net bir video kaybettirdi. Bu yalnız eski MobileNet hibritiyle
retrospektif örnek, ConvNeXt sonucu değil. 100 sınıf pose-only baz ve eşleşmiş
RGB tahminleri, sonra tam 226/cihaz doğrulaması karar kapısı olarak yol
haritasına işlendi. Eğitim veya cache üretimi yapılmadı.

## 20 Ekim 2026: 226 sınıf için güçlü ve seçici RGB yönü

Kullanıcının önceliği 226 sınıf. 10 sınıftaki pose-only %99,50, yakın 226
işarette görüntü ayrıntısının gereksiz olduğunu göstermez. Mevcut hibritin
RGB kolu dondurulmuş MobileNetV3-Small ve erken cross-attention kullanıyordu.
AUTSL literatürü el görünümü, poz ve hareketin tamamlayıcı olabileceğini
gösteriyor. Projeye ConvNeXt-Tiny tabanlı 226 sınıf RGB+poz araştırma config'i,
EfficientNetV2-S omurga seçeneği ve ConvNeXt'in ön eğitimli son normalizasyonu
eklendi. Bunlar eğitilmedi; tam 226 strict cache henüz yok.

Güncel sıra: yakın sınıfları içeren 100 sınıf stres testi, ardından tam 226
pose-only ve güçlü RGB+poz adayının aynı resmî val'de eşleştirilmesi. RGB
gerçekten hata kurtarırsa yalnız zor örneklerde çalışan iki aşamalı sistem
sınanacak; güvenli görünen ama yanlış poz tahminlerinin atlanması özellikle
ölçülecek. 10 sınıf seed 42 tekrarı korunuyor ancak zorunlu sıradaki adım
değil. Tam araştırma ve karar kapıları [RGB uzmanı raporunda](docs/rgb_226_secici_uzman_arastirmasi_2026-10-20.md),
[güncel sırada](docs/AUTSL_226_yol_haritasi.md).

## 20 Ekim 2026: pose-only sonucu ve yeni yol

Masaüstündeki `chatgptye_gonderilecekler_poseonly` paketinin beş dosyası
incelendi ve `outputs/colab_poseonly_seed7_review/` içine kopyalandı. Seed 7,
aynı strict cache ve resmî 199 videoluk 10 sınıf val üzerinde pose-only
**198/199 = %99,50**, Macro-F1 **0,99500**; hibrit **192/199 = %96,48**.
Pose-only hibritin yedi hatasını düzeltti, bir yeni hata yaptı. İşaretçi 11
26/30'dan 30/30'a çıktı. 17. epoch en iyi; 20. epoch 196/199. Train/val
işaretçileri ayrık, ID ve etiketler doğru, iki split arasında birebir aynı
landmark dizisi bulunmadı. “Pose-only” iki elin parmak noktalarını da içeriyor.

Yeni 10 sınıf çalışma adayı pose-only seed 7; eski hibrit checkpoint korunuyor.
Seed 42 pose-only tekrarı isteğe bağlı dayanıklılık deneyi olarak kaldı;
güncel sırada 100 sınıf için ayrı ön işleme ve eğitim var. Test ve gerçek
kamera doğruluğu ölçülmedi. Tam bulgu,
sınırlar ve karar kapıları [pose-only raporunda](docs/poseonly_seed7_sonuclari_2026-10-20.md),
güncel sıra [yol haritasında](docs/AUTSL_226_yol_haritasi.md).

## 20 Ekim 2026: işaretçi 11 ve canlı uygulama çözüm yolu

Kullanıcının kürklü yelek, koyu kıyafet, oturma açısı ve gölge gözlemleri
ham val videolarında görüldü. Seed 7 strict işaretçi 11'de 26/30, diğer
işaretçilerde 166/169 doğru. Ancak el-kare tespit oranı işaretçi 11'de
%84,79, diğerlerinde %84,58; düşük tespit sıklığı **tek başına** hataları
açıklamıyor. Hatalı dört işaretçi 11 videosunda tespit oranı %83,59,
doğru 26 videosunda %84,98. Yedi hatanın ham/landmark ve crop panoları
üretildi. “Okul→öğretmen” iki kez, “kitap” iki farklı yanlış sınıfa gitti.
Sahne görünümü makul hipotez, kanıtlanmış tek neden değil.

Çözüm sırası [işaretçi 11 raporunda](docs/signer11_dayaniklilik_analizi_2026-10-20.md):
seed 7 hibrit korunur; önce aynı cache ve bütçede **pose-only** teşhis modeli
eğitilir. Bu, RGB doku bağımlılığını sınar; sonuç gelmeden iyileşme iddiası
yok. Sonuca göre geç füzyon, train-only görünüm çeşitlendirme veya üst beden
RGB bağlamı tek tek denenir. Canlı sürüm için resmî testin yanı sıra yeni
kişilerde, gerçek kamera/ışıkta kesintisiz işaret videoları ve gecikme
ölçümleri gerekir. İki seed'de strict lehine yön olsa da yöntem üstünlüğü
istatistiksel olarak kesinleşmiş sayılamaz; val'de merkez onarımı sıfır.

## 20 Ekim 2026: ZIP sonuçları ve güncel karar

`C:\Users\Bünyamin\Desktop\chatgptye_gonderilecekler.zip` içindeki 10 dosya
incelendi. Dört tahmin dosyasının her biri seçilmiş 10 sınıfın **199 resmi val
videosunu** eksiksiz kapsıyor; etiket/ID ve hesaplanan Top-1/Macro-F1
doğrulandı. Güncel seed 42 base/strict sonuçları **%87,44/%88,94**; yeni
seed 7 sonuçları **%94,47/%96,48**. Strict her iki seed'de **+3/+4 doğru
video** kazandı; ancak seed etkisi 14–15 video. İşaretçi 11 strict ile her
iki seed'de geriliyor. İki seed strict'i 10 sınıf geliştirme baz çizgisi
seçmek için yeterli; kesin genellenebilir üstünlük için yeterli değil. Önce
seed 7 strict'in yedi val hatasını inceliyoruz; seed 2026 eşleştirilmiş
çifti bu çalışmaya başlamanın zorunlu şartı değil, sonraki dayanıklılık
kontrolüdür. Seed sayısı sıralı bir kalite ayarı değildir; 7'nin neden iyi
geldiği başlangıç, eğitim sırası ve augmentasyon rastgeleliğinden
ayrıştırılmamış. %96,48, 226 sınıf veya resmi test başarısı sayılmaz.

Kaynak, tam eşleştirilmiş sayımlar, hata örnekleri, sınırlamalar ve karar
kapıları [seed 7 ayrıntılı raporunda](docs/colab_seed7_sonuclari_2026-10-20.md),
uygulanacak komutlar [Colab rehberinde](docs/colab_onisleme_karsilastirmasi.md),
deney sırası [yol haritasında](docs/AUTSL_226_yol_haritasi.md). Resmi test
kullanılmadı; bu oturumda eğitim veya buluta aktarım yapılmadı.

## 20 Ekim 2026: altı Drive çıktısıyla önceki seed 42 kararı

Yeni val JSON'ları ve iki eğitim geçmişi/grafiği incelendi. Düzeltilmiş
değerlendirmede temel model **%87,44 (174/199), macro-F1 0,87185**;
`centerrepair_strict` adayı **%88,94 (177/199), macro-F1 0,88852**. Fark
3 video = 1,51 yüzde puan. İki model de en yüksek val doğruluğu ve en düşük
val kaybına 20. epoch'ta ulaştı. Tek seed ile karar verilmeyecek; seed 7
çifti ve video bazlı val tahminleri sıradaki adım. Dört tekrar config'i
hazırlandı; resmî test kullanılmadı. Dosya eşlemesi, mimari/parametre
incelemesi, sınırlar ve Colab komutları
[ayrıntılı karşılaştırma rehberinde](docs/colab_onisleme_karsilastirmasi.md).

## 20 Ekim 2026: ilk Colab eğitim sonucu

Bu bölüm ilk AMP'siz logların tarihsel kaydıdır. Güncel rakamlar üstteki
Drive çıktısı incelemesindedir.

Resmî 10 sınıflı val setindeki 199 videoda, seed 42 ile 20 epoch eğitilen
temel 16 kare modelinin ayrı değerlendirmesi **%86,93 (173/199)**,
`centerrepair_strict` adayının **%89,45 (178/199)** oldu. Fark 5 video,
yaklaşık 2,51 yüzde puan. Bu tek koşu ön işleme adayını destekler; kesin seçim
için iki ek seed gerekir. Notebook'taki 0,0000 Macro F1 tablo anahtarı hatası;
eski logdaki F1 formülü de düzeltilip mevcut checkpoint'ler yeniden val üzerinde
ölçülecek. Resmî test kullanılmadı. Sonuç, düzeltmeler ve sonraki komutlar
[Colab karşılaştırma rehberinde](docs/colab_onisleme_karsilastirmasi.md);
güncel sıra [yol haritasında](docs/AUTSL_226_yol_haritasi.md).

**Tarih:** 19 Ekim 2026  
**Kapsam:** Proje kodu, `BitirmeProjesi/outputs` deney çıktıları, masaüstündeki `AUTSL_Dataset`, önceki `literatür araştırması.docx` ve doğrulanabilir AUTSL yayınları. Bu rapor yeni bir model eğitiminin sonucu değildir.

## Güncel ön işleme durumu — 19 Ekim 2026

Merkez onarımı ve tam 10 sınıf train/val ön işlemesi tamamlandı. Son cache
`cache/autsl_10_features_centerrepair_strict`: 1.245 train + 199 val, 16 kare.
Bütün ham videolar/kırpımlar yeniden doğrulandı; son beş merkez değişikliği görsel
incelendi. İlk tam denemedeki iki gerileme daha dar müdahale kuralıyla önlendi.
63/63 test geçti. 19 Ekim itibarıyla model eğitimi henüz yapılmamıştı;
20 Ekim Colab sonuçları ve ölçüm düzeltmeleri yukarıda kayıtlıdır.

- [Nihai ön işleme raporu](docs/merkez_onarimi_tam_onisleme_raporu.md)
- [Colab aktarım, kontrol ve eğitim rehberi](docs/colab_onisleme_karsilastirmasi.md)
- [Yapılan bütün işlemlerin günlüğü](docs/preprocessing_calisma_gunlugu.md)
- [Antigravity devir notu](docs/antigravity_preprocessing_devir.md)
- [Güncel yol haritası](docs/AUTSL_226_yol_haritasi.md)

Aşağıdaki proje/literatür incelemesi başlangıçtaki kod durumunu anlatır; güncel
veri yolu ve ön işleme düzeltmeleri yukarıdaki belgelerden izlenmelidir.

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

## 22 Ekim 2026 — 100 sınıf yinelenen el kaybı devam raporu

100 sınıf cache'inde ağır iki el kaybı 46 train + 33 val videoda görülüyor; train'deki 46'nın 39'u signer28/13/17 üzerinde. Bu otomatik bayrakların aktif el kaybı anlamına gelmediği görsel olarak teyit edildi. AUTSL literatüründeki dirsek–bilek/omuz ölçekli ROI, 7 train klipte 132 eksik yuva için 12 geometrik kabul üretse de bazıları pasif ele oturduğu için reddedildi. Üst gövde ikinci tespiti 10 ek train klipte 104 ham yeni aday üretti; signer28/24 örneklerinde gerçek elleri tekrar buldu, signer17/örtüşen ellerde boşlukları genel olarak çözemedi. Ardından 19 işaretçi/26 sınıftan sabitlenen 40 train videosunun tüm panoları üretildi: temel 576, üst gövde ROI 712 el tespitli yuva ve 256 ham yeni aday; iyi kontrollerde ROI'yi yerine koymak 245→230 düşüşe neden oldu. Bu sayılar doğru tarafı doğrulanmış kurtarma değildir; görsel etiket sütunları henüz boş. Hiçbir üretim cache'i veya model değiştirilmedi. Karar: bu kuyrukta aktif/pasif ve tarafı kare düzeyinde etiketle, yalnız eksik tarafa güvenli ekleme ve regresyon kapısından sonra küçük model karşılaştırmasına geç. Ayrıntılı deneyler, kaynaklar ve panolar: [AUTSL 100 yaygın el kaybı deneyleri](docs/AUTSL_100_yaygin_el_kaybi_deneyleri_2026-10-22.md).
