# AUTSL 226 sınıf model doğruluğu yol haritası

> **29 Ekim ürün mimarisi — sunucu çıkarımı adayı:** 4.418 resmî AUTSL val MP4 dosyasının ortanca boyutu 375 KB; 20 Mbps etkin uplink'te yalnız aktarım alt sınırı yaklaşık 150 ms. Telefonun gerçek codec/ağı, sunucu MediaPipe+iki model süresi ve kuyruk henüz ölçülmediğinden bu uçtan uca gecikme değildir. İlk ürün pilotu için **telefonda sıkıştırılmış klip, sunucuda aynı ön işlem + pose-only + RGB+pose + sabit füzyon** seçildi; daha sonra gerekiyorsa telefonda pose-only çevrimdışı yedek. Mevcut FastAPI prototipi tek model/10 sınıf ve yalnız sunucu içi zaman ölçüyor, dağıtıma hazır değil. 226 cache ve eğitim sırası değişmiyor; önce mevcut 100 ağırlıklarıyla küçük ağ pilotu, sonra 226 sonuçlarıyla gerçek telefon p50/p95, boyut, ısı ve güvenlik kapısı. [Ölçümler, seçenekler ve uygulama denetimi](AUTSL_bulut_mobil_cikarim_mimarisi_2026-10-29.md).

> **29 Ekim Gemini literatür kontrolü:** Geç füzyon fikri 100 sınıfta ölçülmüş %90,88 ile destekleniyor; 1.862/1.963 oracle birleşim uygulanabilir yönlendirici değildir. 174 bozulan vs 180 yerelde incelenebilen kurtarılan videoda kayıp-el kare sayısı AUC 0,528; basit kalite eşiğiyle RGB kapama için güçlü sinyal yok. Mask/dropout yeni eğitim ve kod değişikliği, tam kare/PoseC3D ayrı büyük mimari deneyidir. Mobil gecikme telefon üzerinde henüz ölçülmedi. **226 base16 kalite kapısı ve iki eş koşullu 226 model sırası değişmiyor.** [Birincil literatür, kod tanısı ve mobil artı/eksi raporu](AUTSL_Gemini_literatur_ve_mobil_fizyon_degerlendirmesi_2026-10-29.md).

> **29 Ekim aktif uygulama — 101 ortak hata ve 226 önbellek kapısı:** İki 100 sınıf modelinin ortak 101 hatası incelendi; **71/101 videoda iki elin birlikte kaybolduğu örnek kare yok**, hatalar 38 sınıfa yayılıyor. Genel bir yeni el onarım kuralı kanıtlanmadı; `centerrepair_strict` v1 korunuyor, 100 sınıfta üçüncü tam eğitim açılmıyor. Resmî 226 metadata **28.142 train + 4.418 val + 3.742 test**, 226 sınıf ve işaretçi ayrıklığıyla temiz. **Tam 226 özellik cache'i henüz yok.** 226 base16 küçük çıkarım ve 100→226 yeniden etiketleme pilotları geçti; hazır 14.368 NPZ yeniden kullanılabilir, kalan 18.192 video için 10 adet ≤2.000 video partisi hazır. **Aktif sıra:** Colab'da 226 base16 train/val tamamlama ve kalite kapısı → 226 pose-only → bütçe uygunsa aynı protokolde 226 RGB+pose ve sabit 0,5/0,5 füzyon denetimi. Resmî test model seçimine kapalı. [Kanıt, görsel inceleme, betikler ve tam uygulama sırası](AUTSL_100_ortak_101_hata_ve_226_onbellek_kapisi_2026-10-29.md). Aşağıdaki eski “aktif sıra” paragrafları tarihsel kayıttır.

> **29 Ekim 100 sınıf RGB sonucu ve yeni aktif karar:** A100'de 60 epoch ConvNeXt-Tiny + pose eğitimi bitti. Resmî 100 sınıf val'de hibrit tek başına **1.688/1.963 (%85,99)**; pose-only **1.678/1.963 (%85,48)**. 184 kurtarma/174 bozma nedeniyle tek modelde yalnız **+10 video** ve exact McNemar p≈0,634; açık üstünlük yok. İki modelin **sabit 0,5/0,5 olasılık füzyonu**, yeni eğitim olmadan **1.784/1.963 (%90,88)** ve Macro-F1 **0,9070** verdi; altı val işaretçisinin tümünde artış var. **Karar:** 100 sınıfta yeni tam koşu yapma. Önce iki modelin ortak 101 hatasındaki tekrar eden nedenleri ayır, sonra 226 strict cache kalite kapısı, 226 pose-only baz ve bütçe uygunsa aynı protokolde 226 RGB+pose. Füzyon yalnız iki 226 modelin eşleşmiş tahminleri hazır olduğunda yeniden değerlendirilir; mobil çıkarım maliyeti ayrı kapı. [Tam doğrulama, eğri, işaretçi ve sınıf analizi](AUTSL_100_RGB_pose_egitim_sonucu_2026-10-29.md). Aşağıdaki "tam eğitim henüz başlamadı" ifadeleri bu sonuç öncesi tarihsel kararlardır.

> **28 Ekim Colab batch kararı:** Önceki fiziksel `batch_size: 2` yerel bellek için temkinli ön kontrol ayarıydı; etkin batch zaten `2 × 8 = 16` idi. Seçilen 100 sınıf ConvNeXt koşusunu **A100'de 16 × 1 = 16** ile başlat; A100 yoksa ayrı **T4/L4 8 × 2 = 16** profiliyle ön kontrol yap. AMP ve checkpointing açık; Colab preflight bellek payı ve süreyi doğrulamadan tam eğitimi başlatma. Profil eğitim başladıktan sonra değiştirilmez. Bu kararın doğruluk garantisi yoktur; val eğrisi ve eşleşmiş 1.963 video sonucu karar verir. [Ayarlar, bellek kapısı ve gerekçe](AUTSL_100_RGB_pose_mimari_karari_2026-10-28.md).

> **28 Ekim kesin 100 sınıf RGB araştırma mimarisi:** Kullanıcı gerekli Colab tokenlarını sağlayabileceğini netleştirdi. Kalan ikinci tam 100 sınıf koşusunda maliyet odaklı MobileNetV3-Large taslağı yerine **ön eğitimli ve eğitilebilir ConvNeXt-Tiny + aynı pose kolu** seçildi. Aynı 100 sınıf sırası, 16 kare, iki 160×160 el, strict cache ve seed 7 korunacak; 60 epoch üst sınırı, 12 epoch erken durdurma var. Gerçek cache üzerinde yerel RTX 5050 mikro batch ileri/geri kontrolü geçti (tepe ayrılmış bellek 1,48 GiB); Colab ön kontrolü yine zorunlu. Bu araştırma modelidir; mobil ürün ve 226 sınıf başarısı ayrı ölçülecek. Tam eğitim henüz başlamadı. [Mimari gerekçe, literatür, ölçümler ve karar ölçütü](AUTSL_100_RGB_pose_mimari_karari_2026-10-28.md). Aşağıdaki MobileNetV3-Large önerisi bütçe netleşmeden önceki taslaktır.

**Aktif sıra:** (1) Seçilen ConvNeXt-Tiny config'ini ve strict cache'i Colab'da mikro batch ileri/geri kontrolünden geçir; (2) aynı 100 sınıfta tek tam RGB+pose eğitimini yap; (3) 1.963 val videosunda pose-only'ye göre kurtarılan ve bozulan tahminleri incele; (4) bu sonuca göre 226 sınıf önbellek ve model dizisini seç. Test kapalı.

## Önceki kararların kaydı

> **28 Ekim RGB karar düzeltmesi:** 226 pose-only koşusu, RGB katkısını kesin söyleyemez. Kullanıcı kalan tek tam 100 sınıf koşusunu doğrudan RGB+pose karşılaştırmasına ayırmayı önerdi; bunu, mevcut 100 sınıf strict cache ve aynı resmî train/val kimlikleriyle, **eğitilebilir MobileNetV3-Large + pose** olarak öneriyorum. Önce tek batch GPU maliyet/bellek ve veri yolu kontrolü; Colab süresi/belleği bütçeye uyarsa tek tam eğitim, pahalı çıkarsa koşu başlamadan 226 pose-only planına dön. Tam koşudan sonra 1.963 val videosunda pose-only ile eşleşmiş düzelen/bozulan tahminler. 226 cache/eğitim bundan sonra. El crop'u kullanan bu modelin başarısızlığı bütün RGB yöntemlerini elemez; başarısı da 226 sınıfa otomatik aktarılmaz. [Güncel gerekçe ve sınırlar](AUTSL_100_hata_tanisi_ve_226_karari_2026-10-28.md). Aşağıdaki “100 koşusunu atla” kararı önceki öneridir.

> **28 Ekim yeni karar — 35 video hata tanısı tamamlandı:** 23 tekrarlı yanlış, 12 doğru val kontrolü, 36 train referansı ve 54 karede landmark bindirmesi incelendi. `onlar→kalem`, `şeker→Atatürk`, `okul→beklemek` karışıklıkları için genel bir eksik el/yanlış nokta örüntüsü bulunmadı; `centerrepair_strict` v1 korunuyor. Pose-only yolunda strict'in değiştirdiği görüntü kırpımları okunmuyor; 48 eşlenmiş base16/strict NPZ'nin landmark/etiket/kare alanları birebir aynı. **Karar:** ikinci tam 100 sınıf koşusuna bütçe harcama; resmî train/val için ayrı 226 sınıf base16 landmark cache'i ve kalite kapısı hazırla, sonra tek 226 sınıf pose-only bazını eğit. 100 sınıf NPZ'leri etiket indeksleri farklı olduğundan doğrudan kopyalanamaz. RGB+pose adayını 226 hata dağılımından sonra seç. Test kapalı. [Görsel kanıt ve uygulama sırası](AUTSL_100_hata_tanisi_ve_226_karari_2026-10-28.md). Aşağıdaki eski “35 video incelenecek” ifadesi tarihsel kayıttır.

> **28 Ekim güncel karar — ilk 100 sınıf eğitimi tamamlandı:** `autsl_100_poseonly_seed7`, 16 kare strict v1 ile 35 epoch eğitildi. Resmî 100 sınıf val'de en iyi 29. epoch checkpoint'i **1.678/1.963 Top-1 (%85,48)**, **1.914/1.963 Top-5 (%97,50)** ve **0,8524 Macro-F1** verdi. Tahminler resmî val kimlikleriyle, checkpoint SHA-256 ile ve metrikler yeniden hesaplanarak doğrulandı. 285 hatanın 210'unda iki elin aynı anda kayıp olduğu örnek kare yok; sık üç karışıklık çiftinin 23 videosunda da her karede en az bir el bulunuyor. **Strict v1 korunuyor. Şimdiki tek adım:** tekrarlanan karışıklıkların ham video, landmark yörüngesi ve crop panosuyla küçük, dengeli hata tanısı; sonra en fazla bir ikinci 100 sınıf koşusu veya doğrudan 226 sınıf bazına geçiş kararı. Resmî test kapalı. [Ayrıntılı sonuç ve karar](AUTSL_100_poseonly_seed7_sonuc_2026-10-28.md). Aşağıdaki eski “100 sınıf eğitimi henüz yapılmadı” ifadeleri tarihsel kayıttır.

> **22 Ekim güncel karar — motion2 eğitimi bitti:** Drive checkpoint'i ve 199 val tahmini yerelde yeniden doğrulandı. Aynı seed 7/20 epoch hibritte strict v1 **192/199 (%96,48)**, sıfır el ROI `motion2` **176/199 (%88,44)**; 17 strict-doğru video kaybedildi, yalnız 1 video kazanıldı. Motion2'nin değiştirdiği 9 val videosunun 2'si yeni hata; geri kalan 190 videoda net 14 kayıp var. **Motion2 100 sınıfa taşınmayacak; strict v1 ön işleme olarak sabit.** Drive'da strict 100 sınıf cache'i bugün yeniden **12.405 train + 1.963 val** ve sıfır bayt dosya olarak sayıldı. **Sıradaki pahalı koşu:** sabit 100 sınıfta strict cache + pose-only baz çizgisi. Önce manifest, tek batch maliyeti ve devam edilebilir checkpoint ayarı; sonra eğit, video bazlı val hata analizi yap. İkinci 100 sınıf eğitimi ancak ilk koşunun hatası haklı çıkarırsa. Resmî test kapalı. [Yeni karar ve kanıt](AUTSL_10_motion2_egitim_karari_2026-10-22.md). Aşağıdaki motion2 eğitim talimatları karar öncesi tarihsel kayıttır.

> **22 Ekim uygulanan ön işleme kararı:** Sıfır el karelerinde üst gövde ROI ve **en az iki bitişik örnek kare** koşulunu taşıyan `motion2` ayrı 10 sınıf train/val cache'i üretildi. 40 seçilmiş zor train videosunda 67 etkin el adayı kare korunurken 8 sağlam kontrol birebir aynı kaldı. Resmî 10 sınıfta 269 train + 26 val sıfır elli kareye yeni el verisi eklendi; 1.444 NPZ tamamı kaynakla denetlendi ve Drive'a SHA-256 eşliğiyle aktarıldı. Eski seed 7 checkpoint'i strict ve motion2 val'de **192/199** verdi; **yeni model henüz eğitilmedi**. **Sıradaki tek koşu:** [hazır Colab betiği](../scripts/colab_train_zero_hand_roi_motion2.sh) ile 20 epoch seed 7 hibriti motion2 cache'te sıfırdan eğit ve aynı 199 val video tahminini strict 192/199 ile karşılaştır. 100 sınıf cache/eğitim kararı bu sonuçtan sonra. [Tam sonuç ve neden motion2](AUTSL_10_sifir_el_roi_motion2_sonuc_2026-10-22.md).

> **22 Ekim önceki geniş X/Y pilotu:** Yoğun VIDEO ve üst gövde ROI tüm kayıplara birlikte uygulanınca pasif el ve taraf hatası riski nedeniyle genel birleşim reddedilmişti. Yeni `motion2` kuralı bu geniş birleşim değildir: yalnız iki elin de kayıp olduğu, en az iki bitişik karelik dar durumdaki ROI tekrarını seçer. [Önceki karşı örnekler](AUTSL_100_zamansal_roi_birlesim_karar_2026-10-22.md).

> **22 Ekim yeni karar kapısı — yüz önündeki el:** Mevcut akış yüz landmarkı çıkarmıyor (`enable_face=False`); sorun elin örtülmesi ve sol/sağ kimliği. 40 seçilmiş train videosunda bütün kaynak kareleri izleyen VIDEO pilotu 16 örnek karede tespitli yuvaları 576→764 çıkardı, fakat 8 sağlam kontrolde 5 kayıp kare yarattı. 194 yeni uzamsal adayın katı geometri filtresinden geçen 27'si görsel incelendi: yalnız 6 aktif, 4 pasif, 15 nötr, 2 belirsiz. Filtre `signer38` yüz elini reddediyor; `signer17` çapraz yüz ellerini çözmüyor. **Genel cache birleştirme ve 10 sınıf eğitim kapısı açılmadı.** Önce 640 karelik etiket kuyruğunda etkin orta bölümü tamamlayıp aynı videolarda zaman/kol izi destekli taraf atamasını ayrı pilotta sınayacağız. [Yeni ayrıntılı rapor](AUTSL_100_yuz_onu_el_kimligi_pilotu_2026-10-22.md).

> **22 Ekim önceki kalite kapısı:** 100 sınıfta iki elde ağır kayıp 46 train + 33 val videoda; train'deki 46'nın 39'u signer28/13/17. Dirsek–bilek ROI pilotu pasif/yanlış el riskiyle reddedildi. Üst gövde ROI 40 seçilmiş train videoda bazı aktif elleri buldu; sağlam kontrollerde yerine koyma 245→230 tespit kaybettirdi, sol/sağ kimliği hâlâ açık. 40 videonun panoları hazır. [Deney defteri](AUTSL_100_yaygin_el_kaybi_deneyleri_2026-10-22.md).

> **22 Ekim son doğrulama:** Base16 ve strict Drive cache adları resmî 100 sınıf train/val listesiyle tam eşleşiyor (**12.405 + 1.963**). 28 eşlenmiş videoda 878 el karesinin strict piksel/geometrisi sıfır farkla doğrulandı; yerelde son üretilen 6.732 strict NPZ CRC taramasında hata 0. [Ayrıntılı bağımsız sonuç](AUTSL_100_onisleme_sonuc_analizi_2026-10-22.md). Üst gövde ROI train pilotu bir aktif el kaybını 11 ardışık karede buldu, diğer zor videoları genel olarak onarmadı. **V1 cache korunuyor; sıradaki iş aktif el/sol-sağ güvenliğini sınıf ve işaretçi dengeli kare etiketleriyle ölçmek.** Eğitim hâlâ başlamadı. Aşağıdaki önceki “Drive kopyası eksik” paragrafı aktarım öncesi kayıttır.

> **22 Ekim ön işleme kararı:** [100 sınıf bağımsız kalite incelemesi](AUTSL_100_onisleme_sonuc_analizi_2026-10-22.md) tamamlandı. Base16 Drive kopyası tam, strict Drive Desktop kopyası 7.619 train / val görünmüyor; Colab kalite JSON'u 14.368 dosya diyor. Öncelik strict aktarımı doğrulamak, aktif/pasif el ve çakışma sorunlarını görsel olarak etiketlemek, yalnız gerekirse küçük eşlenmiş train pilotudur. 100 sınıf eğitimi, 5-fold ve test hâlâ beklemede. Aşağıdaki önceki başlıklar tarihsel karar kaydıdır.

**Aktif sıra, 22 Ekim eğitim sonrası:** (1) 16 kare strict v1, 100 sınıf class ID listesi ve train/val cache **sabit**; (2) geniş VIDEO+ROI ve dar motion2 adayları bu bütçeyle **reddedildi**; (3) 100 sınıf pose-only için class/cache manifestini, resmî train/val kapsamını, tek batch bellek/süreyi ve kesinti sonrası devamı doğrula; (4) **yalnız bir** tam 100 sınıf pose-only Colab koşusu yap; 20 epoch'u 10 sınıftan kopyalamadan öğrenme eğrisine uygun bütçe belirle; (5) en iyi checkpoint'in 1.963 val videosundaki tahminlerini, signer/sınıf/confusion ve kötü nokta panolarıyla analiz et; (6) ikinci pahalı 100 sınıf koşusunu yalnız eksik bilgi türü netleşirse seç veya bütçeyi 226 sınıfa sakla. 640 karenin tam anatomik insan etiketi hâlâ yok; tek pilotun 226 sınıf doğruluğuna aktarımı sınırlı. Resmî test kapalı.

> **21 Ekim güncel tek yol:** [Mevcut durum ve veri sınırı](MEVCUT_DURUM_VE_TEK_YOL_2026-10-21.md).
> Ön işleme v1, 10 sınıfta denetlendi; 100/226 sınıfta henüz tamamlanmadı.
> Şimdi [sabitlenen 100 sınıfta](AUTSL_100_onisleme_kalite_kapisi_2026-10-21.md)
> v1 train/val cache + otomatik/görsel kalite kapısı var. Kullanıcı bu aşamada
> **model eğitimi istemiyor**; 100 sınıf koşusu kalite bulgularından sonra
> ayrıca kararlaştırılacak. 5-fold ve train+val final eğitim bugün yapılmıyor.
> Aşağıdaki eski başlıklar tarihsel deney kararlarıdır.

**Başlangıç:** 19 Ekim 2026  
**Amaç:** Resmi AUTSL işaretçiden bağımsız ayrımında, yeniden üretilebilir ve giderek artan Top-1 doğruluk elde etmek. Nihai yüksek doğruluk hedefi %95+; bu eşik garanti edilen sonuç değildir.

## Çalışma kuralı

Resmi `train` ile ağırlık öğrenilir, `val` ile epoch ve deney seçilir. Resmi `test` etiketleri ve sonucu, tasarım kararları tamamlanana kadar model seçmek için kullanılmaz. Eski 10 sınıflı %85,31 deney ayrı bir protokoldür; yeni 226 sınıf tablolarına katılmaz. Her deney tek bir ana değişkeni sınar ve önceki başarılı checkpoint korunur. Kullanılan config, ön işleme manifesti, seed ve veri kapsamı sonuçla birlikte saklanır.

## Güncel öncelik — 226 sınıfa ölçekleme ve akıllı RGB

**100 sınıf eğitim bütçesi (en fazla iki tam koşu):**
[Bütçe duyarlı karar planı](100_sinif_iki_kosu_butce_plani_2026-10-20.md)
uygulanacak. Önce sınıf manifesti, strict train/val cache QA, pose-only veri
yolundaki gereksiz RGB okumasının giderilmesi ve tek batch maliyet ölçümü.
İlk tam koşu bilinen 16 kare pose-only baz çizgisidir. İkinci 100 sınıf
koşusu peşinen ConvNeXt'e ayrılmayacak; ilk koşunun landmark kalitesi,
karışıklıkları ve öğrenme eğrisi görüldükten sonra ya hedefli aday seçilecek
ya da bütçe 226 sınıfa saklanacak. Test bu aşamada kullanılmayacak.

**20 Ekim bağımsız karar düzeltmesi:** Gemini'nin “her klipte 20 ms pose, yalnız
%10–15'te 400–800 ms ConvNeXt uzmanı” önerisi mimari kararı değildir. Bu
sayılara ait proje/telefon ölçümü yoktur; ConvNeXt ile EfficientNetV2-S
GFLOPs değerleri farklı giriş çözünürlüklerindedir. 10 sınıf tahminlerinde
%10,05 düşük-margin yönlendirme pose'un tek hatasını kurtarıp bir doğruyu
bozarak net kazanç sağlamadı. Ayrıntı, formüller ve 100/226 sınıf karar
kapıları [bağımsız kaskad incelemesinde](secici_uzman_kaskad_bagimsiz_degerlendirme_2026-10-20.md).
**Sıradaki veri işi 100 sınıfı önceden sabitleyip strict cache ve pose-only
bazını kurmaktır; RGB uzmanı ve yönlendirme seçimi bunların eşleşmiş
sonuçlarına bağlıdır.**

Kullanıcı 10 sınıfı yalnız temel/ön işleme pilotu olarak kullanıyor; ana hedef
**226 sınıf**. 10 sınıf seed 42 pose-only tekrarı artık zorunlu sıradaki koşu
değil, ihtiyaç olursa yapılacak dayanıklılık kontrolü. [RGB uzmanı araştırması
ve deney kapıları](rgb_226_secici_uzman_arastirmasi_2026-10-20.md) güncel karar.

1. Önceden sabitlenmiş, benzer işaret çiftlerini içeren 100 sınıfta resmî
   train/val pilotu ve ayrı cache kalite kontrolü. Poz-only hatalarının
   görüntüyle kurtarılabilirliğini video bazında incele.
2. Tam 226 sınıf strict train/val cache ve manifestini üret; resmî test kapalı.
   Poz-only 226 baz çizgisini eğit. Aynı sınıf sırası ve videolarda RGB+poz
   adayı için `configs/experiment/autsl_226_rgb_convnext_tiny.yaml` hazırlandı;
   **tam cache henüz yok, eğitim yapılmadı**. T4 belleği ölçülerek ayarlar
   gerekirse pilotta düzeltilir.
3. RGB hataları gerçekten kurtarırsa poz modelinin belirsiz/zor örneklerinde
   RGB uzmanını çalıştıran seçici cascade ve geç füzyonu resmî val üzerinde
   karşılaştır; kaçan kurtarmalar, yeni hatalar, RGB çağrı oranı ve gecikme
   birlikte karar ölçütüdür. Güven eşiği tek başına güvenilir kabul edilmez.
4. Nihai tarif sabitlenince resmî test bir kez; sonra yeni işaretçi ve canlı
   kamera videosunda sürekli işaret, bilinmeyen işaret, dayanıklılık ve hız.

Mevcut MobileNetV3-Small ImageNet ön eğitimli fakat dondurulmuş el koludur;
“RGB denendi ve işe yaramadı” sonucunu bütün mimarilere taşımıyoruz.
ConvNeXt-Tiny ilk güçlü araştırma adayıdır, kanıtlanmış en iyi AUTSL omurgası
değildir. Aşağıdaki seed 42 sırası önceki kararın tarihsel kaydıdır.

## Güncel karar — pose-only seed 7 sonucu ve ölçekleme

20 Ekim [pose-only ayrıntılı rapor](poseonly_seed7_sonuclari_2026-10-20.md):
aynı strict cache ve 199 resmî 10 sınıf val videosunda seed 7 hibrit
192/199 (%96,48), pose-only **198/199 (%99,50)**; Macro-F1 0,99500.
İşaretçi 11 **26/30 → 30/30**. Yedi hibrit hata düzeldi, `signer16_sample317_color`
tek yeni hata. Eşleşen fark +6 video, keşifsel exact McNemar p=0,0703.
ID/etiket/olasılıklar, train/val işaretçi ayrılığı ve landmark cache
eşleşmesi yeniden denetlendi; splitler arasında birebir landmark dizisi yok.
Bu güçlü 10 sınıf aday sonucu, 226 sınıf veya canlı kullanım sonucu değildir.

**Önceki sıradaki Colab koşusu (şimdi isteğe bağlı):** `configs/experiment/autsl_10_poseonly_seed42.yaml`
ile aynı 20 epoch/strict cache, seed 42. Mevcut seed 42 hibritin 177/199
tahminleriyle video bazında karşılaştır. Checkpoint, tam log, config/kod sürümü,
history ve val tahminleri korunmalı. Sonuç kararlıysa ayrı 100 sınıf listesi ve
cache kalite kapısı, ardından 100 sınıf pose-only baz çizgisi; sonra 226 sınıf.
RGB kolu ve geç füzyon, ancak ölçekli hata analizinde ek değer görülürse
araştırılsın. Resmî test tarif sabitlenene kadar kapalı; canlı uygulama için
ayrı gerçek kamera/kişi, sürekli işaret ve gecikme değerlendirmesi gerekir.

## Güncel odak — işaretçi 11 ve canlı uygulama dayanıklılığı

20 Ekim [ayrıntılı yerel inceleme](signer11_dayaniklilik_analizi_2026-10-20.md):
seed 7 strict 192/199; işaretçi 11 26/30, diğerleri 166/169. İşaretçi 11'de
tespit edilen el-kare oranı %84,79, diğerlerinde %84,58. Yani düşük tespit
oranı hata kümesinin genel açıklaması değil. Kürklü yelek/koyu kıyafet ve
oturma sahnesi görüldü; görünümün nedensel etkisi henüz kanıtlanmadı. Yedi
yanlışın kare/kırpım panoları ve resmî val kalite denetimi üretildi. İki
“okul→öğretmen” hatası aynı işaretçi/sahneye ait; korelasyon gözetilecek.

**Tamamlanan teşhis deneyi (tarihsel plan):** Hazır
`configs/experiment/autsl_10_poseonly_seed7.yaml` ile Colab'da yalnız
landmark akışını, aynı strict cache/16 kare/seed 7/20 epoch bütçesinde eğit.
Amaç RGB dokuya bağımlılığın işaretçi 11 hatalarına katkısını ayırmak; yeni
modelin daha iyi olacağı iddia edilmiyor. Val video tahminleriyle genel
Top-1/Macro-F1, işaretçi 11 ve yedi hata ID'si karşılaştırılsın. Eğer
tamamlayıcıysa füzyon, değilse train-only görünüm çeşitlendirme veya üst
beden RGB akışı ayrı pilot olarak sınansın. Son tarifte resmî test ve yeni
kişilerle canlı video kalite kapıları ayrıca geçilecek; 10 sınıf Top-5 %100
canlı çeviri doğruluğu anlamına gelmez.

## Güncel karar — 20 Ekim ZIP ve iki seed analizi

Yeni [ayrıntılı rapor](colab_seed7_sonuclari_2026-10-20.md), kullanıcının
`chatgptye_gonderilecekler.zip` içindeki bütün 10 dosyayı, dört modelin 199'ar
resmi val tahminini, iki geçmişi ve iki grafiği kapsar. Güncel doğru Macro-F1
ile seed 42: base **174/199 (%87,44)**, strict **177/199 (%88,94)**; seed 7:
base **188/199 (%94,47)**, strict **192/199 (%96,48)**. Strict iki seed'de
sırasıyla **+3** ve **+4** doğru video verdi. Seed farkı aynı modelde **14–15**
videoya çıktı. Seed 42
eşleştirilmiş değişim 16 gerileme/19 iyileşme, seed 7 değişim 2/6; bu küçük
net farklar kesin üstünlük kanıtı sayılmaz. İşaretçi 11 her iki seed'de strict
ile geriledi. Resmi test kullanılmadı; %96,48 yalnız seçilmiş 10 sınıfın val
sonucu, 226 sınıfa genellenemez.

**20 Ekim karar düzeltmesi:** Seed 2026, model geliştirmeye geçmek için
zorunlu değildir. İki eşleştirilmiş seed'de strict lehine yön, onu **10 sınıf
geliştirme baz çizgisi** seçmek için yeterlidir; kesin/genellenebilir üstünlük
iddiası için yeterli değildir. Seed 7 strict checkpoint'i (%96,48) korunur.
Önce onun yedi val hatası ve özellikle dört işaretçi 11 hatası ham video,
el kırpımı, landmark ve sınıf karışıklığı açısından incelenir. Yedi hatanın
dördünü yeni hata yaratmadan düzeltmek 196/199 = %98,49 demektir; val'e
tekrar tekrar uyum sağlama riski nedeniyle bu bir araştırma hedefidir.
Sonra hata nedenine bağlı **tek bir değişken** seçilip seed 7 ile kontrollü
karşılaştırılır; seed 42'nin 20. epoch'ta hâlâ iyileşiyor olması nedeniyle
eş bütçeli daha uzun eğitim de adaydır. Yeni yöntem diğer sınıflara
ölçeklenmeden önce seçilen iyileşmenin seed/işaretçi kararlılığı denetlenir;
hazır seed 2026 çifti bu aşamada kullanılabilir. Komutları
[Colab rehberinde](colab_onisleme_karsilastirmasi.md) duruyor.
Ardından 100 sınıf train/val, sonra 226 sınıf taban ve gerekli RGB/beden
bağlamı mimarileri gelir. %98 on sınıf val, nihai 226 sınıf test hedefinin
yerine geçmez.

SAM-SLR'nin 226 sınıf resmî testinde RGB çoklu akış ensemble'ı val ile yeniden
eğitim **olmadan %97,51**, val'i eğitime **katarak %98,42** bildirdi. Bizim
train-only/val model seçimi protokolümüze tek model hedef sayısı olarak
taşınamaz. [SAM-SLR sonuç tablosu](https://arxiv.org/html/2103.08833v5).

## Önceki karar — 20 Ekim ayrıntılı seed 42 incelemesi

Masaüstündeki altı Drive çıktısı ve iki eğitim grafiği incelendi. Güncel AMP ve
doğru sınıf bazlı F1 ile `best_model.pt` val sonucu: temel **174/199 = %87,44**,
macro-F1 **0,87185**; strict aday **177/199 = %88,94**, macro-F1 **0,88852**.
Fark **3 video = 1,51 yüzde puan**; adayın son val kaybı **0,6155**, temelin
**0,6990**. Önceki 173/199 ve 178/199 sayıları eski AMP'siz değerlendirme
olduğundan güncel seçimde kullanılmaz. İki koşunun en iyi val epoch'u 20/20;
henüz kalıcı aşırı öğrenme kanıtı yok. Tek seed kalıcı üstünlük için yeterli değil.

Sıradaki somut iş: güncel değerlendirme koduyla seed 42'nin video bazlı
tahminlerini çıkar; ardından ayrı çıktı adlarıyla **seed 7 için iki 20 epoch
koşusu** çalıştır ve val metrikleri, tahminleri ile `history.csv` dosyalarını
incele. Seed 2026 konfigürasyonları da hazır; seed 7 sonucundan sonra çalışır.
Dört config üretildi ve iki çift yerelde 1.245 train/199 val, eşit bütçe,
manifest ve metadata preflight kontrolünden geçti. Colab komutları ve ayrıntılı
model incelemesi [karşılaştırma rehberinde](colab_onisleme_karsilastirmasi.md).
Resmî test kilitli, Colab eğitimi bu oturumda başlatılmadı.

## İlk log incelemesi — sonraki AMP ölçümüyle güncellendi

Seed 42, aynı Tesla T4 üzerinde iki 20 epoch koşusu tamamlandı. 199 resmî val
videosunda ayrı `best_model.pt` değerlendirmesi: temel 16 kare **173/199 = %86,93**,
`centerrepair_strict` **178/199 = %89,45**; fark **5 video = 2,51 yüzde puan**.
Eğitim içi val en iyi değerleri sırasıyla %87,44 ve %88,94, ikisi de 20. epoch.
Bu, yeni eksik-el kırpım politikasının ilk tek seed lehine işaretidir; yalnız
merkez onarımının bağımsız katkısını ölçmez. Resmî test kullanılmadı.

Notebook tablosundaki iki `0,0000` Macro F1, yanlış JSON anahtarı nedeniyleydi.
Eski değerlendirme kodu ayrıca Macro F1 formülünü yanlış uyguluyordu. Güncel
notebook ve ölçüm koduyla **aynı iki checkpoint yalnız val üzerinde tekrar
değerlendirilecek**; eğitim tekrarı gerekmiyor. AMP ayarı eğitim içi val ile
eşitlendi. Kod/komut ve sınırlamalar [Colab karşılaştırma rehberinde](colab_onisleme_karsilastirmasi.md).

Bu bölüm ilk logların tarihsel kaydıdır; üstteki ayrıntılı inceleme güncel karardır.
İlk planlanan karar kapısı: düzeltilmiş iki `_v2.json`, `history.csv` ve grafikler
incelensin. Ardından **seed 7 ve 2026 için iki kolda aynı 20 epoch** tekrarlansın.
Üç seed tutarlıysa aday ön işleme seçilsin; sonra `gap2_v2` ayrı deney olsun.
60 epoch veya yeni omurga seçimine bu kısa taramayı kesin kanıt sayarak
geçilmesin. Test, 10 sınıf deney seçimi bitene kadar kilitli.

## Güncel karar — merkez onarımı ve tam 10 sınıf ön işleme tamamlandı

19 Ekim 2026. Nihai rapor: [Merkez onarımı ve tam ön işleme](merkez_onarimi_tam_onisleme_raporu.md).
İşlem kaydı: [Çalışma günlüğü](preprocessing_calisma_gunlugu.md).
Sonraki uygulama: [Colab karşılaştırma rehberi](colab_onisleme_karsilastirmasi.md).

**Kabul edilen cache:** `cache/autsl_10_features_centerrepair_strict`.
1.245 train + 199 val, 16 kare, yaklaşık 1,647 GiB. Manifest sürüm 3, kutu ölçeği
1,50, merkez uyuşmazlık eşiği 1,0 temel kutu kenarı. 63/63 test geçti.
Bütün 1.444 ham videoda 36.587 tespitli ve 9.619 yedek kırpım piksel olarak yeniden
doğrulandı. Son beş merkez değişikliği görsel kontrol edildi. İlk tam denemedeki
iki hızlı hareket gerilemesi kural daraltılarak dışarıda bırakıldı. İlk keşif
cache'i (`autsl_10_features_centerrepair`, strict içermeyen) eğitim için reddedildi.

Vekil nokta kapsamı tam veride %96,17; bu parmak/tanıma doğruluğu değildir.
İşaretçi bazında düşük gruplar, 2 çözülemeyen örnek ve yüksek dolgu işaretleri
nihai raporda kayıtlıdır. Tamamlanan iş, 10 sınıf için kontrollü karşılaştırmaya
hazır bir ön işleme adayıdır; 100/226 sınıf veya en yüksek model doğruluğu sonucu değildir.

Sıradaki sıra:

1. Rehberdeki güncel kod/config, iki cache'in manifest+train/val içeriği, metadata klasörü ve checksum dosyasını Colab/Drive proje düzenine aktar. Yerel işlem buluta dosya yüklemedi.
2. `scripts/check_colab_inputs.py` ile aktarım sonrasında kapsam/manifest/ayar ve SHA256 kontrollerini çalıştır. Yerel preflight iki cache için toplam 2.890 dosya hash'ini doğruladı; Colab kopyası ayrıca doğrulanmalı. Metadata ile cache eğitimi için ham RGB gerekmediği yerelde sınandı.
3. Colab'da **`autsl_10_base16_screen.yaml`** ve **`autsl_10_centerrepair_screen.yaml`** ile 20'şer epoch çalıştır. Aynı MobileNetV3, seed 42, batch/optimizasyon ve augmentation; hız artırımı/erken durdurma bu taramada kapalı. Test değerlendirmesi yapılmaz.
4. İki config/manifest, `history.csv`, `training_curves.png`, en iyi epoch, val Top-1/macro-F1 ve GPU/süre bilgilerini birlikte incele. 199 val videosunda bir doğru tahmin 0,503 yüzde puan; küçük farkı seed 7 ve 2026 ile tekrar et. Bu deney bütün eksik-el kırpım politikasını kıyaslar, alt değişikliklerin ayrı katkısını ayırmaz.
5. Sonuca göre kırpım politikasını seç; ardından `gap2_v2` tek başına sınanır. 36 kare ve yeni omurga sonraki ayrı deneylerdir. 10 sınıf ölçümü oturunca 100 ve 226 sınıfa genişletilir.

Eğitim ve val komutları, gerçek Colab/Drive yollarıyla [rehberde](colab_onisleme_karsilastirmasi.md).
`*_pilot` klasörleri veya eski `autsl_10_wristcrop_screen.yaml` yeni adayın tam
cache'i değildir. Bu 19 Ekim hazırlık kaydıdır; 20 Ekim eğitim sonuçları üstte.

## Başlangıç — Resmî 10 sınıflı doğrulama

İlk çalışma, önceki on işareti resmî AUTSL bölümlerinden alır: `train=1.245`, `val=199`, `test=167`. Kullanılacak dosya `configs/experiment/autsl_10_mobilenet.yaml`; eski özel CSV, 5-fold CV ve `val+test` birleşimi kaldırılmıştır. Ön işleme üç bölüm için ayrı cache üretir. Eğitim yalnızca train cache'ini, erken durdurma ve seçim yalnızca val cache'ini okur. Test cache'i eğitim aşamasında yüklenmez.

İlk karar sırası: veri denetimi → train/val ön işleme kalite kapısı → MobileNetV3 temel çizgisi → yalnızca val üzerinden hata analizi. Test, bu kararlar tamamlandığında bir kez çalıştırılır. 10 sınıfta güvenilir şekilde çalışan akış, sınıf listesi değiştirilerek 100 ve 226 sınıfa genişletilir.

**19 Ekim düzeltmesi:** İlk üç videoluk teknik pilotta MediaPipe el etiketlerinin poz bileklerine göre ters kaydedildiği saptandı. Etiket eşlemesi düzeltildi ve o üç `.npz` ile eski pilot raporu kaldırıldı. Yeni cache manifesti `pipeline_version=2` ve seçilen özgün sınıf kimliklerini içerir; eski cache ile eğitim durur. Model başlatılamazsa boş özellik yazmak yerine ön işleme hata verir. Hız artırımı sonrası attention maskesi güncellenir; kayıp eklemlere rastgele gürültü eklenmez. El kırpımına uygulanan, fakat eklem koordinatlarına uygulanmayan dönme ve kaydırma ilk temel çizgide kapatıldı.

**10 sınıf pilotu:** Yerelde 38 train, 36 val ve 20 test videosunun cache'i üretildi. Train'de sol/sağ el bulunma oranı %83,39/%85,03; val'de %84,38/%92,53. Train ve val bilek uyumu %96–97 civarında. Hiçbir dosya veya özellik okuma hatası görülmedi. İlk eşit aralıklı örnekleme her sınıfı kapsamıyordu; `--pilot_size` artık sınıf dengeli seçim yapıyor ve 20 videoda her sınıftan iki örnek seçiyor. 94 pilot dosyası 102 MiB; aynı ortalama ile tam 1.611 dosya yaklaşık 1,7 GiB olur. Gerçek boyut değişebilir. Test verisinden model kararı verilmeyecek.

**Güncel sıradaki adım:** İlk seed 42 kısa eğitimleri tamamlandı. Ölçüm kodu düzeltildi; iki mevcut checkpoint yeniden val değerlendirmesinden geçecek, ardından seed 7 ve 2026 tekrarları karşılaştırılacak. Ayrıntı üstteki 20 Ekim bölümünde.

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

**İlk pilotun kararı (tarihsel):** 20 videoluk varyantlar eğitim karşılaştırması için yetersizdi. Daha sonra 1.444 train+val videosunun temel 16, gap2, wristcrop ve 36 cache'leri tamamlandı. Güncel karar, aşağıdaki tam veri incelemesi ve en üstteki bağımsız kontrol bölümüdür. 36 kare için `configs/experiment/autsl_10_36.yaml` zaten `max_seq_len=36` kullanıyor.

```powershell
# Mevcut 16 kare cache'i değiştirmeden aynı 20 videonun kısa boşluk pilotu:
python scripts/repair_hand_gaps.py --video_ids_file configs/experiment/autsl_10_preprocessing_pilot.json --dest_cache cache/autsl_10_features_gap2_v2_pilot
# Aynı videoların 36 kare MediaPipe pilotu:
python scripts/preprocess_tid.py --config configs/experiment/autsl_10_preprocess_36.yaml --splits train val --video_ids_file configs/experiment/autsl_10_preprocessing_pilot.json
python scripts/compare_preprocessing_pilots.py --cache16_repaired cache/autsl_10_features_gap2_v2_pilot --output outputs/preprocessing_pilot_comparison_v2.json
```

`sample_frames` artık okunamayan kare yerine yapay siyah kare eklemiyor; videoyu hata olarak bildiriyor. Bu kontrol, tam cache oluşturulmadan önce eklendi. Ayrı cache manifestleri 16 ve 36 karelerin yanlışlıkla karışmasını engeller. Onarımın `*_hand_imputed` maskesi tahmini noktaları gerçek MediaPipe tespitlerinden ayırır. Kırpım görüntüleri bu pilotta değiştirilmedi; özellikle uzun kayıplar için poz bileği merkezli güncel kare kırpımı ileride ayrı bir deney olmalı.

**İkinci kalite turu — Tam 10 sınıf önbellek ölçümleri (bağımsız incelemeyle düzeltilmiş, 19 Ekim 2026):**
İlk turdaki pilot bulguları üzerine 1.245 train ve 199 val videosunun tamamı (1.444 video) dört ayrı önbellekte tamamlandı ve denetlendi:
1. **Temel 16 kare (`cache/autsl_10_features`):** Train'de iki el ortalama tespit %78,31 (sol %78,13 / sağ %78,49); Val'de %84,61 (sol %82,69 / sağ %86,53). Bilek uyumu Train %95,5–%97,2; Val %96,0–%97,5. Tespit edilemeyen karelerde eski kırpımın donarak taşınma (stale) oranı Train'de sol %20,52, sağ %18,95; Val'de sol %16,27, sağ %11,75'tir. Süre: 19,6 dakika.
2. **Kısa boşluk onarımı:** Eski `cache/autsl_10_features_gap2` train 3.384, val 521 el karesini tahmin etti; fakat train'de 174 tahmini nokta görüntü dışındaydı. Güçlendirilmiş sürüm 2 ayrı `cache/autsl_10_features_gap2_v2` altında tamamlandı: train 1.839 sol + 1.450 sağ; val 286 sol + 216 sağ onarım, görüntü dışı tahmini nokta **0**. Güncel `autsl_10_gap2.yaml` yeni cache/sürümü kullanır. Bu yalnızca geometri kapısıdır; parmak şekli ve sınıflandırma kazancı kanıtlanmadı.
3. **Bilek merkezli canlı yedek kırpım (`cache/autsl_10_features_wristcrop`):** Train'de 8.639, val'de 980 el karesi güncel görüntüden kesildi. Val'de taşınan eski kırpım 0; train'de pozun da kaybolduğu **1 sağ el karesi** eski kırpımı taşıyor ve 1 sol el karesi ilk tespit öncesi varsayılan görüntüde kalıyor. 25 ham video yeniden okundu: 640 tespitli + 159 yedek kırpımda piksel farkı 0. Ancak varsayımsal yedek kutunun 21 noktayı kapsaması 309/639 (%48,36); 1,50 kat kutuyla 619/639 (%96,87). Bu vekil ölçüm kayıp kareler için gerçek doğruluk değildir. Ölçekli pilotun merkez hataları sonraki merkez onarımı çalışmasında ele alındı; güncel `_strict` tam cache ve Colab kararı en üstteki bölümde.
4. **Temel 36 kare (`cache/autsl_10_features_36`):** 1.444 video tamam. İki taraf tespit ortalaması train %77,7789, val %84,0731; temel 16'ya göre −0,53185/−0,53741 **yüzde puan**. Bulanıklık neden olarak kanıtlanmadı. “9+ kayıp örnek” farklı zaman kapsamlarını ölçüyor; klibin en az dörtte birine ulaşan kayıplı video sayısı train'de 392→311, val'de 33→26. Sınıflandırma eğitimi yapılmadığı için 36 kare hipotezi çürütülmüş sayılmaz. Aynı train+val boyutu 1,533→3,451 GiB (**2,251 kat**). Günlük süresi 19,6→56,8 dakika, fakat atlanan eski cache sayıları farklı; kontrollü hız deneyi değil.
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

**10 sınıf temel çizgisi değerlendirildikten sonraki iş.** Colab oturumunda kullanılabilir GPU, RAM, yerel disk ve Drive boşluğunu kaydet. Önce bu projedeki güncel `src/`, `scripts/` ve `configs/` klasörlerini Colab'ın kullandığı Drive proje klasörüne aktar; `colab_egitim.ipynb` içindeki config ve komutların resmî train/val akışını kullandığını doğrula. Resmi AUTSL kök klasörünü de Colab'a erişilebilir bir yere yerleştir. `AUTSL_DATA_DIR` değişkeni bu klasörü göstermeli; içinde `train/`, `val/`, `test/`, üç `*_labels.csv` ve `SignList_ClassId_TR_EN.csv` bulunmalı. Veri aktarımı ve depolama durumu görülmeden tam ön işleme başlatma.

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

**Şimdiki ortak karar noktası:** Yerel ön işleme tamam. Son aday `_strict` cache'i; ilk keşif cache'i eğitimde kullanılmaz. Colab aktarım kontrolü ve iki 20 epoch'luk kısa karşılaştırma sıradaki iştir. Komutlar `docs/colab_onisleme_karsilastirmasi.md`, bütün işlem kaydı `docs/preprocessing_calisma_gunlugu.md` içinde.
