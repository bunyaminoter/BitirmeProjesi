# AUTSL ön işleme: bağımsız kod ve cache incelemesi

**Sonraki durum:** Merkez onarımı ve tam 10 sınıf ön işleme tamamlandı. Güncel sonuç `docs/merkez_onarimi_tam_onisleme_raporu.md`, Colab adımları `docs/colab_onisleme_karsilastirmasi.md` içinde. Bu belge önceki aşamanın tarihsel kanıtıdır.

Tarih: 19 Ekim 2026. İncelenen Git başlangıcı: `aaa0379`. Bu commit veri yolu, eğitim, sunucu ve ön işleme değişikliklerini birlikte içeriyor; Git geçmişi tek başına hangi satırı hangi asistanın yazdığını kanıtlamıyor. Aşağıdaki sonuçlar güncel kod, günlükler, `.npz` içerikleri ve yeniden okunan ham videolara dayanıyor.

**Devam durumu:** Bu rapor ilk bağımsız incelemenin kaydıdır. Sonraki 1,50 ölçek pilotu tamamlandı; güncel sonuç ve iki merkez hatası `docs/wristcrop15_pilot_raporu.md`, yapılan işlemler `docs/preprocessing_calisma_gunlugu.md` dosyasında. Aşağıdaki yapılmamış pilot ifadeleri ilk inceleme anını anlatır.

## Doğrulanan kapsam

Dört eski varyantın her birinde aynı resmî **1.245 train + 199 val** videosu var. Dosya kimlikleri ve etiketleri resmî 10 sınıf alt kümesiyle bire bir eşleşti. Train/val işaretçi çakışması yok. İnceleme test videolarını veya test sonuçlarını açmadı. Temel cache'de önceden oluşturulmuş 20 test dosyası bulunuyor; bunlar hesaplamalara dahil edilmedi.

`wristcrop` içindeki poz/el koordinatları, tespit maskeleri ve tespit edilmiş el kırpımları temel 16 kare cache'iyle **bütün 1.444 videoda bire bir aynı**. Böylece bu varyantın değiştirdiği şeyin kayıp el karelerindeki görüntü kırpımı olduğu doğrulandı.

Yeniden üretim: `python scripts/verify_preprocessing_variants.py`. Kanıt: `outputs/preprocessing_independent_review.json`.

## 16 ve 36 kare hakkında düzeltilen yorum

- İki elin ayrı tespit oranlarının ortalaması train'de %78,3107 → %77,7789; val'de %84,6106 → %84,0731. Farklar **−0,53185 ve −0,53741 yüzde puan**. Bu metrik aynı karede iki elin birlikte bulunduğu kare oranı değildir.
- “9+ kayıp örnek” sayıları gerçekten train'de sol 43 → 170, sağ 65 → 192. Fakat 9/16 ile 9/36 klibin farklı uzunluklarını temsil eder. Bu sayılardan uzun süreli el kaybının katlandığı sonucu çıkmaz.
- Her iki taraftan en az birinde kayıp serisi örneklerin en az dörtte birini kaplayan video sayısı **train 392 → 311, val 33 → 26**. Bu da örnekleme temelli bir göstergedir; saniye cinsinden gerçek örtülme süresi veya elle etiketlenmiş hata değildir.
- Görsellerde hareket bulanıklığı var. Ancak farkın nedeninin bulanıklık olduğunu ayıran bir deney yapılmamış. Örnekleme anları değişiyor; bulanıklık açıklaması **hipotez** olarak kalıyor.
- 36 kare daha çok zaman örneği sağlıyor. Sınıflandırma eğitimi yapılmadığından “36 kare daha kötü doğruluk verir” veya “hipotez çürütüldü” denemez. Kaynak maliyeti nedeniyle şu an ilk deney sırasına alınmıyor.
- Aynı train+val kapsamının ölçülen boyutu 16 karede **1,533 GiB**, 36 karede **3,451 GiB**: **2,251 kat**. 36.302 videoya düz ölçekleme yaklaşık 86,8 GiB verir; sınıf ve görüntü içeriği değiştiğinden kapasite garantisi değildir.
- Günlüklerde 1.176,8 saniye ve 3.409,8 saniye var. İki koşuda önceden hazırlanmış farklı sayıda video atlanmış; bunlar sıfırdan eşit kapsamlı hız testi değildir.

Rapordaki video düzeyinde bootstrap aralıkları aynı videolardaki örnekleme farkını özetler; işaretçilerin bağımsız tekrarları gibi yorumlanamaz ve sınıflandırma doğruluğunu ölçmez.

## `wristcrop`: kazanım gerçek, tamamlandığı iddiası erken

Train'de 4.355 sol + 4.284 sağ, val'de 551 sol + 429 sağ **el karesi** güncel görüntüden kesilmiş. Train'deki toplam 8.639 sayısı benzersiz video karesi sayısı değildir; iki el ayrı sayılır.

Bütün cache'de yedek kırpım olarak işaretlenen görüntüler arasında önceki kırpımla piksel olarak aynı kalan veya tümü siyah olan bulunmadı. Ancak poz da kaybolunca train'de **1 sağ el karesi eski kırpımı taşıyor** (`signer33_sample505_color`, sıfır tabanlı örnek 2); ayrıca 1 sol el karesi ilk tespitten önce varsayılan görüntüde kalıyor. Val'de taşınan eski kırpım yok. “%0,0” yuvarlanmış gösterimdir; mutlak sıfır iddiası doğru değil.

Sınıf başına rastgele bir örnek ve zor örnekler içeren **25 ham video** tekrar açıldı. 640 tespit edilmiş kırpım ve 159 yedek kırpım kontrol edildi; piksel farkı sıfır. Eski hizalama denetimi her videoda yalnızca bir tespit edilmiş kırpımı kontrol ediyordu; bütün karelerin doğruluğunu kanıtlamıyordu.

Yeniden üretim: `python scripts/verify_wristcrop_pixels.py`. Kanıt: `outputs/wristcrop_raw_pixel_review.json`. Mevcut dört önce/sonra panosu da görsel incelendi: eski görüntünün taşınması büyük ölçüde giderilmiş; bulanıklık, elin tamamının kadraja girmemesi ve çaprazlanan ellerde merkez kayması hâlâ kontrol gerektiriyor.

**Yeni kalite uyarısı:** Gerçek el noktalarının bulunduğu karelere, el kayıpymış gibi mevcut poz merkezli yedek kutu uygulanarak bir kapsam testi yapıldı. 639 el karesinin yalnızca **309'unda (%48,36)** 21 noktanın tamamı mevcut kutuya sığıyor. Aynı merkezde kutu kenarını büyütmekle elde edilen kapsam: 1,25 kat için 581/639 (%90,92), 1,50 kat için 619/639 (%96,87), 1,75 kat için 630/639 (%98,59), 2,00 kat için yine 630/639.

Bu, **tespit edilmiş kareler üzerinde bir vekil ölçümdür**. Gerçek kayıp karelerdeki elin doğrusu bilinmediği için onların başarı oranı değildir; MediaPipe'ın 21 noktası da insan anotasyonu değildir. Yine de sabit medyan boyutunun değişen el açıklığında dar kalabileceğine işaret eder. Daha büyük kutu daha çok arka plan ve diğer eli de alabilir; büyütme sınıflandırma kazancı sayılmaz. Cache yalnızca pozun bulunma bilgisini saklıyor; bileğin görünürlük/güven skoru yok. `pose_detected=True`, bileğin doğru konumda olduğunu garanti etmez.

## Düzeltilen açıklar

1. Eski `gap2` train cache'inde 9 videoda **174 tahmini x/y noktası görüntü dışındaydı**. Bu noktaları sınırda kırpmak yerine, uç noktaları görüntü dışında olan boşlukların onarımı engellendi. Aradaki tahmini bileğin o karenin poz bileğiyle uyuşması da artık kontrol ediliyor. Gözlenen gerçek noktalar değiştirilmedi. Görüntü dışındaki MediaPipe noktaları tek başına kesin yanlış tespit kanıtı değildir; yeni politika tahmin üretirken daha ihtiyatlıdır.
2. Düzeltilmiş `gap2` sürüm 2, **ayrı `cache/autsl_10_features_gap2_v2`** altında 1.444 video için üretildi. Train 1.839 sol + 1.450 sağ; val 286 sol + 216 sağ el karesi onarıldı. Tahmini görüntü dışı nokta sayısı **0**. Eski cache inceleme kanıtı olarak duruyor. `autsl_10_gap2.yaml` yeni cache'i kullanıyor. Yapılandırılan türetme sürümü manifestle uyuşmazsa yükleyici duruyor.
3. Kalite denetimine yuvarlanmış oranların yanında kesin taşınan-kırpım ve ilk tespit öncesi kare sayıları eklendi. Kanıt: `outputs/preprocessing_fixed_quality.json`.
4. Üç tahmin girişinde 99 poz koordinatı hazırlanırken model 225 değer bekliyordu; iki el koordinatları ve ImageNet normalizasyonu eksikti. Ortak `src/preprocessing/inference_pipeline.py` eklendi. CLI ve iki API artık 33+21+21 nokta sırasını, kırpım politikasını, seçilen onarımı ve val ile aynı normalizasyon/padding işlemini kullanıyor. Okunamayan kare sessizce atılmıyor.
5. Sunucudaki eski ASL sınıf adlarına dönüş kaldırıldı. Checkpoint yüklenemezse rastgele ağırlıklı modelin tahmin sunması engellendi.

Başlangıçta mevcut 39/39 test geçti. Düzeltmelerden sonra **45/45 test** geçti; Python derleme kontrolü de başarılı. Yeni testler eğitimin val yükleyicisi ile tahmin girdilerini piksel, 225 koordinat sırası ve padding dahil bire bir karşılaştırıyor. Eğitilmiş yeni bir AUTSL checkpoint'iyle gerçek uçtan uca tanıma sonucu henüz ölçülmedi.

## Karar ve bir sonraki adım

**Öncelik daha büyük model veya 36 kare eğitimi değil; yedek kırpımda elin tamamını korumak.**

1. Aynı 25 videoyu sabitle. Mevcut kırpımı değiştirmeden, **1,50 kat kutulu ayrı küçük pilot cache** üret. Gerekirse 1,75 katı ikinci aday olarak ekle. Ölçek parametresi manifestte saklanmalı; ayrı sürüm ve çıktı klasörü kullanılmalı. Bu ölçekli cache henüz üretilmedi; mevcut kodda ölçek seçeneği henüz yok.
2. Güncel RGB görüntüsü + poz/el noktaları + kutu sınırları + gerçek yeni kırpım aynı panoda gösterilsin. Her adayda parmak ucu kesilmesi, diğer elin alınması, aşırı arka plan, siyah dolgu ve merkez sıçraması sayılıp sınıf/işaretçiyle kaydedilsin. Özellikle tespit olmayan kareler elle incelensin. Vekil kapsam için %95 bir tarama eşiği olarak kullanılabilir; bu doğruluk garantisi değildir. Poz bileği yanlışsa yalnızca kutuyu büyütmek yeterli olmaz.
3. Görsel kapıdan geçen tek adayı aynı 1.245 train + 199 val kapsamına genişlet. Manifest, ham kare hizası, kaynak kodları ve tahmin akışı eşleşmesi tekrar doğrulansın. Başarısızsa kutu merkezleme/ölçekleme yeniden ele alınsın; eğitim bu sorunun yerine geçmez.
4. Ardından 20 epoch, seed 42, aynı dondurulmuş MobileNetV3 ve aynı optimizasyon bütçesiyle temel 16 kareye karşı kısa eğitim yap. Hazırlanan `autsl_10_base16_screen.yaml` ve `autsl_10_wristcrop_screen.yaml` bütçeleri eşit; zaman içinde piksel karıştıran hız artırımı ikisinde de kapalı. **İkinci config şu anda mevcut sürüm 1 cache'e bağlıdır; ölçekli aday kabul edilirse yeni manifest/sürüm/cache'e güncellenmeden çalıştırılmamalı.**
5. Bana iki koşunun config/manifesti, `history.csv`, `training_curves.png`, en iyi epoch, val Top-1/macro-F1 ve süre/GPU bilgisi gönderilsin. Val 199 video: tek doğru tahmin yaklaşık 0,503 yüzde puandır. 1–2 videoluk kazanç kesin üstünlük sayılmaz; yararlı aday seed 7 ve 2026 ile de aynı koşullarda tekrarlansın. Test karar sürecine dahil edilmesin.
6. Kırpım kararı sabitlendikten sonra `gap2_v2` tek başına sınansın; iki iyileştirme en başta birleştirilmesin. 36 kare ve takip modu bundan sonra ayrı deneylerdir. 36 kare config'inde `max_seq_len=36` zaten doğru ayarlı.

Bu inceleme yeni model eğitimi başlatmadı. Kullanıcı sonuçları paylaştıkça bir sonraki eğitim varyantı val kanıtıyla seçilecek.
