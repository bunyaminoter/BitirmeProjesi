# AUTSL için bulut, mobil ve hibrit çıkarım kararı

**Tarih:** 29 Ekim 2026  
**Karar:** 226 model eğitimi kararını değiştirmeden, **sunucu merkezli çıkarımı** uygulama adayı olarak kabul et. İlk ürün pilotunda sıkıştırılmış işaret klibini sunucuya gönder; MediaPipe ön işleme, pose-only, RGB+pose ve sabit füzyon sunucuda çalışsın. Telefonda yalnız kamera, klip sınırı/örnekleme, kodlama, ağ ve sonuç gösterimi olsun. Bu mimari gerçek telefon ve hedef sunucuda p50/p95 uçtan uca ölçülmeden “daha hızlı” diye ilan edilmez. Ağsız kullanım istenirse küçük pose-only model daha sonra ayrı **çevrimdışı yedek** olarak eklenebilir. Koşullu RGB çağrısı bugünkü %90,88 füzyon sonucunu koruyacağı kanıtlanmadığından ilk sürüm kararı değildir.

## Projede şu an ne var?

- [`scripts/server.py`](../scripts/server.py) tek bir `HybridASLModel` checkpoint'iyle `/predict` (MP4 yükleme) ve `/predict_frames` (JPEG kare yükleme) sunan FastAPI prototipidir. İki modelin 0,5/0,5 füzyonunu sunmuyor. Varsayılan config/checkpoint hâlâ 10 sınıf eski deneyi işaret ediyor; 226 sunucusu hazır değil. [`src/api/main.py`](../src/api/main.py) de tek model ve sabit 10 sınıf yolu kullanıyor. Dolayısıyla bu dosyalar doğrudan internete açılacak üretim servisi değildir.
- Her istekte sunucu video karelerini ayıklar, [`prepare_inference_batch`](../src/preprocessing/inference_pipeline.py) ile MediaPipe ve `centerrepair_strict` çıkarımını yapar, ardından modeli çalıştırır. Çıkarım tarafının 16 kare ve crop tarifinin eğitim cache'iyle eşleşmesi gerekir. Sunucuda iki modeli çalıştırırken **aynı ön işleme bir kez** yapılmalı; model başına tekrar edilmemeli.
- API'nin `inferenceTimeMs` sayacı endpoint fonksiyonunda başlıyor. Telefonun kamera bekleme/klip oluşturma, video kodlama, ağ yüklemesi, istek kuyruğu öncesi ve yanıtın geri gelişi bu sayıda yoktur. Gerçek gecikme telefonun `capture_end → result_rendered` zaman damgalarıyla ölçülür.
- Mevcut 100 sınıf hibrit `best_model.pt` **382.610.079 bayt**, pose-only **24.023.051 bayt**. Bunlar eğitim checkpoint dosyalarıdır; optimizer ve diğer durumlar içerebilir, **mobil dağıtım boyutu olarak doğrudan alınamazlar**. Sunucuya taşıma büyük hibrit ağı uygulama paketinden çıkarabilir; mobilde kamera/kodlayıcı/ağ kodu yine kalır.

## Ölçülmüş aktarım boyutu ve yalnız aktarım alt sınırı

Drive Desktop'taki resmî AUTSL val **4.418 sıkıştırılmış 512×512 RGB MP4** dosyasının yalnız metadata boyutları sayıldı. [Yeniden üretilebilir ölçüm](../scripts/audit_cloud_clip_sizes.py) ve [JSON sonucu](../outputs/autsl_cloud_clip_size_audit_2026-10-29.json): ortanca **375.423 bayt** (yaklaşık 367 KiB); yüzde 90 **553.017 bayt** (yaklaşık 540 KiB), yüzde 99 **702.866 bayt**. Bunlar telefonun kendi kodlayıcısının üreteceği dosya boyutları değildir.

`salt_aktarim_suresi_ms = dosya_bayt × 8 / etkin_yukleme_Mbps / 1000`:

- **5 Mbps etkin yükleme:** p50 klip **601 ms**, p90 **885 ms**.
- **10 Mbps:** p50 **300 ms**, p90 **442 ms**.
- **20 Mbps:** p50 **150 ms**, p90 **221 ms**.
- **50 Mbps:** p50 **60 ms**, p90 **88 ms**.

Bu sayılar **iyimser fiziksel alt sınırdır**. Telefonun indirme hızından değil, sunucuya **etkin yükleme** hızından hesaplanır. Ağ gidiş dönüşü, TLS/HTTP, dalgalanma, kodlama, sunucu sırası, video çözme, MediaPipe ve iki model çıkarımı dahil değildir. Telefon videosunun bit hızı ve formatı cihaz/tarayıcıya göre değişebilir; [MediaRecorder bit hızı](https://developer.mozilla.org/en-US/docs/Web/API/MediaRecorder/videoBitsPerSecond) istenen değerden farklı olabilir. Ham 16×512×512 RGB yaklaşık **12,6 MB** olduğundan sıkıştırmasız tam kare gönderimi uygun değildir. 16×75×3 float32 landmark ise yalnız **14,4 KB**, fakat onu telefonda üretmek için MediaPipe yükü kalır.

## Gecikme nerede oluşur?

**Klip bittikten sonraki görünen süre** yaklaşık `telefon kodlama + ağ yükleme + ağ gidiş-dönüşü + sunucu sırası + video çözme + MediaPipe/strict + pose ve RGB modelleri + yanıt/gösterim` toplamıdır. İşaretin başlamasından itibaren süreye klibin kendisi ve işaret sonu algılama da eklenir. Bu iki sürenin karıştırılması kullanıcı deneyimini olduğundan hızlı gösterir.

Sunucuda GPU güçlü olduğunda hesaplama telefondan daha hızlı olabilir ve buna karşılık ağ eklenir; **net hız kazancı bir eşitsizliktir**, garanti değildir: `bulut ek ağ/kuyruk + bulut işlem < cihazdaki ön işleme + iki model işlem` ise bulut daha hızlı görünür. Sunucu yoğunluğunda kuyruk p95'i artırabilir; [NVIDIA Triton metrikleri](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/protocol/extension_statistics.html) bu nedenle queue ile compute'u ayrı izler. Mevcut FastAPI prototipinde bu ayrım yok.

## Dört mimari seçeneğin artı ve eksileri

**1. Her şey sunucuda, sıkıştırılmış klip isteği — ilk önerim.** Telefon MediaPipe veya ConvNeXt çalıştırmaz; uygulama paketi ve sürekli hesap yükü küçülür. Sunucu modelleri güncelleyebilir, iki modeli aynı özelliklerden çalıştırabilir. Ağ yoksa tahmin yoktur; video dışarı çıkar, sunucu GPU maliyeti ve kuyruk yönetimi doğar. Kamera ve video kodlama/radyo hâlâ batarya ve ısı tüketir; “ısınma sıfır” değildir. Klibin tümü gönderilen mevcut `/predict` akışında yükleme ve hesaplama büyük ölçüde klip bitişinden sonradır.

**2. Pose telefonda, RGB sunucuda.** Pose sonucu ağsız da gelir; yalnız zor örneklerde RGB isteği mobil veriyi azaltabilir. Ama MediaPipe ve pose modelinin telefon yükü/uygulama boyutu kalır. Basit el-kayıp işareti 100 sınıfta bozulan/kurtarılanları zayıf ayırdı (AUC 0,528); düşük güven veya karışıklık çiftiyle seçici yönlendirme de 226'da ayrıca doğrulanmalıdır. Seçici istek, **her klipte iki modelle ölçülmüş %90,88** sonucunu otomatik korumaz. İki tarafta ön işleme ve sınıf/versiyon eşliği zorlaşır.

**3. Sunucu asıl yol + telefonda pose-only çevrimdışı yedek — ikinci aşama önerim.** Çevrimiçi ağır iş yine sunucuda; bağlantı koparsa pose-only devreye girer. Yedek için MediaPipe ve küçük model telefonda tutulacağından paket boyutu saf buluttan büyük olur. Çevrimdışı doğruluk birleşim doğruluğu kadar olmayacaktır. Ağ durumunda otomatik geçiş ve sonuç tutarlılığı ölçülmelidir.

**4. Tam cihaz içi.** İnternet bağımsızlığı ve videonun cihazda kalması güçlü avantajlar. ConvNeXt ve MediaPipe cihaz belleği, ısı, batarya ve dağıtım boyutu açısından riskli; optimizasyon/quantization ve gerçek telefon ölçümü gerektirir. Sunucu maliyeti yoktur. [Android'in cihaz içi çıkarım notu](https://developer.android.com/ndk/guides/neuralnetworks), ağ gecikmesi ve çevrimdışı erişimi bu yolun artıları olarak listeler.

## Akış protokolü: önce MP4, sonra gerekirse WebRTC

İzole işaret için **tek sıkıştırılmış klip + HTTPS POST** ilk ölçüm açısından basittir ve mevcut sunucu taslağına yakındır. `/predict_frames` ile 16 JPEG göndermek video kodlama/çözmeyi değiştirebilir fakat çoklu JPEG'in toplam baytı daha büyük olabilir; bu, gerçek telefonda karşılaştırılmalı. Tam video ile 16 JPEG'in MediaPipe tespit ve Top-1 eşliği de kontrol edilmeli.

Sürekli canlı akış için [WebRTC](https://www.w3.org/TR/webrtc/) daha sonra değerlendirilebilir: aktarım kayıt süresiyle örtüşebilir, ama kare sırası/timestamp, paket kaybı ve 16 karelik işaret penceresi yeniden tasarlanır. [WebRTC istatistikleri](https://www.w3.org/TR/webrtc-stats/) RTT, gönderilen bayt, paket kaybı ve kullanılabilir giden bant genişliğini izlemeye olanak verir. Bugünkü model **tamamlanmış izole klipten** 16 uniform kare alıyor; sürekli akışta işaret segmentasyonu ve kayan pencere ayrı doğruluk problemidir. WebRTC'ye geçmek modeli kendiliğinden canlı tercümana dönüştürmez.

## Servis hazırlığı ve gerçek karar kapısı

1. **100 sınıf düşük maliyetli ağ prototipi:** Eğitilmiş iki checkpoint'i aynı servis işlemine yükle, 16 kareyi bir kez çıkar, iki modelin 100 sınıf 0,5/0,5 olasılıklarını döndür. Aynı girişlerde mevcut offline tahminlerle eşlik kontrolü yap. Sunucuda `decode`, `MediaPipe`, `pose`, `RGB`, `queue` sürelerini; telefonda `capture_end`, `upload_start/end`, `response`, `render` zamanlarını kaydet. Bu deney ürün doğruluğu kararı değildir, yalnız taşıma ve hız tanısıdır.
2. **Klip/codec karşılaştırması:** Hedef Android/iOS telefondan aynı 50–100 videoluk kontrollü kayıt veya yeniden oynatımda gerçek MP4/WebM dosya boyutu, p50/p95 `capture_end → result`, Wi-Fi ve hücresel ağda yükleme, başarısızlık ve 10–15 dakika sürdürülebilir ısı/batarya ölç. Telefon kodlamasının MediaPipe ve tahminleri değiştirmediğini ayrıca kontrol et. Hız testi **telefonun sunucuya etkin uplink'i** üzerinden yapılmalı; masaüstü Drive hızı yeterli gösterge değildir.
3. **226 hedefi:** 226 cache QA ve iki eş koşullu model tamamlandıktan sonra aynı servisi 226 checkpoint'lerine geçir; 4.418 resmî val ve yeni kamera videolarını ayrı raporla. Sabit füzyon 226'da gerçekten yararlıysa ve telefon p95 gecikmesi kullanıcı için kabul edilebilir ise bulut mimarisini ürün adayı yap. Eşik bugün ölçülmüş değildir.
4. **İnternete açmadan önce:** Mevcut prototipte geniş CORS, varsayılan 10 sınıf yolları, tek model akışı ve belirgin kimlik doğrulama/istek boyutu sınırı eksikleri var. Videolar görüntü verisidir; HTTPS, erişim denetimi, boyut/süre sınırı, güvenli geçici dosya yaşam döngüsü ve kısa saklama kuralı gerekir. [OWASP dosya yükleme rehberi](https://cheatsheetseries.owasp.org/cheatsheets/File_Upload_Cheat_Sheet.html) sunucu kaynak tüketimini ve yükleme güvenliğini özellikle ele alır. Bu servis bugün doğrudan herkese açık dağıtıma hazır değildir.

**Sonuç:** Bulut, özellikle **telefonun büyük RGB ağını ve MediaPipe'ı çalıştırmaması** açısından mantıklı ve ürün boyutunu azaltabilecek bir mimari. Sıkıştırılmış AUTSL klipleri küçük olduğundan iyi uplink'te ağ maliyeti düşük *olabilir*. Kötü uplink, kuyruk, video kodlayıcı farkı veya uzak sunucu bunu tersine çevirebilir. Kararı uydurma milisaniye yerine kısa, mevcut modellerle yapılacak uçtan uca pilot belirleyecek.
