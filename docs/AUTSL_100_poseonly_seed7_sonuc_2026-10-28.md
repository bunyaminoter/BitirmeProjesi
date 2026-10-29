# AUTSL 100 sınıf pose-only eğitimi ve sonraki karar

> **28 Ekim devam sonucu:** Bu rapordaki 35 videoluk sonraki adım tamamlandı. [Ham video, eğitim referansı, landmark bindirmesi, geometri ve 226 sınıf kararı](AUTSL_100_hata_tanisi_ve_226_karari_2026-10-28.md). Aşağıdaki “şimdiki tek sonraki adım” bölümü inceleme öncesi tarihsel plandır.

**28 Ekim 2026. Karar:** 16 kare `centerrepair_strict` v1 korunacak. İlk 100 sınıf tam koşusu başarılı bir baz çizgisi verdi: resmî val'de **1.678/1.963 Top-1 (%85,48)**, **1.914/1.963 Top-5 (%97,50)**, **Macro-F1 0,8524**. İkinci pahalı 100 sınıf eğitimine hemen başlanmayacak. Önce tekrar eden karışıklıkların ham video, tüm kare landmarkları ve crop bilgisiyle sebebi ayrılacak; bu denetim ikinci modeli seçmek veya bütçeyi 226 sınıfa saklamak için kullanılacak.

## Dosya ve protokol doğrulaması

- Drive Desktop kaynağı: `G:\Drive'ım\BitirmeProjesi\outputs\autsl_100_poseonly_seed7`. `val_metrics.json`, 1.963 videolu `val_predictions.json`, 35 epoch `history.csv`, eğitim grafikleri ve `best_model.pt`, `final_model.pt`, `latest_model.pt` mevcut.
- Config 100 sabit AUTSL sınıfı, seed 7, 16 kare, strict v1 cache, `use_rgb: false`, 35 epoch ve batch 16 tanımlıyor. Modelde RGB kolu kapalı; 33 gövde + 42 el landmarkı korunuyor. 100 sınıf cache manifestindeki sınıf sırası ve strict türetim ayarları config/tahminlerle aynı.
- Resmî CSV'de seçili sınıflar için **12.405 train / 1.963 val** kimliği var. Train ve val video kimlikleri ile işaretçileri ayrık. Tahmin JSON'undaki 1.963 benzersiz video ID'si ve gerçek etiketlerin tümü resmî seçili val alt kümesiyle birebir eşleşti. Test kullanılmadı.
- Her 100 elemanlı olasılık dizisi sonlu, yaklaşık 1'e topluyor; argmax, raporlanan tahmin ve güven değeriyle uyumlu. Top-1, Top-5 ve Macro-F1 yeniden hesaplandı ve metrik JSON'uyla eşleşti.
- `best_model.pt` hesaplanan SHA-256 `668023b6502450c29322851d727e3ad68ac80c3e1479cc0e76b40c5fb86a46f9`; tahmin JSON'una yazılan hash ile aynı. Ağırlıkla yerel GPU yeniden inference yapılmadı. `history.csv` en iyi doğruluğu **29. epoch** gösteriyor; son 35. epoch val %85,02. Son epoch'a göre hafif düşüş var, fakat son epochlar %84–85 bandında; ağır ve tek yönlü bir çökme yok.

## Öğrenme ve hata dağılımı

| Ölçüm | Değer |
|---|---:|
| Train / val video | 12.405 / 1.963 |
| En iyi epoch | 29 / 35 |
| Top-1 | 1.678/1.963 = %85,48 |
| Top-5 | 1.914/1.963 = %97,50 |
| Macro-F1 | 0,8524 |
| Yanlış tahmin | 285 |

On sınıftaki pose-only seed 7 sonucu **198/199 (%99,50)** idi. Sınıf sayısı, seçilen örnekler ve eğitim hacmi değiştiği için 100 sınıftaki %85,48, aynı deneyde 14,02 puanlık model gerilemesi gibi yorumlanmaz. Beklenen zor ölçekleme farkının somut ölçümüdür. Val eğrisi 29. epoch'a kadar anlamlı ilerledi; yalnız 20 epoch ile durdurulması bu koşuyu eksik bırakırdı.

**Val işaretçi kırılımı:** signer16 **141/196 (%71,94)**, signer11 **238/294 (%80,95)**, signer1 **514/600 (%85,67)**, signer35 **258/291 (%88,66)**, signer25 **264/293 (%90,10)**, signer18 **263/289 (%91,00)**. Bu farklılıklar işaretçi sayısı ve görünümle birlikte yorumlanmalı; signer16 tek başına 55 hata içeriyor.

**En zayıf sınıflardan örnekler:** özgün AUTSL ID 154 `para` **6/20**, 76 `görmek` **7/19**, 22 `bakmak` **8/20**, 173 `selam` **8/20**, 148 `onlar` **9/20**. Bu sonuçlar sınıf başına yalnız 17–20 val videosu olduğu için tek tek belirsizdir; ortak hata kümelerini önceleyip tek videoya özel kural yazmayacağız.

**En sık yönlü karışıklıklar:** `onlar` 148 → `kalem` 103 **9**, `şeker` 172 → `Atatürk` 16 **8**, `okul` 144 → `beklemek` 29 **6**, `selam` 173 → `psikoloji` 163 **6**, `bakmak` 22 → `söylemek` 183 **5**. İlk üç çift birden fazla val işaretçisinde tekrar ediyor. Örneğin `onlar→kalem` altı signer1 ve üç signer11 videosunda; `onlar` sınıfı signer18/25/35'te 9/9 doğru. Bu, kişiyle etkileşen hareket/şekil farkını incelemeyi gerekli kılıyor; sınıf etiketi hatası veya RGB gerekliliği henüz doğrulanmadı.

## Ön işleme kalitesi ile hatanın ilişkisi

Strict val NPZ dosyalarının yalnız tespit maskeleri tarandı. **Bir istisna dışında 16 kare:** `signer16_sample369_color.npz` yedi kare içeriyor; model veri yükleyicisi kısa diziyi maskeyle pad ediyor. Bu video yanlış tahminli, fakat 285 hatanın ortak nedeni değil.

- Yanlış videoların **210/285'inde** 16 örnek karenin hiçbirinde iki el aynı anda kayıp değil. Sık üç karışıklık çiftindeki **23 videonun hepsinde** her örnek karede en az bir el tespitli. Bu nedenle yeni bir genel “eksik el doldurma” politikası bu hataların büyük bölümünü doğrudan hedeflemiyor.
- Yanlış videolarda ortalama iki elin de kayıp olduğu kare **1,50**, doğru videolarda **0,89**. Bu ilişki büyük ölçüde işaretçi dağılımıyla karışıyor: signer16'da doğru videolar **6,15**, yanlışlar **6,56** kayıp kare ortalamasına sahip. signer1 ve signer11'de yanlış videoların kayıp kare ortalaması doğru videolardan daha düşük. Tespit açığı özellikle signer16 için ayrı bir kalite sorunu; bütün modele uygulanacak yeni ön işleme kararı için tek başına yeterli değil.
- Örnek crop panosunda `onlar→kalem` hataları ile doğru kontrol videolarının seçili dört karesi incelendi. Hatalı videolarda eller görünür, fakat seçili karelerde iki sınıfın el biçimleri oldukça benzer. Dört karelik pano tam video hareketini ve yüz/gövde bağlamını göstermez; bu görsel gözlemden kesin neden çıkarılmıyor. [Pano](../outputs/autsl_100_poseonly_review/error_crop_sheet.png).

**Güven eşiği uyarısı:** 285 hatanın 60'ı ≥0,8, 22'si ≥0,9 en yüksek sınıf olasılığıyla yapıldı; 109 hatada ilk iki sınıf olasılık farkı ≥0,5. Dolayısıyla yalnız düşük margin ile RGB uzmanını çağıran bir kaskad bu hataların çoğunu kaçırabilir. Buna karşılık yanlışların **236/285'inde** gerçek sınıf ilk beşte; ikinci modelin bunları düzelteceği ancak eşleşmiş tahminle ölçülebilir. Top-5, canlı çeviri Top-1 başarısı değildir.

## Şimdiki tek sonraki adım

**İkinci tam eğitime geçmeden hedefli hata tanısı.** En çok tekrar eden üç çiftin yanlış ve doğru kontrol videolarından işaretçi dengeli küçük bir pano oluştur: ham videoda işaretin başlangıç/orta/bitiş hareketi, 16 karelik sol/sağ el landmarkları, eldeki parmak şekli, crop kayması, yüz/gövde bağlamı ve resmî etiket birlikte kontrol edilsin. Aynı sınıfın train işaretçilerinden de birkaç örnek karşılaştırılsın. Tekil signer16 kalite sorunu ayrı tutulmalı. Bu işlem Colab tam eğitim hakkı harcamaz.

İlk **35 videoluk inceleme kuyruğu** hazır: üç çiftteki 23 tekrarlı hata ve her çift için dörder doğru kontrol. [CSV](../outputs/autsl_100_poseonly_review/raw_video_review_queue.csv) video ID, işaretçi, sınıf, güven, el tespit sayısı ve Drive ham video/cache yolunu içeriyor. Bu kuyruk seçim içindir; henüz 35 ham videonun tamamı görsel olarak etiketlenmedi.

Sonuca göre **tek ikinci koşu** seçilecek: (a) şekil/görünüm farkı landmarkta kayıp ama crop'ta açık ise eğitilebilir, bütçeye uygun RGB+pose adayını sınamak; (b) ayırıcı unsur zamansal yörüngeyse yalnız pose zaman/normalizasyon temsilini iyileştirmek; (c) belirgin genel eksik bilgi yoksa ikinci 100 sınıf koşusunu atlayıp 226 sınıf strict pose-only bazına bütçe ayırmak. ConvNeXt-Tiny veya otomatik seçici uzman şu an önceden onaylanmış ikinci koşu değildir. Önceki 10 sınıf hibrit sonuçları RGB kazancını 100 sınıfta garanti etmez. [SAM-SLR'nin AUTSL çoklu akış çalışması](https://arxiv.org/abs/2103.08833), modalitelerin tamamlayıcı olabileceğini gösterir; aynı bütçe ve model olmadığından bizim ikinci koşunun beklenen sayısal artışını vermez.

**Sınır:** Bu rapor resmî 100 sınıf val sonucudur, 226 sınıf ve resmî test sonucu değildir. 100 sınıf listesi amaçlı seçildi. Val üzerindeki devam eden kararlar sonuçta iyimserlik yaratabilir; test mimari sabitlenene kadar kapalı kalır.

**Yerel denetim dosyaları:** `outputs/autsl_100_poseonly_review/analysis.json`, `detection_audit.json`, `training_curves.png`, `error_crop_sheet.png`, `raw_video_review_queue.csv`. Analiz betikleri `scratch/analyze_100_pose_result.py`, `scratch/audit_100_val_detection.py`, `scratch/make_100_error_crop_sheet.py`, `scratch/build_100_error_review_queue.py`.
