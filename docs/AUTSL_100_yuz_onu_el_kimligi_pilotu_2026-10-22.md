# AUTSL 100 sınıf: yüz önündeki el ve sol/sağ kimliği pilotu

> **Devam deneyi tamamlandı:** 194 VIDEO ve ek 130 ROI-only bilek adayının kaba görsel rolü, zaman/iki algılayıcı filtresi ve kol yönü taraf ataması [yeni karar raporunda](AUTSL_100_zamansal_roi_birlesim_karar_2026-10-22.md). Aşağıdaki “27 aday” ve “sıradaki çalışma” bölümleri bu devam deneyinden önceki karar kaydıdır; tam 640 kare anatomik etiket hâlâ hazır değildir.

**22 Ekim 2026.** Bu rapor 40 **train** videosunda, dondurulmuş 16 karelik `base16` ve `centerrepair_strict` akışını değiştirmeden yapılan görsel/algoritmik kalite denetimidir. Resmî val ve test bu pilotta kullanılmadı; 100 sınıf modeli eğitilmedi. Tespit sayısı, doğru parmak noktası veya model doğruluğu değildir.

## Sorunun tanımı ve mevcut kod

`scripts/preprocess_tid.py` çıkarıcıyı `enable_face=False` ile açıyor. `src/preprocessing/mediapipe_extractor.py` içinde FaceLandmarker yalnız bu bayrak açıkken çalışır; el tespiti ayrı HandLandmarker'dan gelir. Pozun 33 noktası içinde burun/göz/kulak dayanakları vardır, fakat yükleyici bunları sol ve sağ elin 21'er noktasından sabit indekslerle ayrı tutar. Dolayısıyla yüz önündeki elin **yüz anahtar noktasıyla doğrudan indeks karışması** bu akışta gerçekleşmez. Pratik sorun, elin yüz/saç önünde kısmen örtülmesi, iki elin kesişmesi, hızlı geçişte tespitin düşmesi ve bağımsız karelerde sol/sağ kimliğinin tutarsızlaşmasıdır. Yüz bölgesini silmek veya elin yüz bölgesinde olmasını yasaklamak, işaretin anlamlı aktif elini de siler.

[MediaPipe Hand Landmarker belgesi](https://developers.google.com/edge/mediapipe/solutions/vision/hand_landmarker/python) VIDEO modunda ardışık kare takibi ve zaman damgası kullanılmasını tarif eder. [MediaPipe Holistic tasarımı](https://github.com/google-ai-edge/mediapipe/blob/master/docs/solutions/holistic.md?plain=1) poz öncülü, hassas el yeniden kırpımı ve zamansal izlemeyi sol/sağ tutarlılığı için birleştirir. Bunlar **araştırma yönü**dür; projemizdeki bağımsız Tasks API ile Holistic aynı hat değildir ve AUTSL kazanımı otomatik çıkarılamaz.

## Sabit 40 video ve inceleme malzemesi

`data/autsl/autsl_100_active_hand_annotation_queue.csv`: 16 iki eli seyrek, 16 tek tarafta uzun boşluk, 8 sağlam regresyon kontrolü; 19 işaretçi ve 26 sınıf. Her video için 16 karelik temel/üst gövde ve yoğun VIDEO panoları üretildi. `outputs/autsl_100_drive_review/dense_video_hand_40train/audit/frame_annotation_queue.csv` **640 kareyi** içerir; tespit alanları otomatik, görsel etiket alanları boştur. `novel_candidate_annotation_queue.csv` 194 yeni uzamsal adayı kare ve bilek konumuyla listeler. Bu CSV'lerde boş hücreyi “el yok” diye okumamak gerekir.

İlk yüz yakınlığı taraması yalnız *inceleme etiketi* kullandı: poz burnuna mesafe `<= max(0,08, 0,9 × omuz aralığı)`. Üst gövde ROI pilotundaki 256 ham yeni adayın **93'ü yüz yakınında** (35 ağır iki el, 57 tek taraf, 1 kontrol). Bunların 29'unda iki poz bileğine yakınlık farkı `<0,03`; yakın bileğe göre otomatik taraf ataması burada zayıftır. Çıktı: `outputs/autsl_100_drive_review/face_zone_40qa/`. Bu bölge yüz maskesi veya doğruluk etiketi değildir.

Üst gövde ROI'nin taraf etiketini taşıyan 12 videoluk tekrarında tam karedeki **aynı fiziksel bileğe** yakın 109 eşleşmenin 105'inde etiket aynı, 4'ünde farklıydı. Ayrıca 119 yeni konumun 42'si yüz yakınında, bunların 9'unda ROI etiketi ile en yakın poz tarafı farklıydı. Bu eşleşme, iki yöntemin tutarlılığıdır; insan anatomik taraf doğrulaması değildir. `signer38_sample21_color` için yüz önünde doğru görünen dokuz yeni el noktası `left` etiketi aldı; temel önbellekte kucaktaki ayrı el de `left` olarak tutuluyor. “Aynı taraf zaten var, at” kuralı gerçek yüz elini kaybeder; iki `left` noktasını doğrudan birleştirmek ise aynı karede yanlış taraf/kimlik yaratır. Çıktı: `outputs/autsl_100_drive_review/upperbody_handedness_12train/audit/`.

## Deney C — Bütün kaynak karelerde VIDEO takibi

Önceki seyrek VIDEO denemesi yalnız seçilmiş 16 kareyi takip etmişti. `scripts/pilot_autsl_dense_video_hand_mode.py` bu kez **her ardışık kaynak kareyi** zaman damgasıyla HandLandmarker VIDEO moduna verdi ve sadece eski 16 örnek indeksinde karşılaştırdı. Aynı 40 train videosunun tamamı çalıştı; ham video ve `base16` NPZ yalnız okundu. Yerel CPU koşusunda ölçülen detector+çözme/pano döngüsü toplamı **49,7 saniye**; bu değer Colab veya mobil gecikme kestirimi değildir.

- Ağır iki el grubu, 16 video: temel 90, yoğun VIDEO 180 tespitli el yuvası; 84 kazanç ve 1 kayıp kare.
- Tek tarafta uzun boşluk, 16 video: 241 → 339; 100 kazanç ve 2 kayıp kare.
- Sağlam kontrol, 8 video: 245 → 245; 6 kazanç ve **5 kayıp kare**.
- Toplam 40 video / 640 örnek kare: **576 → 764** tespitli yuva, 190 kazanç ve 8 kayıp kare. Seçilmiş sorunlu klipler tüm 100 sınıfın başarım dağılımı değildir. Kontrollerde beş kayıp kare bulunması, VIDEO sonucunu temel cache'in yerine topluca koymayı reddetmek için yeterlidir.

`scripts/audit_autsl_dense_video_candidates.py` her VIDEO bileğini temel cache'deki gözlenen bileklere `<0,06` normalize uzaklıkla eşledi. 764 VIDEO tespitinin 516'sı aynı bilek ve aynı etiketle eşleşti; **54'ü aynı bileğe yakınken etiket çatışmalı veya eşleşme belirsiz**, **194'ü temel gözlemlerden uzamsal olarak yeni**. 194'ün 71'i yüz yakınında. `194` doğru yeni el veya aktif el kurtarması sayısı değildir.

## Kare düzeyi görsel karar: ilk katı kural neden geçmedi?

Tanı amacıyla kurulan çok sınırlı geometri filtresi, yalnız bir temel taraf eksikse; VIDEO etiketi eksik tarafla ve en yakın poz bileğiyle uyuşursa; poz uzaklık farkı `>=0,03`, kendi bileğine uzaklık `<=0,18` ise adayı seçti. **Bu filtre üretim kuralı değildir.** 194 adaydan yalnız **27'si** geçti. Magenta işaretli `gate_marked_1..3.png` panolarında bu 27 konum elle incelendi ve `data/autsl/autsl_100_dense_gate_visual_review.csv` dosyasına kare düzeyinde kaydedildi:

- **6** işaretin aktif elinde; bunların dördü `signer12_sample1219_color` videosunda yüz/ağız önündeki tek elde.
- **4** kucakta duran pasif elde; örneğin `signer22_sample484_color` yüz yanındaki aktif eli zaten gözlerken 8. ve 13. karedeki yeni tespit **kucaktaki eldir**.
- **15** işaret başı/sonundaki nötr ellerde.
- **2** karede rol veya üst üste gelen ellerin kimliği güvenle ayrıştırılamadı.

Bu görsel roller ekran panolarından çıkarılmış nitel etiketlerdir; düşük güvenli anatomik taraflar CSV'de ayrıca belirtilmiştir. Özellikle 27/194 gibi bir filtre sonucunu “27 aktif el kurtarıldı” diye yorumlamak ağır biçimde yanıltıcı olur. Katı kural `signer38_sample21_color` yüz önünde gerçekten geri bulunan eli **tamamen reddeder**: VIDEO yüz eline `left` derken temel cache'de kucaktaki ayrı el zaten `left` alanını doldurur. Bu, yanlış birleştirmeyi önler ama hedeflenen genel sorunu çözmez.

Kuru regresyon kontrolünde katı filtre sağlam kontrol grubunda da **iki aday** seçiyor (`signer40_sample208_color` 5. kare, `signer12_sample1262_color` 9. kare). Bunları yazmak, bu grubun mevcut tespit maskesi/tensörünü birebir koruma şartını bozardı; hiçbir NPZ'ye yazılmadı. İkinci adayda eller üst üste ve anatomik taraf görsel olarak belirsiz. Dolayısıyla “yalnız eksik tarafı doldur” tek başına güvenli birleşim garantisi değil.

Görsel karşı örnekler:

- [Yüz elinin gerçek kurtarılması](../outputs/autsl_100_drive_review/dense_video_hand_40train/signer38_sample21_color.png): 5–13. karelerde yüz önünde görünür el; temel 1, yoğun VIDEO 2 el. Aynı videoda kimlik çakışması var.
- [Yüz önünde iki elin ortak kaybı](../outputs/autsl_100_drive_review/dense_video_hand_40train/signer17_sample52_color.png): 5–13. karelerde işaretin iki eli görünüyor; bu video ile `signer17_sample189_color` ve `signer17_sample114_color` videolarının aynı dokuzar karesinde iki yöntem de **27/27 karede sıfır el** verdi. Baştaki/sondaki nötr kazanımlar bu aktif kaybı örtmemeli.
- [Yüz önündeki etkin el](../outputs/autsl_100_drive_review/dense_video_hand_40train/signer12_sample1219_color.png): 6, 7, 9 ve 10. karelerde katı filtrenin kabul ettiği gerçek etkin yüz eli.
- [Sağlam kontrolde gerileme](../outputs/autsl_100_drive_review/dense_video_hand_40train/signer2_sample804_color.png): 3. ve 4. karelerde temel iki eli görürken VIDEO sırasıyla sıfır ve bir el görüyor. Benzer kayıplar `signer3_sample69_color` ve `signer4_sample362_color` panolarında mevcut.

## Karar ve yol haritası

**Geçiş kapısı henüz geçilmedi.** Yüz önündeki tek el için takip ve üst gövde ROI'si yararlı aday üretiyor; çakışan iki el/sol-sağ kimliği için güvenli genel bir ekleme kuralı gösterilemedi. Temel ve strict NPZ'ler, manifestler, model ağırlıkları **değişmedi**. 10 sınıf model karşılaştırmasına geçmiyoruz.

1. Görsel etiketlemeyi 640 karelik kuyrukta özellikle işaretin **etkin orta bölümü** için tamamla: gerçek sol/sağ, aktif/pasif/nötr, el görünürlüğü, çift el örtüşmesi, parmak noktasının doğru ele oturması. Şu an tam 640 karelik insan etiketi **yoktur**; yalnız 27 katı aday etiketlenmiştir. Bu eksikliği sayı gizlemek için otomatik bayrakları insan etiketi yapmayacağız.
2. Aynı videolarda, önceki ROI ve VIDEO adaylarının kimliğini iki kare komşuluğu ve kol izine göre birlikte sınayan **ayrı, yalnız eksik alanı hedefleyen** pilot kur. Tek kare poz bileği veya HandLandmarker etiketi tek başına karar vermemeli. Yüz bölgesinde eli maskeleme; belirsiz iki el çakışmasında tahmin uydurma.
3. Önce `signer17` tekrar eden orta-bölüm kaybına ve `signer38` kimlik çakışmasına bak. Ardından 8 sağlam kontrolde mevcut 245 tespitli el yuvasının ve mevcut NPZ tensörlerinin birebir korunduğunu doğrula. Kural bu iki karşı örneği birlikte çözemiyorsa `strict v1` kalır.
4. Yalnız insan etiketinde etkin el kurtarma, doğru anatomik taraf, parmak geometrisi ve sıfır kontrol gerilemesi birlikte gösterilirse, **ayrı pilot cache** üret. O zaman aynı resmî 10 sınıf train/val, seed, model ve epoch bütçesinde kısa model karşılaştırmasına geç. Val doğruluğu artışı garanti değildir.

## Tekrar üretim ve sınırlar

Komut/çıktı: `scripts/pilot_autsl_dense_video_hand_mode.py`, `scripts/audit_autsl_face_zone_candidates.py`, `scripts/audit_autsl_roi_handedness.py`, `scripts/audit_autsl_dense_video_candidates.py`; JSON ve panolar `outputs/autsl_100_drive_review/` altında. Yüz bölgesi eşiklerinin hepsi tanı amaçlıdır. VIDEO pilotu dörtte bir kadar daha fazla tespit verse bile tüm kaynak kareleri işler; 40 klipte ölçülen ~50 saniye, 100 sınıf tam cache'e ve mobil uygulamaya doğrudan ölçeklenmemelidir. Etkin elin gerçekten geri kazanıldığı küçük alt küme dışında doğruluk veya genelleme sonucu çıkarılmadı.
