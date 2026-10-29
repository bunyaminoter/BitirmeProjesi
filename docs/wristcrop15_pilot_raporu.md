# 1,50 kat bilek kırpımı pilotu

**Sonraki durum:** Merkez onarımı ve tam 10 sınıf ön işleme tamamlandı. Güncel sonuç `docs/merkez_onarimi_tam_onisleme_raporu.md`, Colab adımları `docs/colab_onisleme_karsilastirmasi.md` içinde. Bu belge önceki aşamanın tarihsel kanıtıdır.

Tarih: 19 Ekim 2026. Karar: **ölçek tek başına tam cache'e ve eğitim deneyine kabul edilmedi; merkez onarımı gerekiyor.** Model eğitimi Colab'da yapılacak.

## Yapılan işlem ve kapsam

Önceki bağımsız incelemenin 25 videosu `configs/experiment/autsl_10_wristcrop_pilot25.json` dosyasında sabitlendi: **13 train + 12 val**, her bölümde 10 sınıfın tamamı ve ilave zor örnekler. 16 kare, 160×160 çıktı, mevcut sınıf sırası ve tespit maskeleri korundu. Test bölümü bu koşuda kullanılmadı.

Yalnızca MediaPipe el tespitinin kaybolduğu, pozdan kutu üretilebildiği karelerde kutunun **kenarı 1,50 kat** büyütüldü. Alan yaklaşık 2,25 kat artar; çıktı hâlâ 160×160 olduğu için el görüntüde daha küçük görünebilir. Nokta koordinatları tahmin edilmedi/değiştirilmedi.

Yeni ayrı cache: `cache/autsl_10_features_wristcrop15_pilot`, 25 dosya, **29,64 MiB**. Manifest: `derivation.method=pose_wrist_fallback_crop`, `version=2`, `box_scale=1.5`, kaynak manifest SHA256. `configs/experiment/autsl_10_wristcrop15_pilot.yaml` yalnızca bu pilotu tanımlar; tam eğitim verisi değildir. Eğitim yükleyicisi eksik train/val dosyalarında durur.

## Kodda yapılanlar

- `src/preprocessing/wrist_fallback_crop.py`: `box_scale` parametresi ve gerçek tam sayı kutu sınırlarını paylaşan `crop_bounds`. Varsayılan 1,00 ölçeğin davranışı korunur; geçersiz ölçek reddedilir.
- `scripts/repair_hand_crops.py`: `--box_scale`, sürüm/ölçek manifesti, kaynak config/manifest uyumu ve pilot ID denetimi. 1,00 ölçek eski sürüm 1 manifestini üretmeye devam eder.
- `src/core/config.py`, `src/data/tid_datamodule.py`: `fallback_box_scale` ve manifest ölçeği eşleşmesi. 1,50 cache'i 1,00/1,75 yapılandırmasıyla sessizce yüklenemez.
- `src/preprocessing/inference_pipeline.py`: sürüm 2 ve aynı kutu ölçeği; eğitim/tahmin politikasının ayrışması önlenir.
- `scripts/audit_wristcrop_pilot.py`: bütün seçili dosyaların ham kare hizası, değişmemesi gereken alanlar, tespitli/yedek kırpım pikselleri, gerçek kutu geometrisi ve sınıf/işaretçi özetleri. Her kayıp el örneği için pano üretir.
- `scripts/diagnose_wristcrop_center.py`: yalnızca iki hatalı örnekte komşu tespitlerden merkez kestiriminin teşhisi; cache yazmaz ve genel onarım algoritması değildir.
- Testler: ölçek büyütmenin kenardaki içeriği geri alması, tespitli kırpımları koruması, geçersiz ölçek, manifest uyuşmazlığı ve ölçekli tahmin/val girdi eşitliği.

## Sayısal denetim

Toplam 400 video karesi, 800 el örneği incelendi:

- **640 tespitli kırpım** ham videodan yeniden kesildi: piksel farkı sıfır. Bunlar temel cache'le de bire bir aynı.
- **159 yedek kırpım** güncel ham kare ve kaydedilen ölçekten yeniden üretildi: piksel farkı sıfır; 94 train, 65 val.
- **1 çözülemeyen örnek**: `train/signer33_sample505_color`, sağ el, sıfır tabanlı örnek 2. Hem el hem poz kayıp; önceki kırpımın taşınması sürüyor.
- Bütün seçili koordinatlar, tespit maskeleri, etiketler ve kare sayıları temel cache'le bire bir aynı.
- Yedek görüntülerde tümü siyah veya önceki kırpımla tamamen aynı görüntü: **0**. Bu, içinde el bulunmasıyla aynı ölçüm değildir; yanlış merkezli giysi kırpımı da siyah olmayabilir.
- Gözlenen elin 21 noktasının tamamının varsayımsal yedek kutuya sığması: gerçek tam sayı kutularıyla **313/639 (%48,98) → 619/639 (%96,87)**. Train 151/320 → 308/320, val 162/319 → 311/319. Önceki rapordaki 309/639 değeri sürekli merkez/yarı kenar geometrisinden hesaplanmıştı; yeni denetim gerçekten kullanılan yuvarlanmış piksel sınırlarını kullanıyor. 1,50 sonucu değişmedi.
- %25'ten fazla dış görüntü dolgusu olan **2** yedek: `signer22_sample106_color`, sol el, örnek 13 ve 15. Kaynak görüntünün alt sınırına yakın eller. Kutunun diğer tespitli elin merkezini de içerdiği **39** örnek var; temas eden/çaprazlanan iki el işaretlerinde bu otomatik kimlik hatası değildir.

Kapsam oranı **tespitli karelerde vekil ölçümdür**; gerçek kayıp karelerdeki el doğruluğunu veya val sınıflandırma doğruluğunu göstermez. Sınıf/işaretçi gruplarının örnek sayıları küçük ve eşit değil. Örneğin model etiketi 2'de 54/60, `signer22` için 25/30 kapsam var; genel ortalama bütün alt grupları garanti etmez.

Kanıt: `outputs/wristcrop15_pilot_qa/audit.json`. Buradaki `visual_review_status` otomatik betik tarafından bilinçli olarak pending yazılır; insan/görsel kararının kaydı ayrı `visual_review.json` dosyasındadır.

## Görsel inceleme ve merkez sorunu

14 panonun **160 kayıp el örneği** incelendi; bunlar 159 yedek + 1 çözülemeyen örnektir. Ham RGB üzerinde gerçek kutular, poz ve mevcut el noktaları; yanında 1,00/1,50 kırpımlar var. Parmak kaybı ve geniş el açıklığında 1,50 kutunun yararı görülüyor. Hızlı geçişlerde bulanıklık sürüyor; bazı temas/örtülme karelerinde tek tek parmakların doğruluğu görüntüden onaylanamıyor. Bu kontrol kare düzeyinde insan anotasyonlu doğruluk ölçümü değildir.

**Belirgin merkez hataları:** `train/signer33_sample505_color`, sağ el, **örnek 5 ve 6**. Eller göğüs önündeyken pozun el noktaları aşağıda kalıyor; 1,50 kutu koyu giysiyi alıyor. Büyütmek bu hatayı çözmüyor. Pano: `missing_08.png`, hücre 2–3. Bütün örnek indeksleri sıfır tabanlıdır.

![İki merkez hatası ve komşu tespitlerle teşhis](C:/Projelerim/BitirmeProjesiBirlesim/BitirmeProjesi/outputs/wristcrop15_pilot_qa/center_diagnostic.png)

Örnek 4 ve 7'de gerçek sağ el tespitleri var. Bu uçların el bileği kendi poz bileğine daha yakın (0,0282 / 0,0151), diğer poz bileğine daha uzak (0,0594 / 0,0680); noktaları sonlu ve görüntü içinde. Bu iki tespit kutusunun merkezleri arasında doğrusal kestirimle yalnızca teşhis görüntüsü üretildi. Örnek 5 ve 6'da merkez mevcut poz merkezinden **88,34 ve 65,71 piksel** ayrılıyor ve kesilen görüntüde eller yeniden görünüyor.

Bu iki kareye bakarak genel merkez onarımı kabul edilmedi. Doğrusal yol hızlı/kavisli harekette gerçek eli kaçırabilir. Aynı 25 videoda regresyon kontrolü şart. Teşhis, el/parmak noktalarını doldurmuyor ve herhangi bir cache'i değiştirmiyor.

Görsel karar: `outputs/wristcrop15_pilot_qa/visual_review.json` → `revise_before_full_cache`.

## Tekrarlanabilir komutlar

Proje kökünde:

```powershell
python scripts/repair_hand_crops.py --dest_cache cache/autsl_10_features_wristcrop15_pilot --box_scale 1.5 --video_ids_file configs/experiment/autsl_10_wristcrop_pilot25.json --max_alignment_error 0
python scripts/audit_wristcrop_pilot.py
python scripts/diagnose_wristcrop_center.py
python -m pytest -q
python -m compileall -q scripts src tests
git diff --check
```

Cache üreticisi var olan dosyaları atlar; `fallback_summary.json` o çağrıda işlenen/atlanan sayıları gösterir. Tekrar koşudan sonra tam kapsamı bu özetin `videos` alanından çıkarmayın; `audit.json` bütün sabit pilot dosyalarını tekrar doğrular. Temiz yeniden üretim gerekiyorsa farklı **yeni** hedef klasör kullanın ve auditor'a `--candidate` yolunu verin.

Günlükler: `outputs/logs/wristcrop15_pilot.log`, `wristcrop15_pilot_audit.log`, `wristcrop15_center_diagnostic.log`, `wristcrop15_tests.log`. Son test sonucu **53/53 başarılı**; Python derleme ve diff boşluk kontrolü başarılı.

## Devam kararı

Tam 1.444 video için ölçekli cache üretilmedi; mevcut temel, wristcrop sürüm 1 ve gap2_v2 cache'leri değiştirilmedi. `autsl_10_wristcrop_screen.yaml` yeni adaya çevrilmedi. Colab eğitimi bu aday için henüz başlatılmamalı.

Sıradaki sınırlı iş: aynı 25 videoda, en çok iki örneklik iç boşluklarda güvenilir aynı-el tespit kutularından **merkez onarımı** adayı geliştir. Yeni yöntem ayrı cache/manifest/sürüm kullanmalı; koordinatları ve gözlenen kırpımları değiştirmemeli. Yukarıdaki iki hata düzelirken diğer örneklerde hedef elden sapma oluşmadığı panolarda kontrol edilmeli. Ayrıntılı devir: `docs/antigravity_preprocessing_devir.md`.
