# Ön işleme çalışma ve devir günlüğü

## 22 Ekim 2026 — İlerleme raporu DOCX güncellemesi

`C:\Projelerim\BitirmeProjesiBirlesim\Bitirme_Projesi_Ilerleme_Raporu.docx` belgesine “10. Preprocessing Geliştirme ve AUTSL Ön İşleme Kararı” eklendi. ASL Citizen bölümleri tarihsel olarak bırakıldı; kapakta AUTSL güncelleme tarihi belirtildi. Yeni bölüm resmî train/val protokolünü, 16/36 kare, kısa boşluk, canlı bilek kırpımı, 1,5× kutu, VIDEO/ROI ve motion2 denemelerini karşılaştırır; iki seed ve pose-only sonuçlarını aynı 199 val videosunda açıkça ayırır. Strict v1'in sabitlenmesi ve 100 sınıf pose-only sonraki adımı kaydedildi. DOCX 13 sayfa olarak Word PDF dışa aktarımıyla görsel kontrol edildi; yeni tablolarda satır kesilmesi yok ve ZIP yapısı hatasız. Paketlenmiş `render_docx.py` Windows ortamında `soffice.exe` bulunmadığı için çalışmadığından görsel denetimde yerel Word kullanıldı.

## 22 Ekim 2026 — Drive'daki motion2 eğitimi ve ön işleme kararı

Yeni Colab sonuçları Drive Desktop'tan okundu; `best_model.pt` SHA-256'sı yerel kopyayla eşit (`824f8518bd624072f355ca90d0c3b00dac0187c58d2bf93bd4a078c47e0140ab`). Checkpoint, config ve 199 val tahmini kontrol edildi. Yerel tekrar değerlendirme Top-1 %88,44, Top-5 %100, Macro-F1 0,88414 ve 199/199 aynı tahmin etiketi verdi. Eşleşmiş strict seed 7 **192/199**, motion2 **176/199**; strict-doğru/motion2-yanlış 17, ters yönde 1. Değiştirilen 9 val videosunda 2 yeni hata, değişmeyen 190 videoda net 14 hata artışı. Motion2 aynı 20 epoch bütçesinde reddedildi; daha fazla ROI ayarı veya 100 sınıf motion2 cache'i üretmeyeceğiz. Strict 100 sınıf Drive cache'i yeniden 12.405 train + 1.963 val, sıfır bayt NPZ yok. Sıradaki iş strict + pose-only 100 sınıf temel modelin maliyet/manifest kontrolü ve tek Colab koşusu. Test kapalı. [Tam rapor](AUTSL_10_motion2_egitim_karari_2026-10-22.md).

## 20 Ekim 2026 — 226 sınıf RGB mimari araştırması

Kullanıcı 10 sınıfı pilot, 226 sınıfı ana hedef olarak yeniden vurguladı.
AUTSL/SAM-SLR ve torchvision model kartları incelendi. `hand_cnn_encoder.py`
ConvNeXt'in son ön eğitimli normalizasyonunu koruyacak şekilde düzeltildi;
`efficientnet_v2_s` seçeneği eklendi. Ayrı
`configs/experiment/autsl_226_rgb_convnext_tiny.yaml` ve testler yazıldı.
Bu yalnız kod/config hazırlığı; 226 strict cache ve model eğitimi henüz yok.
Kademeli RGB araştırma planı [raporda](rgb_226_secici_uzman_arastirmasi_2026-10-20.md).
ConvNeXt ve EfficientNetV2-S özellik çıkarımı testleri, config/model kurma
kontrolü ve tam proje testleri geçti (**69/69**); `git diff --check` temiz.

## 20 Ekim 2026 — pose-only seed 7 çıktı denetimi

Masaüstündeki beş pose-only Colab çıktısı `outputs/colab_poseonly_seed7_review/`
altına kopyalandı. 199 resmî val ID/etiketi, 10 olasılık vektörü ve Macro-F1
bağımsız hesapla doğrulandı. 1.245 train / 199 val strict cache tam;
işaretçi/video ID'leri ayrık ve train–val arasında birebir 16×75×3 landmark
tekrarı yok. Pose-only 198/199, hibrit 192/199; işaretçi 11 30/30.
Grafikte en iyi val 17. epoch. Colab eğitimi bu yerel incelemede yapılmadı,
resmî test açılmadı. `best_model.pt` ve tam Colab logu pakette yok.
[Tam analiz ve sıradaki seed 42 deneyi](poseonly_seed7_sonuclari_2026-10-20.md).

Bu dosya Codex ve Antigravity'nin aynı duruma göre devam etmesi için tutulur.
Model eğitimi Colab'da yapılacak; bu çalışma yerel ön işleme ve kalite denetimidir.

## 19 Ekim 2026 — ölçekli bilek kırpımı pilotu

Başlangıç kanıtı: `docs/preprocessing_bagimsiz_inceleme_2026-10-19.md`.
Önceki incelemenin 25 videosu sabitlenecek. Test açılmayacak.

Plan:

1. Yalnızca kayıp el karelerinin kutu kenarına 1,50 ölçek parametresi ekle.
2. Ölçeği manifest, yükleyici ve tahmin akışında doğrula; sürüm 1 sonuçlarını koru.
3. 25 videoluk ayrı cache üret; tespitli kırpımları, koordinatları ve maskeleri temel cache'le karşılaştır.
4. Her kayıp el karesi için ham RGB + gerçek kutu + eski/yeni kırpım panoları üret.
5. Görsel bulgulara göre kabul veya revizyon kararı ver. Tam cache'e genişletme yalnızca bu karardan sonra yapılacak.
6. Sonuç, komutlar, kalan işler ve Colab'a aktarılacak dosyaları bu günlüğe ve yol haritasına yaz.

Tamamlandı: ölçek parametresi, sürüm 2 manifest desteği, cache/yapılandırma ölçeği
eşleşmesi ve ortak tahmin akışı. Yeni sabit liste/config, üretim/audit betikleri,
merkez teşhisi ve testler kaydedildi. Ayrıntılı dosya listesi ve komutlar:
`docs/wristcrop15_pilot_raporu.md`.

Üretim komutu:

```powershell
python scripts/repair_hand_crops.py --dest_cache cache/autsl_10_features_wristcrop15_pilot --box_scale 1.5 --video_ids_file configs/experiment/autsl_10_wristcrop_pilot25.json --max_alignment_error 0
python scripts/audit_wristcrop_pilot.py
python scripts/diagnose_wristcrop_center.py
python -m pytest -q
```

Sonuç: 25/25 dosya, 29,64 MiB; 640 tespitli ve 159 yedek kırpım ham kareden
yeniden doğrulandı, piksel farkı sıfır. Koordinatlar/tespit maskeleri/etiketler
değişmedi. Gerçek kutu sınırlarıyla vekil 21 nokta kapsamı 313/639 → 619/639.
53/53 test, Python derleme ve Git diff boşluk kontrolü başarılı.

Görsel kontrol: 14 panodaki 160 kayıp örnek incelendi. 1 çözülemeyen taşınmış
kırpım, 2 yüksek dolgu ve 39 diğer-el-merkezi içerme işareti kaydedildi. Örtüşme
sayısı yanlış el hata sayısı değildir. `signer33_sample505_color` sağ örnek 5/6'da
poz merkezi eli kaçırıyor. 1,50 büyütme tek başına bu iki hatayı çözmedi.

Ek teşhis: 4/7'deki el tespit kutusu merkezleri arasında doğrusal kestirimle
iki örneğin güncel görüntüsünde eller yeniden kadraja giriyor. Bu yalnızca
teşhis görselidir; genel onarım politikası veya cache değişikliği uygulanmadı.
`outputs/wristcrop15_pilot_qa/center_diagnostic.png` ve JSON'u kaydedildi.

**Karar: tam cache'e genişletmeden merkez onarımını pilotta sınamak.**
Tam 1.444 video için yeni ölçekli cache oluşturulmadı; eğitim ekran config'leri
yeni adaya çevrilmedi. Colab eğitimi, dosya yükleme veya bulut çalıştırma yapılmadı.
Mevcut cache'ler ve önceki inceleme kanıtları korundu.

Otomatik kanıt: `outputs/wristcrop15_pilot_qa/audit.json`.
Görsel karar: aynı klasörde `visual_review.json` (`revise_before_full_cache`).
Koşu günlükleri: `outputs/logs/wristcrop15_*`.
Antigravity için sınırlı devam işi: `docs/antigravity_preprocessing_devir.md`.
Yol haritası, ana ChatGPT rapor indeksi ve önceki raporun devam bağlantısı güncellendi.

## Merkez onarımı pilotu — başlatıldı

Kullanıcı pilot geçerse tam train/val ön işlemeye devam edilmesini yetkilendirdi.
Yeni aday sürüm 3: yalnızca kısa, iki güvenilir aynı-el tespitiyle çevrili iç
boşluklar; uçlar arası hareket sınırı ve mevcut poz merkeziyle uyuşmazlık eşiği.
Kararlı poz merkezleri korunacak. Eşikler manifestte, merkez kaynağı ayrı kodda
saklanacak; parmak/eklem koordinatları değiştirilmeyecek. İlk cache ayrı 25
videoluk pilot olacak. Görsel karar ve tam koşu sonucu bu bölümün devamına yazılacak.

İlk politika (`min_disagreement_ratio=0.75`) pilotta iki bilinen hatayı düzeltti;
diğer 158 kayıp örneğin görüntüsü aynı kaldı. 1.444 videoluk ilk tam cache üretildi,
tam ham kare/piksel denetimi geçti. Ancak 7 merkez değişikliğinin görsel kontrolü,
`signer38_sample58_color` sol 11 ve `signer40_sample243_color` sol 12'de iki
gerileme gösterdi: hızlı/kavisli hareket doğrusal kestirimden sapıyor.
Bu ilk tam aday eğitim için reddedildi; cache ve görseller silinmedi.

Kural daraltıldı: uyuşmazlık en az **bir tam temel el kutusu kenarı** olmalı
(`min_disagreement_ratio=1.0`). Diğer eşikler aynı. Her iki ayar JSON dosyalarında
saklandı: `configs/experiment/center_repair_exploratory.json` ve
`configs/experiment/center_repair_strict.json`. Üreticiye `--center_policy`
seçeneği eklendi; tekrar üretimde eşikler varsayılanlara bırakılmamalı.

Dar kuralın ayrı 25 videoluk pilotu ilk kabul edilen pilotla bütün NPZ alanlarında
bire bir aynı çıktı; aynı iki olumlu değişiklik korunuyor. İkinci tam cache
`cache/autsl_10_features_centerrepair_strict` altında üretiliyor. Tam ham kare
denetimi ve bütün yeni merkezlerin görsel kontrolü bitince nihai sonuç yazılacak.

## Merkez onarımı ve tam ön işleme — tamamlandı

Nihai aday `cache/autsl_10_features_centerrepair_strict`: **1.245 train + 199 val**,
16 kare, yaklaşık 1,647 GiB. Ham kaynakta bütün 1.444 video yeniden açıldı;
36.587 tespitli ve 9.619 yedek kırpım piksel düzeyinde doğrulandı. Nokta koordinatları,
maskeler, sınıf kimlikleri ve gözlenen kırpımlar korunuyor.

Son kural 5 merkezi değiştirdi: signer33/sample186 sağ 2–3, signer33/sample505
sağ 5–6, signer41/sample269 sol 10. Beşi de görsel incelendi; eller yeniden
kadrajda. İlk koşudaki iki hızlı hareket gerilemesi son kuralda değişmeden
poz kırpımında kaldı. Görsel karar `accepted_for_short_colab_comparison`.

Kalanlar: 1 taşınmış + 1 ilk tespit öncesi görüntü; 89 yüksek dolgu işareti.
En yüksek 12 dolgu örneği ayrıca incelendi: kaynak kadrajın altında/dışında
dinlenen eller görüldü. Tüm kayıp eller için parmak doğruluğu iddia edilmedi.
Vekil kapsam tam veride %96,17; düşük işaretçi grupları rapora kaydedildi.

63/63 test geçti. `inference_pipeline` ile cache yükleyicisinin gerçek onarılan
videodaki normalize tensörleri bire bir eşleşti (cache'lenmiş detector çıktıları
kullanılarak yalnızca ön işleme karşılaştırıldı). Model tanıma sonucu ölçülmedi.

Colab hazırlığı: iki 20 epoch config'i aynı model/optimizasyon/augmentation/seed
ile denetlendi. İki cache'in manifestleri ve train/val dosyaları için toplam
2.890 SHA256 kaydı üretildi ve doğrulandı. Dört resmî CSV, ayrı
`outputs/colab_training_metadata` klasörüne kopyalandı; ham RGB taşımadan cache
eğitimi için metadata yüklemesi doğrulandı. Test videosu/değerlendirmesi kullanılmadı.

Yeni/güncellenen iş dosyaları: ortak fallback geometri/politikası, config ve cache
eşleşmesi, üretici `--center_repair/--center_policy`, kaynak-3 kalite sayımı,
`audit_center_repair.py`, `check_colab_inputs.py`, merkez/girdi/manifest testleri,
iki politika JSON'u ve iki `autsl_10_centerrepair*.yaml` config'i. Tam dosya,
komut, ölçüm ve sınırlama kaydı `docs/merkez_onarimi_tam_onisleme_raporu.md` içinde.

Sonraki iş **Colab'da iki kısa eğitim**. Rehber:
`docs/colab_onisleme_karsilastirmasi.md`. Antigravity devir notu, yol haritası,
ana rapor ve önceki pilotların devam bağlantıları güncellendi. Burada eğitim,
Drive aktarımı veya bulut çalıştırma yapılmadı. Eski keşif cache'i reddedilmiş
deney kanıtı olarak korunuyor; eğitim için yalnızca `_strict` aday seçilecek.

## 20 Ekim 2026 — kullanıcının Colab logları incelendi

Kullanıcı iki 20 epoch eğitim ve iki resmî val değerlendirme logu gönderdi.
Her iki eğitim 1.245 train + 199 val, seed 42, Tesla T4 üzerinde tamamlanmış.
İki ayrı `best_model.pt` val değerlendirmesinde temel 173/199 (%86,93),
strict aday 178/199 (%89,45); fark 5 video. Eğitim içi en iyi val temel
174/199 (%87,44), aday 177/199 (%88,94). İki kol da en iyi epoch 20'de.
Bağımsız değerlendirmedeki birer video farkı tekrar ölçülmeden kesin bir
nedenle açıklanmadı; eğitim içi val CUDA AMP kullanırken eski değerlendirme
kullanmıyordu.

Notebook karşılaştırma hücresi `macro_f1` anahtarıyla olmayan alanı okuyup
0,0000 gösteriyordu; `f1_macro` ve doğrudan anahtar erişimiyle düzeltildi.
Eski `MetricCalculator` sınıf başına F1 ortalaması yerine makro precision ve
recall'un harmonik ortalamasını hesaplıyordu. Macro ve weighted F1 doğru
sınıf bazlı hesaplamaya çevrildi. Değerlendirme CUDA AMP ayarı eğitim içi val
ile eşitlendi. `train.py` sonundaki otomatik test çağrısı kaldırıldı; eski logda
bu çağrı yükleyici kurulmadığı için uyarıyla atlanmış, test ölçülmemiş.

Notebook ve Colab rehberindeki değerlendirme dosya adları `_v2.json` oldu;
eski sonuçlar korunacak. `tests/test_metrics.py` gerçek F1 regresyon testi
eklendi: hedefli 6/6, tüm proje 64/64 test geçti. Notebook JSON doğrulaması,
Python derleme ve `git diff --check` başarılı. `evaluate.py` varsayılan bölümü
yanlışlıkla test açılmasını önlemek için `val` oldu; nihai test için açık
`--split test` gerekir.
Bu oturumda Colab'a bağlanılmadı, eğitim veya yeniden val değerlendirmesi
başlatılmadı. Güncel kod Colab'a aktarılarak iki mevcut checkpoint tekrar
val üzerinde ölçülecek. Sonra aynı 20 epoch bütçesi seed 7 ve 2026 ile
tekrarlanacak. Test kilitli kalacak.

## 20 Ekim 2026 — masaüstü Drive çıktılarının ayrıntılı incelemesi ve tekrar hazırlığı

`C:\Users\Bünyamin\Desktop\Güncel Drive Çıktıları` içindeki altı dosya
okundu; iki JSON, iki 20 satırlık CSV ve iki PNG grafik. Dosyalar açıklayıcı
adlarla `outputs/colab_seed42_review` içine kopyalandı. CSV/grafik eşlemesi
son epoch val noktalarıyla doğrulandı. Yeni JSON'lar AMP ve doğru sınıf bazlı
F1 hesaplamasıyla uyumlu: temel 174/199 (%87,44), F1 0,8718487;
strict aday 177/199 (%88,94), F1 0,8885203. Fark 3 doğru video ve
0,01667 macro-F1. Adayın son val kaybı 0,6155, temelin 0,6990. Her iki
koşunun en iyi val epoch'u 20/20. Önceki AMP'siz 173/199 ve 178/199
rakamları güncel karar sayıları olarak kullanılmayacak.

Model akışı incelendi: 16 kare, 225 nokta değeri, iki 160×160 el kırpımı,
paylaşılan/dondurulmuş MobileNetV3-Small, çapraz dikkat füzyonu,
TCN+Transformer, 10 sınıf. `get_num_parameters` paylaşılan omurgayı iki kez
sayarak 5.969.610 yazıyordu; fonksiyon düzeltildi. Gerçek benzersiz parametre
4.894.890, eğitilen 3.959.434. Önceki checkpoint/tahminler değişmedi.
Kaynak klasörde checkpoint veya video bazlı tahminler yoktu; ağırlık içeriği
ve hangi videoların düzeldiği denetlenemedi. `evaluate.py` içine isteğe bağlı
`--predictions_output` eklendi; video ID, etiket, tahmin, güven ve checkpoint
SHA256 kaydedilecek. Test bölümü varsayılanı val olarak kalıyor.

`scripts/prepare_seed_repeats.py` ile seed 7 ve 2026 için dört config
oluşturuldu. Her çifte `check_colab_inputs.py` yerelde uygulandı:
1.245 train/199 val, 16 kare, eşit model/augmentation/eğitim bütçesi,
metadata yükleme ve manifest doğrulandı. Bu iki tekrar preflight'inde
hash kontrolü çalıştırılmadı; önceki seed42 cache hash kontrolü ayrı kayıtlı.
Güncel çalışma komutları `docs/colab_onisleme_karsilastirmasi.md` içinde.
Hedefli model/metric/tahmin testleri 14/14, tüm proje testleri 66/66 geçti;
derleme ve `git diff --check` başarılı. Altı kaynak/kopya SHA256 değeri
bire bir eşleşti.
Bu oturumda eğitim veya buluta yükleme yapılmadı. Sıradaki iş Colab'da seed42
tahminlerini almak, seed7 çiftini çalıştırmak ve sonuçları incelemek.

## 20 Ekim 2026 — ZIP içindeki seed 7 ve eşleştirilmiş val tahminleri

Kaynak `C:\Users\Bünyamin\Desktop\chatgptye_gonderilecekler.zip` (SHA256
`7FBA74C872D39AD195C35689619747B506BEDA15848B85A34215FBD22FA7681E`)
envanterlendi; 10 düz giriş yolu güvenli biçimde
`outputs/colab_seed7_review/` klasörüne alındı. İki seed 42 tahmin dosyası,
seed 7'nin iki metrik JSON'u, iki tahmin JSON'u, iki 20 epoch CSV'si ve iki
PNG grafiği korundu. ZIP'te checkpoint, preflight veya cache manifesti yok.

`scripts/analyze_val_pairs.py` yazıldı: yalnız `val` kabul eder, 199 video ID'si
ve gerçek etiketleri resmi `outputs/colab_training_metadata/val_labels.csv`
ile doğrular, eşleştirilmiş doğru/yanlış değişimini sınıf/işaretçi kırılımıyla
çıkarır. Seed 42 ve 7 için JSON kanıtlar `seed42_pair_analysis.json` ve
`seed7_pair_analysis.json` olarak üretildi. Dört tahmin dosyasının Macro-F1
ve Top-1 ölçümleri bağımsız yeniden hesaplamayla metrik JSON'larına uydu.
İki eğitim grafiği ve epoch CSV'leri incelendi.

Sonuç: seed 42 base 174/199, strict 177/199 (yalnız base doğru 16, yalnız
strict doğru 19); seed 7 base 188/199, strict 192/199 (yalnız base 2, yalnız
strict 6). Seed 7 strict'in en iyi checkpoint'i epoch 19; epoch 20'de 187/199'a
iniyor. İşaretçi 11 seed 42'de 28/30→22/30, seed 7'de 27/30→26/30.
`signer11_sample570_color` ve `signer1_sample1012_color` dört modelde de
yanlış. Bu iki video için ham kare ve strict el kırpımı panoları, ayrıca
işaretçi 11'in 342/447/600 örnekleri için ham kare panoları üretildi; hepsi
aynı çıktı klasöründe. Etiket yanlışlığı iddia edilmedi, veri taşınmadı.

Raporlar güncellendi: `docs/colab_seed7_sonuclari_2026-10-20.md`, ana
`chatgpt raporu.md`, yol haritası, Colab rehberi ve Antigravity devir notu.
Sıradaki deney seed 2026'nın iki 20 epoch Colab koşusu. Bu oturumda model
eğitimi, Drive yüklemesi ve resmi test değerlendirmesi yapılmadı.
`tests/test_analyze_val_pairs.py` 1/1, tüm proje 67/67 test geçti;
`compileall` ve `git diff --check` başarılı.

## 20 Ekim 2026 — seed 2026 zorunluluğu kaldırıldı

Kullanıcının itirazıyla deney sırası yeniden değerlendirildi. Seed 42/7
config diff'inde yalnız deney adı, çıktı yolu ve `seed` değişiyor. Strict,
iki eşleştirilmiş seed'de +3/+4 doğru video verdi; bu onu **10 sınıf geliştirme
baz çizgisi** olarak kullanmak için yeterli. Seed 2026 yeni model fikrine
başlamanın zorunlu kapısı olmaktan çıkarıldı, daha sonraki dayanıklılık
kontrolü olarak tutuldu. Seed 7 strict'te yedi hata var; dördü işaretçi 11.
Yol haritası, ana rapor, ayrıntılı ZIP raporu, Colab rehberi ve Antigravity
notu bu sıraya göre düzeltildi. Bu değişiklik yeni eğitim veya test sonucu
değil, mevcut kanıtın daha uygun deney planıdır.

## 20 Ekim 2026 — işaretçi 11 dayanıklılık ve canlı ürün incelemesi

Kullanıcının işaretçi 11'in kürklü/koyu kıyafeti, mor koltuğu ve oturur
pozuna dair gözlemi ham val karelerinde kontrol edildi. MediaPipe'nin düşük
kontrast/örtüşme sınırlaması ve AUTSL'de RGB+iskelet/temporal bağlamın
literatürü birincil kaynaklardan incelendi; bağlantılar
`docs/signer11_dayaniklilik_analizi_2026-10-20.md` içinde.

`scripts/audit_val_error_quality.py` eklendi ve seed 7/42 strict tahminleri
199 resmî val cache dosyasıyla eşlendi. JSON'lar
`outputs/colab_seed7_review/seed7_strict_quality_audit.json` ve
`seed42_strict_quality_audit.json`. Seed 7 işaretçi 11 26/30, diğerleri
166/169; el-kare tespiti sırasıyla 814/960 (%84,79) ve 4574/5408
(%84,58). İşaretçi 11'de dört hatada 107/128 tespit, 26 doğruda 707/832.
Kayıp tespit oranı tek başına hataları açıklamadı; tespit edilmiş ama yanlış
parmak noktalarının doğruluğu bu sayıdan ölçülemez. Yedi yanlışın tamamı
`scripts/inspect_preprocessing.py --video_id ... --frame_counts 16` ile
ham/landmark ve el kırpımı panolarına dönüştürüldü; çıktılar
`outputs/colab_seed7_review/error_panels/` altında. `447` ve `600` ayrı,
benzer okul videolarıdır; cache dizileri özdeş değil. Etiketler değiştirilmedi,
işaretçi 11 çıkarılmadı, resmî test açılmadı.

`configs/experiment/autsl_10_poseonly_seed7.yaml` oluşturuldu. Seed 7
strict config'inden deney/çıktı adı ve yalnız `model.use_rgb: false`
farklı; cache/16 kare/20 epoch aynı. Yerel yükleyici 1.245 train/199 val
dosyasını açtı ve pose-only modelin ileri geçişi `(16,10)` çıktı verdi.
Bu bir **Colab'da yapılacak teşhis koşusu**, doğruluk iyileşmesi henüz yok.
Yeni dayanıklılık raporu, yol haritası, ana rapor, Colab rehberi ve
Antigravity notu güncellendi. Canlı kullanımda Top-5 ve yüksek tek
checkpoint val skorunun yeterli olmadığı, yüksek güvenli hatalar ve gerçek
kamera/kişi/gecikme değerlendirmesi kayıt altına alındı.
Olası füzyonun gerçek sınıf skorlarını görebilmesi için
`src/evaluation/evaluator.py` tahmin JSON'una `class_probabilities` ekliyor;
eski alanlar ve ağırlıklar değişmiyor. `tests/test_evaluator_predictions.py`
alan uzunluğu, toplamı, argmax ve üst güven tutarlılığını kontrol edecek
şekilde genişletildi; hedefli 2/2 test ve Python derleme geçti. Eski ZIP
tahminlerinde bu skor alanı yok; hibrit checkpoint Colab'da yeniden yalnız
val üzerinde değerlendirilebilir.

## 21 Ekim 2026 — 100 sınıf eğitim öncesi kalite kapısı hazırlandı

Kullanıcı 10 sınıf v1'i kabul etti; 100 sınıfta yalnız train/val ön işleme,
parmak/bilek hareketi ve kaçırma nedenlerinin incelenmesini istedi. Model
eğitimi başlatılmadı. `scripts/prepare_autsl_100_preprocessing.py` önceki 10
ClassId'yi koruyup train'deki kalan sınıflardan seed `20261021` ile 90 tane
seçti. Liste/etiket eşlemesi ve sınıf sayıları
`data/autsl/autsl_100_preprocessing_selection.json`; toplam 12.405 train,
1.963 val. Birer video/sınıf pilotu, base16 ve strict config'ler üretildi.
Yerel resmi train/sınıf-listesi CSV hash'leri Desktop AUTSL köküyle eşleşti.

İki aşamalı Colab komutları, beklenen dosya/manifest sayıları, tam NPZ taraması,
ham kare/landmark panoları, hata taksonomisi, belirsiz sebep işaretlemesi,
güvenli val/test sınırı ve teslim dosyaları
`docs/AUTSL_100_onisleme_kalite_kapisi_2026-10-21.md` içine yazıldı.
100 sınıf cache henüz üretilmedi; val tahmini ve 100 sınıf eğitim sonucu yok.
`scripts/audit_autsl_preprocessing_motion.py` tüm NPZ'lerin kapsama,
etiket/şekil, eksik seri, bilek karşılaştırması ve ani parmak/bilek sıçraması
bayraklarını çıkarıyor. 10 sınıf strict cache üzerinde duman testinde
1.245 train + 199 val dosyası tam, geçersiz dosya sıfır; bayraklar yalnız
görsel inceleme önceliğidir, hata nedeni olarak raporlanmaz.
