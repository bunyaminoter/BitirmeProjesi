# Motion2 eğitimi sonrası ön işleme kararı

**22 Ekim 2026 — karar: `zero_hand_upperbody_roi_motion2` 100 sınıfa taşınmayacak.** Ana ön işleme `centerrepair_strict` v1 olarak kalacak. Yeni modelin el tespiti eksik karelerde daha fazla nokta üretmesi, aynı bütçeli eğitimde sınıflandırma kazancı sağlamadı. Bu adayın cache'i, checkpoint'i ve tahminleri inceleme için korunuyor; aktif model veya 100 sınıf cache'i yerine geçmiyor.

## Deney ve bağımsız kontrol

İki hibrit RGB+pose koşusu aynı resmî 10 AUTSL sınıfını, **1.245 train / 199 val** videosunu, seed 7'yi, 16 kareyi, 20 epoch'u, MobileNetV3-Small dondurulmuş RGB kolunu ve diğer eğitim ayarlarını kullandı. Eski koşu `autsl_10_centerrepair_seed7`, yeni koşu `autsl_10_zero_hand_roi_motion2_seed7` idi. Fark cache'ti: strict v1 yerine sıfır el karelerinde üst gövde ROI ve en az iki bitişik örnek kare koşulu. Yeni cache train'de 82 videonun 269 karesine, val'de 9 videonun 26 karesine el tespiti eklemişti. Önceki audit, eski tespit ve tensörlerin korunduğunu göstermişti.

Drive Desktop'taki `outputs/autsl_10_zero_hand_roi_motion2_seed7` klasöründen `val_metrics.json`, `val_predictions.json`, `history.csv`, eğitim grafiği ve `best_model.pt` okundu. Checkpoint SHA-256: `824f8518bd624072f355ca90d0c3b00dac0187c58d2bf93bd4a078c47e0140ab`; Drive ile yerel kopya hash'i aynı. Checkpoint config'i seed 7, 20 epoch, aynı 10 sınıf ve motion2 cache'i gösteriyor. En iyi checkpoint 20. epoch sonunda (kayıtta sıfır tabanlı `epoch=19`) seçilmiş. Yerelde 199 val videosunun tamamı aynı cache ve checkpoint ile yeniden değerlendirildi: Top-1, Top-5 ve Macro-F1 Drive sonuçlarıyla aynı; **199 video ID'sinin gerçek ve tahmin etiketleri birebir aynı**. İki değerlendirmenin olasılık değerlerinde en fazla 0,00271 fark var; etiket sıralaması değişmedi. Resmî val ID/etiket eşleşmesi `scripts/analyze_val_pairs.py` ile kontrol edildi.

| Aynı 199 resmî val videosu | Doğru / 199 | Top-1 | Macro-F1 | Top-5 | En iyi epoch |
|---|---:|---:|---:|---:|---:|
| Strict v1, hibrit seed 7 | 192 | %96,48 | 0,96455 | %100 | 19 |
| Motion2, hibrit seed 7 | 176 | %88,44 | 0,88414 | %100 | 20 |
| Motion2 − strict | **−16 video** | **−8,04 yüzde puan** | **−0,08041** | 0 | — |

Eşlenmiş karşılaştırmada iki model 175 videoyu doğru, 6 videoyu yanlış bildi. Strict'in doğru/motion2'nin yanlış olduğu **17** video, ters yönde **1** video var. Exploratory exact McNemar iki taraflı p≈0,000145; bu analiz aynı val üzerinde aday seçme sürecinden bağımsız bir test değildir. Video bazlı kayıt: `outputs/autsl_zero_hand_roi_10_review/trained_motion2/paired_vs_strict_seed7.json`.

Önemli ayrım: Motion2'nin değiştirdiği 9 val videosunda strict **9/9**, motion2 **7/9**. İki yeni hata `signer11_sample598_color` ve `signer35_sample596_color`; hiçbir düzelme yok. Değişmeyen 190 val videosunda strict **183/190**, motion2 **169/190**. Bu gruptaki fark **15 yeni hata ve 1 düzelme**, yani net −14. Bu durum yalnızca yeni karelerin kendi tahminlerini bozmasından ibaret değil; 82 train videosundaki girdilerin değişmesi ve aynı seed'li eğitimin farklı yola girmesiyle tüm modelin öğrenmesi değişti. Tek koşu ile kesin nedensel pay ayrıştırılamaz.

En büyük sınıf gerilemesi özgün ClassId **106 (`kardes`)**: strict 20/20, motion2 13/20. Signer 35: 29/30→24/30; signer 11: 26/30→23/30. Motion2 son epoch train accuracy %86,77, val accuracy %88,44; strict koşuda en iyi 19. epoch val %96,48, 20. epoch train %95,94. Motion2 val eğrisi son epoch'ta hâlâ yükseliyor; 20 epoch sonrası asla yetişmeyeceği söylenemez. Fakat eldeki **eşit eğitim bütçesinde** güçlü gerileme ve yalnız 9 val videosunda doğrudan etki olması, onu 100 sınıfa genişletmenin maliyetini haklı çıkarmıyor. Ek epoch/seed veya yeni ROI ayarı için Colab harcamıyoruz.

## Sonraki tek yol

1. **Ön işlemeyi sabitle:** 16 kare `centerrepair_strict` v1. Tamamlanmış 100 sınıf Drive cache'i `cache/autsl_100_features_centerrepair_strict` içinde bugün yeniden sayıldı: **12.405 train + 1.963 val**, sıfır bayt NPZ yok; manifest 100 sınıf, 16 kare, 160×160 crop ve strict merkez onarımını gösteriyor. Bu dosyalar model eğitimine hazır bir temel; önceki görsel/CRC audit sınırları ayrıca [100 sınıf raporunda](AUTSL_100_onisleme_sonuc_analizi_2026-10-22.md).
2. **İlk pahalı 100 sınıf koşusu:** Strict cache ile **pose-only** modeli eğit. Gerekçe: aynı strict 10 sınıfta seed 7 pose-only 198/199, hibrit 192/199; 100 sınıfta parmak ayrımlarını kaçırıp kaçırmadığını gerçek hata matrisi gösterecek. Bu, 100 sınıfta pose'un üstünlüğünü önceden kanıtlamaz. Sınıf listesi ve resmî train/val protokolü sabit; test kapalı. Başlamadan cache manifesti, 1 batch ileri/geri ve Colab maliyeti ölçülüp makul üst epoch ve checkpoint devamı ayarlanacak. Sırf 10 sınıf koşusu 20 epoch sürdü diye 100 sınıfı 20 epoch'ta kesmeyeceğiz.
3. **İlk sonucun karar kapısı:** En iyi checkpoint, val video olasılıkları, Top-1/Top-5/Macro-F1, sınıf ve işaretçi bazlı hatalar, confusion matrix ve 16 karelik kaynak panoları incelenecek. Parmak biçimi/renk görünümü açıkça eksikse ikinci tam 100 sınıf koşusu için eğitilebilir RGB+pose adayı seçilebilir. Sorun düşük kaliteli noktalar veya yetersiz öğrenme ise önce veri/optimizasyon sorunu çözülür. İlk model yeterliyse ikinci 100 sınıf hakkı 226 sınıfa saklanır. `motion2`, ikinci pahalı koşu adayı değildir.
4. **Ölçekleme:** 100 sınıf bulgusu ile nihai 226 sınıf strict train/val cache'i ve seçilen modeli hazırla. Resmî test yalnız mimari ve bütün kararlar val'de dondurulduktan sonra kullanılır. Gerçek zamanlı mobil başarı ayrıca farklı kullanıcı ve kamera akışında ölçülür.

**Sınır:** 10 sınıf seçilmiş bir pilot; burada gözlenen −8,04 yüzde puan 100/226 sınıfa sayısal tahmin olarak taşınamaz. Bu karar, mevcut Colab bütçesi ve doğrulanmış 10 sınıf sonucu için aday elemesidir. Strict v1 de 226 sınıfta en iyi ön işleme olduğunu henüz kanıtlamış değildir; fakat şu anda en sağlam ve en ucuz referanstır.
