# Karar: sıfır el karelerinde üst gövde ROI + iki kare hareket koşulu

> **Eğitim sonrası düzeltme, 22 Ekim 2026:** Aşağıdaki karar, eğitim öncesindeki aday seçimidir. Colab'daki aynı seed 7/20 epoch yeni ağırlıklar artık eğitildi: motion2 **176/199 (%88,44)**, strict v1 **192/199 (%96,48)**. Drive sonucu yerelde yeniden üretildi ve eşleşmiş 199 video kontrol edildi. **Motion2 reddedildi; 100 sınıfa aktarılmayacak.** Strict v1 ile 100 sınıf pose-only koşusuna geçiyoruz. [Eğitim sonrası tam karar](AUTSL_10_motion2_egitim_karari_2026-10-22.md).

**22 Ekim 2026.** Ön işleme için seçtiğim yeni deney sürümü **`zero_hand_upperbody_roi_motion2`**. Kaynak `centerrepair_strict` 16 kare/1,5× kırpım/merkez onarımıdır. Yalnız kaynakta **iki elin de tespit edilmediği** karede üst gövde kırpımında ikinci HandLandmarker denemesi yapılır. Yeni el ancak en az **iki bitişik örnek karede** bu durum sürerse eklenir. Kaynakta görülen el, poz, sınıf ve başka kareler aynen kalır. Sağ/sol baskın el hakkında ek varsayım yoktur; tek elde algılayıcının etiketi, iki aynı etiketli elde yalnız belirgin kol izi kullanılır.

## Neden bu özelliği seçtim?

40 önceden seçilmiş zor **train** videosunun 640 örnek karesinde `strict v1` iki eli 242 karede de kaybetmişti. Üst gövde ROI bunların **109**'unda 153 el tespiti üretti. Önceki görsel aday taramasına göre 109 karenin **80'inde etkin el**, 24'ünde nötr, 2'sinde pasif el, 3'ünde belirsiz el vardı. Sekiz sağlam kontrol videosunun kaynakta boş olan tek karesine ROI el eklemedi; mevcut 245 tespitli yuva ve kontrol tensörleri korunur.

Tek karelik yeniden tespitler, işaretin hareketi yerine gürültü taşıyabilir. En az iki ardışık kare kuralı aynı 40 videoda **67 etkin el** karesini korudu; nötr adayları **24→11**, pasif adayları **2→1** indirdi; 3 belirsiz kaldı. Bu seçilmiş zor kümenin 640 karesinin **67'sinde (%10,47)** kaynakta sıfır el varken etkin el üzerinde yeni tespit konumu bulunması demektir. Bu oran 100/226 sınıf geneline veya model doğruluğuna taşınamaz. 21 noktanın her parmak eklemine tam oturduğu bağımsız insan etiketi yoktur.

## Gerçek ve ayrı önbellek sonuçları

`scripts/build_autsl_zero_hand_roi_cache.py` iki el yoksa ROI denemesini **ayrı** cache'e yazdı. İlk 40 videoluk kaynak/çıktı kontrolünde 109 kare ve **153 yeni el kırpımı** ham videoyla piksel düzeyinde eşleşti; kaynakta tespit edilmiş hiçbir el değişmedi; 8 kontrolün tamamı birebir korundu. Makine kaydı: `outputs/autsl_100_drive_review/zero_hand_roi_40train_verification.json`.

Resmî 10 sınıfın tamamında önce koşulsuz sıfır el ROI önbelleği üretildi, sonra `scripts/derive_autsl_zero_hand_motion2_cache.py` ile iki karelik sürüm çıkarıldı:

| Ayrı önbellek | Train, 1.245 video | Val, 199 video | Mevcut eski seed 7 checkpoint ile val |
|---|---:|---:|---:|
| Strict v1 | referans | referans | **192/199 = %96,48**, Macro-F1 0,9645 |
| Sıfır el ROI, tek kareler dahil | 666 kare / 864 el yuvası eklendi | 82 kare / 109 el yuvası eklendi | **191/199 = %95,98**; tek yeni hata `signer1_sample614_color`, 4. karede tekil ROI tespiti |
| **Seçilen motion2** | **269 kare / 369 el yuvası**, 82 video | **26 kare / 42 el yuvası**, 9 video | **192/199 = %96,48**; strict ile 199 tahminin tamamı aynı |

`scripts/audit_autsl_zero_hand_roi_cache.py` tüm 1.444 kaynak ve motion2 NPZ'sinde dosya adı, tensör şekli/türü, poz/sınıf, kurtarılmayan kareler ve eski el tespitlerinin korunmasını denetledi. **Sıfır eksik dosya ve sıfır kaynak gerilemesi**. Bu, model başarısının artacağı kanıtı değil; hareket girdisinin eksik orta karelerde arttığı ve mevcut ağırlıkla val'in bozulmadığı somut sonuçtur. Eski checkpoint ile yapılan karşılaştırma **yeniden eğitim değildir**. İlk tek kareli sürümün 1 video gerilemesi, iki kare kuralını seçmemizin gözlenmiş nedenidir; val setinde bu seçim yapıldığından yeni val sonucu bağımsız test gibi yorumlanamaz.

## Kesin karar ve tek sonraki koşu

**`motion2`yi 10 sınıf eğitim denemesi için seçiyorum.** Strict v1 silinmedi. 100/226 sınıf cache'ine şimdilik dokunulmadı. Bu seçim, 40 zor train videosunda etkin el hareketine ait ardışık noktalar kazandırması ve eski modelin 199 resmî val tahminini bozmaması nedeniyle `ROI'yi her kareye uygula` ve `tek kareli ROI` seçeneklerinden daha iyi ön işleme adayıdır.

Yeni [eğitim config'i](../configs/experiment/autsl_10_zero_hand_roi_motion2_seed7.yaml) mevcut strict seed 7 hibritinin mimarisi, seed'i, 16 karesi ve 20 epoch bütçesiyle eşlenmiştir. Colab'da Drive bağlandıktan ve `/content/BitirmeProjesi` hazırlandıktan sonra:

```bash
bash scripts/colab_train_zero_hand_roi_motion2.sh
```

Betik 1.245 train + 199 val cache sayısını ve manifesti kontrol eder, **yeni model ağırlığını sıfırdan** eğitir, en iyi checkpoint'i resmî val'de değerlendirir ve video bazlı tahminleri Drive'a yazar. Eski strict seed 7'nin **192/199** sonucuyla aynı video ID'lerinde kıyaslanmalı. Yeni model ≥192/199 ve değişen videolarda daha az aktif el hatası verirse motion2'yi 100 sınıf ön işleme adayı olarak taşırız; daha kötüyse strict v1 ile devam ederiz. 10 sınıf zaten kolay olduğundan eşitlik 100 sınıf başarısını ispatlamaz. Bütçesi sınırlı 100 sınıf eğitimlerini bu tek denemenin sonucu görülmeden açmayacağız.

## Colab'a hazır dosyalar ve sınır

Seçilen 1.444 NPZ'lik, **1.765.607.491 bayt** motion2 cache Google Drive Desktop'ta `BitirmeProjesi/cache/autsl_10_features_zero_hand_roi_motion2` altına aktarıldı; **1.444 dosyanın SHA-256'sı yerel kaynakla doğrulandı**. Config ve [Colab betiği](../scripts/colab_train_zero_hand_roi_motion2.sh) de Drive projesine birebir kopyalandı. `outputs/autsl_zero_hand_roi_10_review/drive_copy_verification.json` aktarım kanıtı; Drive Desktop'ın buluta eşitlemesi Colab'dan ayrıca görünür olmalıdır.

Üretim/canlı çıkarım kodu bu yeni yöntemi henüz çalıştırmıyor. Bu aşama eğitim denemesidir; model kazancı doğrulanırsa aynı ön işlemeyi canlı çıkarıma eşleyeceğiz. Resmî test açılmadı. Val etiketleri cache üretimine girmedi, ancak eski checkpoint'in val sonucu son iki kare kuralı kararını etkiledi; bu nedenle aynı val ile ölçülen fark bağımsız test değildir. Model eğitimi bu bilgisayarda yapılmadı.
