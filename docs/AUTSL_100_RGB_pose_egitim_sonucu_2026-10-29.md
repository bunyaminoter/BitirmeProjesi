# AUTSL 100 sınıf RGB + pose eğitimi: sonuç ve karar

**29 Ekim 2026.** Resmî AUTSL `val` içindeki sabit 100 sınıf ve 1.963 video üzerinde, `centerrepair_strict` v1 ve seed 7 ile eğitilen ConvNeXt-Tiny + pose modeli incelendi. **Karar:** RGB + pose tek model, pose-only'ye göre yalnız **10 ek doğru** verdiği için üstünlüğü kanıtlanmış sayılmıyor. Buna karşılık, iki modelin **sabit 0,5/0,5 olasılık ortalaması** hiçbir yeni eğitim yapmadan **1.784/1.963 = %90,88 Top-1** elde etti. Bu güçlü bir tamamlayıcılık bulgusudur; 226 sınıf veya canlı kullanım sonucu değildir.

## Kaynak ve doğrulama

- Kaynak: Drive Desktop `G:\Drive'ım\BitirmeProjesi\outputs\autsl_100_rgb_convnext_tiny_seed7`. `val_metrics.json`, 1.963 videolu `val_predictions.json`, 60 epoch `checkpoints/history.csv`, grafikler ve `best_model.pt`, `final_model.pt`, `latest_model.pt` mevcut. İlgili metrik, tahmin, history ve `paired_val_analysis.json` dosyaları yerel `outputs/` kopyalarıyla SHA-256 düzeyinde eşleşti. `paired_comparison_with_poseonly.json` ham dosya hash'i farklı, fakat iki kopyadaki karşılaştırma sayıları ve video listesi uzunlukları aynı; bu raporun sayıları **tahmin JSON'larından yeniden hesaplandı**.
- Drive'daki config `seed: 7`, `use_rgb: true`, eğitilebilir ImageNet ön eğitimli `convnext_tiny`, 16 kare, fiziksel batch **16**, biriktirme **1**, AMP, gradient checkpointing, 60 epoch üst sınırı, patience 12 ve label smoothing 0,05 tanımlar. Colab A100 ön kontrolü **16×16 kare×2 el** için başarılı: **8,283 GiB** tepe ayrılmış, **8,641 GiB** tepe rezerve bellek, **39,494 GiB** toplam GPU belleği, ısınmış adım medyanı **0,275 sn**. Bu tam epoch veya token maliyeti ölçüsü değildir.
- [Yeniden üretilebilir füzyon betiği](../scripts/analyze_fixed_val_fusion.py), iki JSON'un sınıf sırasını, argmax'ını, olasılık toplamlarını ve tahmin ID/etiketlerinin resmî `data/autsl/val_labels.csv` alt kümesiyle birebir eşliğini doğrular. [Üretilen özet JSON](../outputs/autsl_100_rgb_convnext_tiny_seed7/fixed_equal_fusion_val_analysis.json). Resmî `test` kullanılmadı.
- `best_model.pt` dosyası Drive'da mevcut ve tahmin JSON'u bir checkpoint SHA-256 değeri kaydediyor. Büyük checkpoint'in Drive akışı üzerinden bağımsız tam hash denemesi zamanında bitmedi; **dosyanın o hash ile eşliği bu turda doğrulanmadı**. Ağırlıkla yeniden inference de yapılmadı. Bu, metrik/tahmin yeniden hesabının sınırıdır.

## Tek modeller: küçük net fark, büyük hata değişimi

| Ölçüm | Pose-only | ConvNeXt-Tiny + pose | Fark |
|---|---:|---:|---:|
| Top-1 doğru | 1.678/1.963 | 1.688/1.963 | +10 video |
| Top-1 | %85,48 | %85,99 | +0,51 yüzde puan |
| Macro-F1 | 0,8524 | 0,8563 | +0,0039 |
| Top-5 doğru | 1.914 | 1.868 | −46 video |
| Top-5 | %97,50 | %95,16 | −2,34 yüzde puan |
| Yanlışken ≥0,90 sınıf olasılığı | 22 | 102 | +80 |

Eşleşmiş 1.963 videonun **1.504'ünde ikisi de doğru**, **101'inde ikisi de yanlış**. Hibrit, pose-only'nin **184 hatasını düzeltti**, fakat onun **174 doğrusunu bozdu**. Exact McNemar p≈**0,634** (keşifsel); +0,51 puan bu tek val/seed'de güvenilir üstünlük kanıtı değildir. Video düzeyinde eşleşmiş bootstrap %95 aralığı yaklaşık **−1,32 ile +2,39 yüzde puan**; aynı işaretçiden gelen videolar bağımsız olmadığından bu aralık da genellemeyi olduğundan dar gösterebilir.

**İşaretçi farkı:** Hibrit signer11'de **238→249/294**, signer1'de **514→521/600**, signer35'te **258→267/291** doğruya çıktı; signer16'da **141→130/196** düştü, signer18/25'te üçer doğru kaybetti. Bir kişideki belirgin düşüş, RGB'nin genel bir çözüm olmadığını gösteriyor. Önceki sık karışıklıklardan `onlar→kalem` **9→2**, `şeker→Atatürk` **8→3**, `okul→beklemek` **6→2** oldu; buna karşılık `Atatürk→şeker` **2→7** yükseldi. El görünümü bazı yönlerde yararlı, bazı yönlerde yanıltıcı olabilir.

**Öğrenme eğrisi:** En iyi val doğruluğu ve en düşük val kaybı **54. epoch**ta: train acc **%99,99**, val acc **%85,99**, train loss **0,522**, val loss **1,107**. 60. epoch val **%85,38**. Train doğruluğunun neredeyse %100'e çıkmasına karşı yaklaşık **14 puanlık train–val boşluğu**, işaretçiden bağımsız genellemede ciddi sınır olduğunu gösteriyor. Bunun tek nedeni batch 16 olarak belirlenemez. Val eğrisi 54. epoch'a kadar küçük iyileşmeler içeriyor; "erken epoch kesin daha iyiydi" sonucu da çıkmıyor. Label smoothing nedeniyle train loss'un sıfıra inmemesi beklenir.

## Eğitimsiz sabit füzyon

Füzyon formülü: `p_fused(class) = 0.5 * p_pose(class) + 0.5 * p_hybrid(class)`. **0,5 önceden sabit tutuldu; val üzerinde ağırlık taraması yapılmadı.** Aynı sınıf sırası ve video ID'si doğrulandı.

| Ölçüm | Sabit 0,5/0,5 füzyon | Pose-only'ye fark |
|---|---:|---:|
| Top-1 doğru | **1.784/1.963** | **+106 video** |
| Top-1 | **%90,88** | **+5,40 yüzde puan** |
| Macro-F1 | **0,9070** | **+0,0546** |
| Top-5 doğru | **1.934/1.963 (%98,52)** | **+20 video** |
| Yanlışken ≥0,90 olasılık | **8** | −14 |

Pose-only'ye göre füzyon **163 videoyu kurtardı, 57 doğruyu bozdu**; net **+106**. Hibrite göre **120 kurtarma, 24 bozma**, net **+96**. İki modelin de doğru olduğu 1.504 videonun tümü füzyonda doğru kaldı; ikisinin de yanlış olduğu 101 videonun yalnız **3'ü** kurtarıldı. Füzyonun temel katkısı modellerin **farklı videolardaki** güçlü yönlerini birleştirmek. Pose-only'ye karşı exact McNemar p≈**5,1×10⁻¹³** ve video düzeyinde bootstrap fark aralığı yaklaşık **+3,92 ile +6,93 puan**; aynı resmî val üzerinde seçilmiş modeller ve yalnız altı işaretçi olduğu için bunlar **bağımsız test kanıtı değildir**.

Altı val işaretçisinin tamamında füzyon pose-only'den daha çok doğru verdi: signer1 **514→551/600**, signer11 **238→263/294**, signer16 **141→150/196**, signer18 **263→274/289**, signer25 **264→271/293**, signer35 **258→275/291**. Signer16 yine **%76,53** ile en zor grup. Bu, tek bir işaretçi kazancının toplam sonucu sürüklediği bir durum olmadığını gösterir; gerçek dünya dayanıklılığını ise ayrıca ölçmek gerekir.

## Sonucun anlamı ve sonraki karar

**İyi mi?** 100 sınıflık resmî val alt kümesinde iki eğitimli modelin sabit füzyonuyla **%90,88** güçlü bir araştırma ilerlemesi. ConvNeXt hibritin **tek başına %85,99** sonucu ise daha yüksek hesap maliyetini haklı çıkaracak açık bir kazanım değil. Yanlışken yüksek güvenin 22'den 102'ye çıkması ve Top-5'in düşmesi tek hibritin kalibrasyon/sıralama yönünden dikkat gerektirdiğini gösteriyor. Ensemble'da bu sorun hafifliyor, fakat iki modelin çıkarım maliyeti ekleniyor.

**Yeterli mi?** 226 sınıf hedefi veya canlı tercüman için **hayır**. Bu deney seçilmiş 100 sınıf, izole edilmiş işaret videosu ve altı val işaretçisiyle sınırlı. Canlı sistemde sürekli akıştan işaret bölümleme, yeni ortam/kişiler, gecikme ve cihaz ısısı ayrıca ölçülmeli. Tam 226 sınıf literatür yüzdeleriyle bu %90,88 doğrudan karşılaştırılamaz. [De Coster ve ark.](https://openaccess.thecvf.com/content/CVPR2021W/ChaLearn/html/De_Coster_Isolated_Sign_Recognition_From_RGB_Video_Using_Pose_Flow_and_CVPRW_2021_paper.html) el crop'u ve pose flow'u, [SAM-SLR](https://openaccess.thecvf.com/content/CVPR2021W/ChaLearn/html/Jiang_Skeleton_Aware_Multi-Modal_Sign_Language_Recognition_CVPRW_2021_paper.html) ise çoklu iskelet/görünüm akışlarını tam veri üzerinde inceler; veri, ön eğitim, giriş ve ensemble maliyetleri bu projeyle aynı değildir.

**Şimdiki adım:** 100 sınıfta yeni tam eğitim açma. İki checkpoint'i, strict v1 cache'i ve **sabit 0,5/0,5** füzyonu araştırma adayı olarak dondur. Sonraki ucuz inceleme, **iki modelin de yanlış yaptığı 101 videonun** tekrar eden hata kümelerini (hareket/zaman bilgisi, gövde/yüz bağlamı, el kırpımı, etiket) ayırmak; her videoyu tek tek elle düzeltme hedefi yok. Ardından 226 sınıf strict cache için kapsam ve kalite kapısını tamamla. Bütçe elverdiğinde önce 226 pose-only bazını, sonra aynı train/val/seed üzerinde RGB+pose modelini eğit; sabit 0,5/0,5 füzyonu yalnız o iki **226 sınıf modelin** çıktılarında değerlendir. Füzyon ağırlığını bu 100 sınıf val'inde arayıp 226 başarısı gibi sunma. Mobilde iki model çalıştırma maliyeti ancak 226 kazanımı ve cihaz ölçümü varsa kabul edilir.
