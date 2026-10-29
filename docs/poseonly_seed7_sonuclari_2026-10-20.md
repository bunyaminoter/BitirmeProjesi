# Pose-only seed 7: resmî AUTSL 10 sınıf val analizi ve karar

**Tarih:** 20 Ekim 2026  
**Kaynak:** `C:\Users\Bünyamin\Desktop\chatgptye_gonderilecekler_poseonly` içindeki beş dosyanın değişmeden kopyası `outputs/colab_poseonly_seed7_review/` altında. Resmî test değerlendirilmedi.

## Doğrulanan sonuç

| Aynı 199 resmî val videosu | Doğru | Top-1 | Macro-F1 | İşaretçi 11 |
| --- | ---: | ---: | ---: | ---: |
| Seed 7 hibrit, strict cache | 192/199 | %96,48 | 0,96455 | 26/30 |
| Seed 7 pose-only, strict cache | **198/199** | **%99,50** | **0,99500** | **30/30** |

Fark **6 video, +3,02 yüzde puan**. Eşleşen videolarda 191 ikisi de doğru, 7 yalnız pose-only doğru, 1 yalnız hibrit doğru, ikisi de yanlış 0. Pose-only hibritin yedi yanlışının hepsini düzeltti: `signer11_sample342/447/570/600`, `signer1_sample866/1012`, `signer35_sample60` (hepsinin son eki `_color`). Tek yeni yanlış `signer16_sample317_color`: gerçek AUTSL sınıf ID **114**, tahmin **106**, pose-only güveni 0,7763; doğru sınıf olasılığı 0,2045. Pose-only işaretçi 11'de 30/30, diğerlerinde 168/169. On sınıfın dokuzunda tüm val örnekleri doğru; sınıf 114'te 19/20. Top-5 %100, fakat 10 sınıfın yarısını aday olarak vermek canlı çeviri doğruluğu sayılmaz.

`poseonly_seed7_history.csv` ve grafikte en yüksek val doğruluğu ve en düşük val kaybı **17. epoch**: 198/199, val loss 0,3292. 20. epoch val doğruluğu 196/199, train doğruluğu %97,32. `best_model.pt` bu akışta val accuracy iyileştiğinde saklanır; dışa aktarılan değerlendirme 17. epoch sonucuyla uyumlu. 20 epoch sonunda kalıcı ve tek yönlü aşırı öğrenme yok; son üç epoch dalgalanıyor. En iyi epoch'un val ile seçildiği sonuç değerlendirilirken hesaba katılmalı.

## Bağımsız dosya ve ayrım denetimi

- Tahmin JSON'undaki 199 benzersiz video ID'si ve sınıf sırası, `outputs/colab_training_metadata/val_labels.csv` içinden seçilen resmî 199 ID ile birebir eşleşti. 199 gerçek etiketin tamamı ve önbellekteki etiketler doğru. Olasılık dizileri 10 elemanlı, sonlu, toplamları 1; `pred_label` ve `top1_confidence` ile uyumlu. Top-1 ve Macro-F1, karışıklık sayımlarından tekrar hesaplandı ve JSON ile eşleşti.
- 1.245 train / 199 val `.npz` dosyası tam. Train ve val işaretçileri, video ID'leri ayrık. Train ile val arasında 16×75×3 landmark dizisinin SHA256 düzeyinde birebir kopyası yok. Aynı split içinde iki birebir landmark çifti var; bu inceleme iki split arasındaki sızıntı sorusunu değiştirmiyor, ama veri kalitesi açısından ileride bakılabilir.
- Config farkı esas olarak `model.use_rgb: false`: RGB el kırpımı kod kolu ve füzyon devre dışı; 33 gövde + 21 sol el + 21 sağ el noktasından oluşan **75 nokta / 225 koordinat** ve zamansal model çalışıyor. Dolayısıyla “pose-only”, el parmak noktalarını da içeriyor ve MediaPipe takibine hâlâ bağımlı. Aynı strict cache, 16 kare, seçilmiş 10 sınıf, seed 7, 20 epoch ve diğer eğitim ayarları kullanılmış.
- Çiftli exact McNemar keşifsel p=0,0703125 (7 düzelme / 1 gerileme). Pratik kazanç belirgin; bu küçük ve daha önce tekrar tekrar incelenmiş val üzerinde genellenebilir/istatistiksel üstünlük **kesinleşmedi**. %99,50, 226 sınıfın veya canlı kameranın başarısı değildir.
- Pakette `best_model.pt`, tam Colab logu ve Colab'da kullanılan kod/config anlık görüntüsü yok. Tahmin JSON'u checkpoint SHA256 değerini içeriyor: `16c19538c1397c009d9abeb217acd9e9981b5c797f111ad8c054b70268c7a14d`. Ağırlığı bağımsız yeniden çalıştırmak ve mobil/web'e aktarmak için checkpoint ayrıca korunmalı.

Bu sonuç, işaretçi 11'de kürklü yeleğin MediaPipe tespitini bozduğunu kanıtlamıyor. Önceki kare tespit oranları işaretçi 11'de düşük değildi. RGB kolunu kapatınca sınıflandırma belirgin iyileşti; ek parametrelerin, füzyonun, görsel özelliklerin veya eğitim dinamiğinin hangisinin farkı yarattığı bu tek ablation ile ayrılamaz.

## Karar ve sıradaki deney sırası

1. **10 sınıf için yeni çalışma adayı pose-only seed 7; hibrit seed 7 korunacak.** Önce ağırlık (`best_model.pt`), Colab logu ve kullanılan config/kod sürümü saklanmalı. Test açılmayacak. Val'deki tek hata ham kare/landmark/maske ve sınıf 114↔106 açısından incelenecek; bu tek örneğe özel kural yazılmayacak.
2. **Tek kısa tekrar:** Hazır `configs/experiment/autsl_10_poseonly_seed42.yaml` ile aynı strict cache ve 20 epoch bütçesinde pose-only'yi seed 42 ile Colab'da eğit; mevcut seed 42 hibritin 177/199 tahminleriyle resmî val'de video bazında eşleştir. Bir seed daha “zorunlu sayı tamamlama” için değil, 10 sınıfta görülen %99,50 sıçramasının başlangıç rastgeleliğine ne kadar bağımlı olduğunu görmek içindir. Çıktı: config, checkpoint SHA256, history, val metrikleri, tüm video tahminleri, işaretçi ve sınıf kırılımı. Seed 42 sonucu düşükse önce seed/öğrenme kararlılığını incele; yüksekse ölçeklemeye geç.
3. **100 sınıf geçişi:** Resmî train/val'de önceden sabitlenmiş 100 sınıf listesini oluştur, ayrı manifestli 16 kare landmark cache üret ve görsel/sayısal kalite kapısından geçir. Pose-only 20 epoch baz çizgisini ayrı 100 sınıf config'iyle çalıştır. Burada 10 sınıfın %99,50'sini koruması beklenmez. Hibrit karşılaştırma ancak 100 sınıfta sınıf/signer hata analizi RGB'nin ek değerini gösterirse yapılacak.
4. **226 sınıf ve ürün kapısı:** Aynı protokolü tam resmî train/val'e büyüt; zorlanan sınıflar için üst beden/RGB/temporal alternatiflerini ayrı deneyler olarak seç. Tarif sabitlenince kilitli resmî test bir kez değerlendirilecek. Canlı kullanım için farklı kişiler/kamera/ışık, sürekli işaret bölütleme, bilinmeyen işaret reddi, gecikme ve MediaPipe kaybı ayrıca ölçülecek.

Seed 42 koşusu için Colab'daki güncel projede (strict cache ve metadata yerindeyken):

```bash
cd /content/BitirmeProjesi
export AUTSL_DATA_DIR=/content/drive/MyDrive/BitirmeProjesi/outputs/colab_training_metadata
python scripts/train.py --config configs/experiment/autsl_10_poseonly_seed42.yaml --cache_dir cache/autsl_10_features_centerrepair_strict
python scripts/evaluate.py --config configs/experiment/autsl_10_poseonly_seed42.yaml --cache_dir cache/autsl_10_features_centerrepair_strict --checkpoint /content/drive/MyDrive/BitirmeProjesi/outputs/autsl_10_poseonly_seed42/checkpoints/best_model.pt --split val --output outputs/poseonly_seed42_val.json --predictions_output outputs/poseonly_seed42_predictions.json
python scripts/analyze_val_pairs.py --base outputs/centerrepair_seed42_predictions.json --candidate outputs/poseonly_seed42_predictions.json --official_labels "$AUTSL_DATA_DIR/val_labels.csv" --output outputs/hybrid_vs_poseonly_seed42_pairs.json
```

Seed 42 hibrit tahmin dosyası Colab'da bu yolda değilse önce korunmuş tahmin
JSON'unu aynı adla `outputs/` altına koy. Koşudan önce 42 config'inin Colab
kopyasının güncel olduğuna ve strict cache manifestinin eşleştiğine bak.

Şimdilik val'e ağırlık seçilmiş geç füzyon veya tek hatayı düzeltmeye dönük eşik eklemek gereksiz ve aşırı uyum riski taşıyor. Pose-only daha yalın adaydır; gerçek hız/enerji üstünlüğü ayrıca ölçülmelidir çünkü mevcut veri yükleyici pose-only'de bile RGB kırpımlarını okuyor.
