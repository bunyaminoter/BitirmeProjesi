# Colab seed 7 sonuçları ve iki seed karar analizi

**Tarih:** 20 Ekim 2026  
**Kaynak:** `C:\Users\Bünyamin\Desktop\chatgptye_gonderilecekler.zip`  
**Kaynak ZIP SHA256:** `7FBA74C872D39AD195C35689619747B506BEDA15848B85A34215FBD22FA7681E`  
**Yerel kanıt:** `outputs/colab_seed7_review/` ve önceki `outputs/colab_seed42_review/`.

## Ne doğrulandı?

ZIP'in 10 girdisi açılmadan önce ad, tür ve yol bakımından denetlendi; iki seed 42 tahmin JSON'u, seed 7 için iki metrik JSON'u, iki tahmin JSON'u, iki 20 epoch geçmişi ve iki grafik korundu. Dört tahmin dosyasının her birinde 199 benzersiz video bulunuyor. Video kimlikleri ve gerçek sınıflar `outputs/colab_training_metadata/val_labels.csv` içindeki resmi AUTSL val bölümünün seçili on sınıfıyla bire bir eşleşiyor. Sınıflar orijinal AUTSL ID'leriyle `[8,14,15,20,64,106,114,143,144,196]`; model içi indeksler 0–9. Doğruluk ve gerçek sınıf bazlı macro-F1, tahminlerden yeniden hesaplanarak JSON'larla eşleştirildi. Her tahmin dosyası checkpoint SHA256 kimliği içeriyor; ağırlık dosyalarının kendisi ZIP'te olmadığı için ağırlıkları bağımsız açıp doğrulamadım. Test bölümü açılmadı.

## Düzeltilmiş karşılaştırma

| Seed | Ön işleme | Val doğru / 199 | Top-1 | Macro-F1 | Top-5 |
| --- | --- | ---: | ---: | ---: | ---: |
| 42 | base16 | 174 | %87,44 | 0,87185 | 198/199 |
| 42 | centerrepair_strict | 177 | %88,94 | 0,88852 | 199/199 |
| 7 | base16 | 188 | %94,47 | 0,94517 | 199/199 |
| 7 | centerrepair_strict | 192 | **%96,48** | **0,96455** | 199/199 |

Seed 42'de adayın net kazanımı **3 video / 1,51 yüzde puan**, seed 7'de **4 video / 2,01 yüzde puan**. Aynı 199 val videosunda bir video **0,503 yüzde puan** eder. Aday her iki seed'de önde; fakat seed değişince base16 sonucu 14, strict sonucu 15 video yükseldi. Dolayısıyla bu iki koşudaki seed duyarlılığı ön işleme kazancından daha büyük. Seed 7'nin %96,48 değeri on sınıflı **val** sonucudur; 226 sınıf veya resmi test sonucu değildir.

Eşleştirilmiş video sayıları (iki modelin aynı videoda doğru/yanlış durumu):

| Seed | İkisi doğru | Yalnız temel doğru | Yalnız aday doğru | İkisi yanlış | Keşifsel exact McNemar p |
| --- | ---: | ---: | ---: | ---: | ---: |
| 42 | 158 | 16 | 19 | 6 | 0,736 |
| 7 | 186 | 2 | 6 | 5 | 0,289 |

Bu p değerleri aday lehine kesin üstünlük kanıtı sunmuyor. İki seed aynı 199 videoyu kullandığı için 398 bağımsız doğrulama örneği varmış gibi toplama veya birleşik p değeri çıkarma doğru olmaz. Buradaki amaç kontrollü deneyde yönü ve sorunlu örnekleri görmek; val tekrarları model seçiminde kullanıldıkça sonuç iyimserleşebilir.

Bu karşılaştırma bütün `centerrepair_strict` ön işleme politikasının etkisini
ölçer. Önceki piksel denetimine göre val'de merkez düzeltmesi sıfırdı; farkı
yalnız merkez onarımının nedensel kazancı olarak yorumlamıyoruz.

## Eğitim eğrileri ve hata kümeleri

Seed 7 base16 en iyi val doğruluğuna 17. epoch'ta (188/199), en düşük val kaybına 20. epoch'ta (0,50365) ulaştı. Strict adayda ikisi de 19. epoch'ta (192/199; kayıp 0,48052). Adayın 20. epoch val doğruluğu 187/199'a indi ve kayıp 0,51891'e çıktı; değerlendirmedeki 192/199 doğru sonuç **en iyi checkpoint'e** ait. Tek epoch'luk yükselişten sürekli aşırı öğrenme sonucu çıkarmıyoruz. İki grafik de incelendi.

İşaretçi 11'de strict her iki seed'de geriledi: seed 42'de 28/30 → 22/30, seed 7'de 27/30 → 26/30. Buna karşılık işaretçi 16'da seed 42 için 16/20 → 20/20 ve işaretçi 25'te seed 7 için 27/29 → 29/29. Seed 42'de sınıf 106 (`kardeş`) 20/20 → 13/20 gerilerken seed 7'de 19/20 → 20/20; tek sınıfta kalıcı bir sorun olduğu henüz söylenemez.

Dört modelin tamamında yanlış olan iki dikkat çekici val videosu: `signer11_sample570_color` (resmi ID 114, kitap) ve `signer1_sample1012_color` (resmi ID 8, aile). İkincisinde strict cache'in 16 karenin her birinde iki eli de tespit ettiği görüldü; bu örnek, eksik-el fallback'inin tek başına çözeceği bir hata değil. `signer11_sample570_color` için tespit boşlukları ve bulanık/örtüşen el görüntüleri mevcut. `signer11_sample342_color`, `signer11_sample447_color` ve `signer11_sample600_color` videolarının ham kare panoları da incelendi; etiket hatası teyit edilmedi. Görsel kanıtlar aynı çıktı klasöründeki `*_raw_contact.png` ve `*_strict_crops.png` dosyalarında. Bu örnekler için etiket değiştirme veya seçili val'i eğitim setine aktarma kararı yok.

## Bu sonuçlarla hedefe ulaşabilir miyiz?

**Evet, ilerlemeye değer:** on sınıfta signer bağımsız val doğruluğu tek bir seed için %96,48'e çıktı; iki seed'de aday lehine yön var. **226 sınıfta %95+ henüz kanıtlanmış değil:** sınıf sayısı, benzer işaretler ve yeni işaretçiler artacak; resmi test sonuçları elimizde yok. 2021 ChaLearn AUTSL yarışmasında 226 sınıf için resmi testte RGB birincisi %98,42, RGB+D birincisi %98,53 bildirdi; bunlar çoklu akış/ensemble ve farklı eğitim kaynakları kullanan sistemlerdi. Mevcut tek MobileNetV3 el kırpımı + nokta modeliyle bu puanın otomatik geleceği çıkarılamaz. Kaynak: [ChaLearn yarışma makalesi, Tablo 2 ve yöntemler](https://arxiv.org/html/2105.05066).

## Karar ve sıradaki kapılar

1. **Şimdi:** Strict politikayı 10 sınıf geliştirme baz çizgisi, seed 7 `best_model.pt` dosyasını korunacak referans olarak kullan. Seed 7 strict'in **yedi** yanlış val videosunu incele: beşi iki seed 7 modelinde ortak yanlış, ikisi yalnız strict'te yanlış; dördü işaretçi 11'e ait. Ham RGB, crop, landmark/tespit boşluğu, tahmin güveni ve karışan sınıfları karşılaştır. Bu analiz yeni model eğitimi gerektirmez.
2. **Hata nedenine göre tek değişkenli deney:** Sorun eğitim süresi/optimizasyonsa 20 epoch'tan devam ettirilmiş koşu yerine yeni 40 epoch tarifini eş bütçeli karşılaştır; bağlam eksikse üst beden/yüz RGB dalını ayrı dene; el ayrıntısı eksikse el çözünürlüğü/omurga kapasitesini ayrı dene. Her aday aynı resmî train/val ve seed 7 baz çizgisine karşı ölçülsün. Sırf yedi videoyu düzeltmek için val'e özel kural veya etiket müdahalesi yapılmasın. 196/199 = %98,49 için mevcut 192 doğrudan **net dört** doğru video daha gerekir.
3. **Dayanıklılık kontrolü:** Yeni reçete seçilmeden veya 100 sınıfa ölçeklenmeden önce ikinci bir seed ve işaretçi 11 kırılımı gözden geçirilsin. Hazır seed 2026 base16/strict çifti bu amaçla kullanılabilir; **şimdiki çalışmaya başlama şartı değildir**. Koşul çalıştırılırsa preflight, val metrik/tahmin JSON'ları ve geçmiş/grafikler [rehberdeki](colab_onisleme_karsilastirmasi.md) gibi saklansın.
4. **100 sınıf ölçekleme:** Seçilen tek politika ile resmi train/val için yerel cache üret; manifest, sayım, sınıf eşlemesi, bozuk video ve örnek kalite denetimi tamamlanınca Colab'da aynı tabanı eğit. 10 sınıfın %96,48'ini 100 sınıfa projekte etme.
5. **226 sınıf ve mimari:** Önce aynı tabanı tüm 226 sınıfta ölç. Hatalar el biçimi, konum, yüz/beden bağlamı ya da uzun hareket ayrımında yoğunlaşıyorsa tek değişkenli güçlü RGB/üst beden video akışı veya daha iyi önceden eğitilmiş uzamsal-zamansal omurga dene; val'de gerçekten tamamlayıcılık varsa füzyon/ensemble'a geç. Yarışma makalesi el/yüz/beden, iskelet ve zamansal akışların katkısına işaret ediyor. Resmi test, son tarif sabitlenene kadar seçim aracı olmasın.

**Seed yorumu (20 Ekim karar düzeltmesi):** Seed 42 ve 7 config'lerinde deney
adı, çıktı yolu ve `seed` dışında fark yok. Seed sayısı büyüklük sırası olan
bir performans hiperparametresi değildir; `7 < 42` modeli sistematik olarak
iyileştirmez. Seed başlangıç ağırlıklarını, veri karıştırmayı ve train
augmentasyonunu etkiler. Seed 42 iki kolda da en iyi val doğruluğunu 20.
epoch'ta gördü; seed 7 strict 19. epoch'ta zirve yaptı. Bu, 20 epoch
bütçesinin seed 42'yi erken kesmiş olabileceği hipotezini doğurur; sebep
kanıtlanmış değildir. Üçüncü seed faydalı olabilir, zorunlu değildir.

**Maliyet notu:** Mevcut 1.444 videoluk **tek strict cache** yaklaşık 1,647 GiB; iki varyantın cache'leri birlikte yaklaşık 3,18 GiB. Tek politika için aynı ortalama ile 100 sınıf train+val yaklaşık 16–17 GiB, 226 sınıf yaklaşık 37 GiB olabilir; gerçek boyut ve süre ön pilotla ölçülmeli. 36 kareyi şimdilik açmıyoruz; önceki tam veri taramasında tespit oranı ve uzun boşluklar kötüleşmişti.

**Yerel doğrulama:** `tests/test_analyze_val_pairs.py` 1/1, tüm proje
`pytest` 67/67 geçti; `compileall` ve `git diff --check` başarılı. Bunlar
Colab koşusunun yeniden üretildiği anlamına gelmez; ZIP'te checkpoint ve
Colab ortamı yoktu.
