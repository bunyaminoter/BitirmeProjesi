# Antigravity için güncel devir notu

**20 Ekim 226/RGB yön değişikliği:** Kullanıcı 10 sınıf seed 42 tekrarını
zorunlu görmüyor; 10 sınıf yalnız pilot. Güncel öncelik 100 sınıf yakın işaret
stres testi ve tam 226 sınıf. [RGB uzmanı raporu](rgb_226_secici_uzman_arastirmasi_2026-10-20.md)
okunmalı. Yeni `autsl_226_rgb_convnext_tiny.yaml` yalnız araştırma config'i;
226 strict cache henüz yok. Hiçbir 226 eğitim sonucu varmış gibi sunulmasın.
Seçici RGB eşiği, eşleşmiş 100/226 val skorları gelmeden sabitlenmesin.

**20 Ekim pose-only güncellemesi:** Colab seed 7 pose-only çıktı paketi
denetlendi: 198/199 resmî 10 sınıf val, işaretçi 11'de 30/30; hibrit 192/199.
Güncel karar [pose-only raporunda](poseonly_seed7_sonuclari_2026-10-20.md).
Sıradaki küçük Colab işi hazır `configs/experiment/autsl_10_poseonly_seed42.yaml`
ile aynı strict cache/20 epoch tekrarını çalıştırıp video bazlı tahminleri
korumak. Resmî test açılmasın. Aşağıdaki eski pose-only hazırlık talimatları
tarihsel kayıttır; seed 7 deneyi tekrar edilmesin.

**20 Ekim işaretçi 11 incelemesi:**
[Dayanıklılık raporu](signer11_dayaniklilik_analizi_2026-10-20.md) yeni
önceliktir. Resmî val cache'inde işaretçi 11 el tespit oranı diğerlerinden
düşük çıkmadı; kürk/kontrast sebebi kesin ilan edilmesin. Yedi yanlışın
panoları `outputs/colab_seed7_review/error_panels/` altında. Seed 7 hibrit
checkpoint'i korunacak; ilk ayrı Colab teşhis koşusu hazır
`configs/experiment/autsl_10_poseonly_seed7.yaml`. Bu deney yeni
ön işleme veya resmî test gerektirmez. Önceki devir metni tarihsel kayıt.

**20 Ekim karar düzeltmesi:** Seed 42 ve seed 7 sonuçları
[ayrıntılı raporda](colab_seed7_sonuclari_2026-10-20.md) denetlendi. Strict,
10 sınıf geliştirme baz çizgisi; seed 7 checkpoint'i korunacak referans.
Sıradaki küçük iş onun **yedi yanlış val videosunun** ham RGB, el kırpımı,
landmark/tespit ve sınıf karışıklığı açısından panolarını tamamlamak. Dördü
işaretçi 11'e ait. Seed 2026 çifti zorunlu başlangıç işi değil; sonra
dayanıklılık kontrolü için [Colab rehberinde](colab_onisleme_karsilastirmasi.md)
hazır. Resmi test açılmasın. Alt bölüm önceki aşamanın tarihsel devridir.

Son durum: 20 Ekim 2026. **Merkez onarımı ve tam 10 sınıf train/val ön işlemesi
tamamlandı; ilk iki Colab kısa eğitimi çalıştı.** Sıradaki küçük iş güncel ölçüm
kodunu Colab'a aktarıp mevcut checkpoint'leri yalnız val üzerinde tekrar ölçmek.

**20 Ekim devamı:** Güncel AMP/F1 ile iki val JSON'u alındı ve grafikler
incelendi. Temel 174/199, strict aday 177/199; fark 3 video. Sıradaki küçük
iş `docs/colab_onisleme_karsilastirmasi.md` son bölümündeki komutlarla
seed42 video bazlı tahminlerini çıkarmak ve seed7 çiftinin Colab aktarımını
kontrol etmek. Seed7/2026 config'leri hazır, yerel preflight geçti. Önceki
paragraf bu güncellemeden önceki adımı anlatır.

## Önce okunacaklar

- `docs/merkez_onarimi_tam_onisleme_raporu.md`: iki iterasyon, reddedilen ilk tam aday, nihai ölçümler ve sınırlamalar.
- `docs/colab_onisleme_karsilastirmasi.md`: aktarılacak dosyalar, SHA256 kontrolü, iki eğitim ve val komutları.
- `docs/preprocessing_calisma_gunlugu.md`: bütün yapılan işlerin sıralı kaydı.
- `docs/AUTSL_226_yol_haritasi.md`: güncel deney sırası.

## Doğru son aday

`cache/autsl_10_features_centerrepair_strict`: 1.245 train + 199 val, 16 kare,
160×160 çıktı, manifest türetme sürümü 3, ölçek 1,50. Politika dosyası:
`configs/experiment/center_repair_strict.json`; uyuşmazlık eşiği **1.0**.
Son beş merkez değişikliği görsel incelendi; 36.587 tespitli ve 9.619 yedek
kırpım ham videodan piksel düzeyinde tekrar doğrulandı. 63/63 test geçti.

`cache/autsl_10_features_centerrepair` (strict olmayan ilk tam deneme) 0,75
uyuşmazlık eşiği kullanıyor ve iki hızlı hareket gerilemesi nedeniyle reddedildi.
Bu klasör veya 25 videoluk `*_pilot` klasörleri eğitim adayı olarak seçilmez.
Eski `autsl_10_wristcrop_screen.yaml` bu son karşılaştırmanın ikinci kolu değil.

## Güncel sınırlı iş: val ölçüm kodu ve çıktı kontrolü

Notebook tablosunda `macro_f1` yanlış anahtarı iki sahte sıfır üretti. Eski
değerlendirme kodu F1 formülünü de yanlış hesapladı; eğitim içi val ile ayrı
değerlendirme AMP ayarı farklıydı. Güncel `colab_egitim.ipynb`, `scripts/evaluate.py`,
`src/evaluation/evaluator.py` ve `src/evaluation/metrics.py` dosyaları Colab'a
aktarılmalı. Notebook'un 7–8. hücreleri **eski iki `best_model.pt`** dosyasını
yalnız `--split val` ile yeniden ölçer ve `base16_val_v2.json` ile
`centerrepair_val_v2.json` üretir. Yeni eğitim gerekmez. İki JSON, epoch
geçmişleri ve grafikler rapora eklendikten sonra seed 7/2026 tekrarı kararlaştırılır.
Eski JSON/loglar korunur; resmî test açılmaz.

## İlk aktarım kontrolünün kaydı

1. Rehberdeki güncel `src/`, `scripts/`, `configs/`, bağımlılık dosyaları,
   iki cache'in manifest+train/val içeriği, metadata klasörü ve checksum dosyasını
   kullanıcının Colab/Drive proje düzenine yerleştir. Kaynak videoları yeniden
   ön işlemeye gerek yok. Yerel çalışma buluta dosya yüklemedi.
2. `outputs/colab_training_metadata` içindeki dört resmî CSV aynı sınıf/split
   eşlemesidir; metadata ile cache yükleme yerelde doğrulandı. Test videosu veya
   test değerlendirmesi bu aşamada kullanılmaz.
3. `python scripts/check_colab_inputs.py` komutunu rehberdeki metadata/checksum
   seçenekleriyle çalıştır. 2.890 dosya hash'i, manifest eşikleri, 1.245/199 örnek,
   model/augmentation/bütçe eşitliği doğrulanmalı. `outputs/colab_preflight.json`
   yerel kanıttır; yeni Colab kopyası için ayrı preflight çıktısı alınmalı.
4. GPU türü, RAM/boş disk, gerçek cache/çıktı yolları ve aktarım kontrolünü
   çalışma günlüğüne yaz. Eksik aktarım varsa onu düzelt; manifesti uyumlu
   görünsün diye elle değiştirme.

## İlk eğitimlerin kaydı

Kullanılacak iki dosya:

- `configs/experiment/autsl_10_base16_screen.yaml`
- `configs/experiment/autsl_10_centerrepair_screen.yaml`

İkisi de 20 epoch, seed 42 ve aynı model/optimizasyon/augmentation ayarlarına
sahip. Kullanıcının gönderdiği Colab loglarına göre ayrı val Top-1 temel %86,93,
aday %89,45; 199 videoda fark 5. Gerçek Macro F1 düzeltilmiş kodla tekrar
ölçülecek. Yeni omurga, 36 kare, gap2 birleşimi veya 100/226 sınıf henüz bu
karşılaştırmaya eklenmeyecek.

## Korunacak kayıtlar

Kodda ve raporlarda henüz commit edilmemiş önceki çalışmalar olabilir.
`git status` ve diff'i okuyarak devam et; mevcut değişiklikleri reset etme,
eski cache/raporları silme. Reddedilen ilk koşu karşılaştırma kanıtıdır.
`outputs/` ve cache Git tarafından yok sayılabilir; yerel dosyaları da kontrol et.
Her yaptığını komut, dosya, sayım, test sonucu ve kalan işlemlerle çalışma
günlüğüne kaydet. Model doğruluğu artışı eğitim sonucu olmadan iddia edilmez.
