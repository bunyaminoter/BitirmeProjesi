# AUTSL 100: zamansal takip + üst gövde ROI için karar

**22 Ekim 2026.** Amaç model eğitiminden önce 16 karelik ön işlemede **işaretin etkin elini, doğru anatomik tarafta ve doğru parmak noktalarıyla** korumaktır. Bütün deneyler önceden seçilmiş 40 **train** videosunda salt okunur çalıştı: 16 iki eli seyrek, 16 tek tarafı uzun süre kayıp, 8 sağlam kontrol. Resmî val/test, mevcut `base16` ve `centerrepair_strict` NPZ'leri, model ağırlıkları değiştirilmedi. Model eğitimi yapılmadı.

## Görsel tarama ve birleşimin gerçek katkısı

194 yoğun VIDEO yeni bilek konumu ve buna ek olarak VIDEO ile `<0,06` normalize mesafede eşleşmeyen 130 üst gövde ROI konumu, 300×300 temas panolarında tek tek **el konumu/işaret rolü** açısından tarandı. Bu 324 adayın 270 farklı örnek kareye dağıldığı `data/autsl/autsl_100_640_frame_review_progress.csv` içinde kayıtlıdır. Kalan 370 karede aday konumu yoktur. Bu, **640 karenin tam anatomik insan etiketlemesi değildir**: tam karede hangi fiziksel sol/sağ elin göründüğü ve 21 parmak noktasının doğruluğu hâlâ boş; VIDEO adaylarının 149/194'ünde anatomik taraf çözümsüz, ROI-only adaylarında da doğrulanmış taraf yoktur.

| Salt okunur aday kaynağı | Yeni konumların görsel rolü | Etkin el konumu önerilen farklı kare | 640 seçilmiş karede oran |
|---|---|---:|---:|
| X: tüm kaynak kareleri izleyen VIDEO | 125 etkin, 23 pasif, 41 nötr, 5 belirsiz / 194 | 120 | %18,75 |
| Y: üst gövde ROI | VIDEO ile eşleşmeyenlerde 64 etkin, 15 pasif, 47 nötr, 4 belirsiz / 130 | 130 | %20,31 |
| X + Y konum birleşimi | ROI, VIDEO'ya 43; VIDEO, ROI'ye 33 ek etkin konumlu kare getirir | 163 | %25,47 |

**Bu yüzdelerin anlamı:** seçilmiş 40 zorlu videonun 640 örnek karesinde, mevcut bilekten `>0,06` uzakta **görsel olarak etkin el üzerinde görünen yeni bilek adayının bulunduğu kare oranı**. Bunlar model doğruluğu artışı, gerçekten eksik anatomik el sayısı, doğru 21 nokta veya tüm 100 sınıfta beklenen kazanım değildir. Aynı fiziksel elin biraz kaymış tespiti de `>0,06` nedeniyle “yeni” sayılabilir. Seçim zor örneklere ağırlık verdiğinden oran 100 sınıf geneline taşınamaz. X ve Y aynı konumları sıkça paylaştığı için kazanımları toplanamaz.

Kullanıcının istediği “X eklenirse yaklaşık yüzde kaç daha iyi?” biçimindeki **ölçülebilen koşullu yanıt** budur: X'in *aday erişim fırsatı* bu kuyrukta yaklaşık **%19 kare**, Y'ninki **%20 kare**, birlikte **%25 kare**. X üstüne Y **+%6,72 puan**, Y üstüne X **+%5,16 puan** aday kare ekler. Bu sayıları “ön işleme kalitesi +%19/+%20/+%25” veya “val acc +%4/+%10” diye yazmak verinin desteklemediği bir nedensel iddia olur. Bugün doğrulanmış **güvenli cache kazanımı 0** ve model doğruluğu etkisi **bilinmiyor**.

## Tekrarlanan yüz önünde el kaybı

Önceki VIDEO denemesinde `signer17_sample52/189/114` için 5–13. karelerin **27/27'sinde** temel ve VIDEO sıfır eldi. ROI'nin aynı etkin pencerede bulduğu kareler: `sample52` 6–12 (**7/9**, 7–11'de iki el), `sample114` yalnız 7 (**1/9**), `sample189` **0/9**. Toplam **8/27** kare. Dolayısıyla ROI, tekrarlanan hatanın bir örneğinde önemli görüntü bilgisi kurtarıyor; diğer iki örneği genel olarak çözmüyor. [ROI panosu](../outputs/autsl_100_drive_review/upperbody_roi_40qa/signer17_sample52_color.png).

## Zaman, çift algılayıcı ve kol izi kapıları

Önceden belirlenen tanı eşikleriyle, bitişik 16 örnek karede bilek konumu `<=0,14` yakın olan en az üç karelik iz arandı. Taraf etiketi iz kurarken **kullanılmadı**; bu, çakışan ellerde iz karışmasını mümkün bırakan iyimser bir sınamadır.

| Kuru seçim (VIDEO adayları) | Seçilen | Görsel etkin | Pasif + nötr | Sağlam kontrol adayı |
|---|---:|---:|---:|---:|
| Zamansal iz ≥3 | 115 | 97 | 18 | 0 |
| İki algılayıcı aynı bilek + iz ≥3 | 71 | 64 | 7 (5 pasif, 2 nötr) | 0 |
| İz ≥3 + poz tarafı/eksik taraf uzlaşması | 44 | 39 | 5 (2 pasif, 3 nötr) | 0 |

Zaman ve iki algılayıcı uzlaşması nötr başlangıç/son adaylarını ciddi azaltıyor, fakat beş **pasif** el hâlâ kapıyı geçiyor. Bu sayılar aynı 40 klipte kural incelemesidir; ayrı doğrulama kümesinde hassasiyet tahmini değildir. ROI-only için ≥3 karelik iz 57 aday seçiyor: yalnız 31 etkin, 12 pasif, 14 nötr. Bu dalı koşulsuz eklemek özellikle riskli.

Kol izi için poz dirsek→bilek vektörünün bileğin %25 ötesindeki nokta hedef alındı; iki ROI eli varsa iki tarafa toplam uzaklığı en aza indiren birebir atama yapıldı. ROI bileği ile önbellekteki bilek `<0,06` eşleşen **456** konumun **382'sinde** atama mevcut tarafla uyumlu, **74'ünde** uyumsuzdu (**%16,2**). Atama maliyet farkı `>=0,03` olan güçlü görünen 438 örnekte bile **69 uyumsuzluk** vardı. Bu karşılaştırma insan anatomik taraf etiketi değil; mevcut cache de yanlış olabilir. Yine de `signer38` gibi sol/sağ çakışmasına güvenli düzeltme kanıtı değildir. `signer17_sample52` 7–11. karelerde iki ROI eli için maliyet farkı yaklaşık `0,042–0,057` ve tutarlı iki taraf önerisi verdi; tek video başarısı genel kuralı doğrulamaz.

## Karar ve sonraki deney

**X, Y veya X+Y bugün üretim önbelleğine yazılmıyor; 10 sınıf model deneyi açılmıyor.** Yöntemlerin potansiyeli var: 43 ROI'ye özgü etkin aday kare ve `signer17_sample52` kurtarması, araştırmayı sürdürmeye değer. Ancak aktif/pasif ve anatomik taraf karışması, yeni parmak geometrisinin doğrulanmaması, 40 videonun amaçlı seçimi ve tam 640 kare insan etiketinin eksikliği nedeniyle kalite kapısı geçilmedi.

1. Önce 40 videonun etkin bölümünden **yaklaşık 120 karelik ayrı bir taraf/rol doğrulama alt kümesi** seç: üç `signer17`, iki `signer38`, pasif el karşı örnekleri (`signer22_sample484`, `signer41_sample239`, `signer24_sample177`, `signer32_sample63`) ve 8 sağlam kontrol. Videoları/işaretçileri kural geliştirme ve sonradan açılacak kontrol olarak ayır; aynı videonun komşu karelerini iki tarafa bölme. Kırpılmış tek kare yerine video akışı ve yakın planla fiziksel sol/sağ, aktif/pasif, örtüşme ve doğru 21 noktayı işaretle. `data/autsl/autsl_100_640_frame_review_progress.csv` içindeki boş tam kare alanlarını yalnız gerçekten incelenen karelerde doldur; şu an sıfır tam kare insan etiketi var. Bu alt küme kuralı ayırt etmeye yetmezse kalan 640 kareyi tamamla.
2. Ayrı **salt okunur** pilotta eksik el için üst gövde ROI'yi, ardışık VIDEO izlemesini ve dirsek→bilek izini birlikte kullan. Aynı bileğe iki kez eklemeyi engelle; taraf belirsizse eklemeyi reddet. Pasif elin görünmesi, aktif elin bulunmasıyla karıştırılmasın. `signer17_sample189/114` hâlâ kayıpsa dar kırpım/yeniden tespit gibi yeni yöntemleri yalnız bu karşı örneklerde dene.
3. **Kapı:** farklı işaretçi ve sınıflarda tekrarlanan etkin el kazanımı, görsel olarak doğru anatomik taraf ve 21 nokta, 8 sağlam kontrolde mevcut tensörleri birebir koruma, pasif ele yanlış ekleme olmaması. Tam 100 sınıf ve val bu küçük seçilmiş pilotun yüzdeleriyle yeniden yazılmaz. Kapı geçerse yalnız yeni sürüm adlı küçük cache çıkar ve **aynı 10 sınıf/seed/epoch/resmî val** ile eşlenmiş kısa model karşılaştırması yap. Aksi halde `strict v1` korunur ve 100 sınıf model adımına onunla geçilir.

## Tekrar üretim

Ham ölçümler `outputs/autsl_100_drive_review/dense_video_hand_40train/audit/` altında `candidate_union_summary.json`, `temporal_gate_summary.json`, `forearm_assignment_summary.json`; 13 VIDEO ve 9 ROI-only temas panosu `contact/` ile `roi_only_contact/` altında. Görsel karar CSV'leri `data/autsl/autsl_100_dense_candidate_visual_review.csv` ve `data/autsl/autsl_100_roi_only_candidate_visual_review.csv`. Kod: `scripts/audit_autsl_candidate_union.py`, `scripts/build_autsl_candidate_contact_sheets.py`, `scripts/label_autsl_roi_only_review.py`, `scripts/audit_autsl_temporal_proposals.py`, `scripts/audit_autsl_forearm_assignment.py`, `scripts/build_autsl_640_frame_review_progress.py`. Tüm sonuçlar train pilotudur; val/test etiketi veya model eğitimi kullanılmadı.
