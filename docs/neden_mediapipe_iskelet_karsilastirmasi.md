# İskelet ve Poz Çıkarım Kütüphaneleri Karşılaştırması: Neden MediaPipe?

**Tarih:** 22 Ekim 2026  
**Kapsam:** TÜBİTAK 2209-A ve Lisans Bitirme Projesi (AUTSL 10 / 100 / 226 Sınıf İşaret Dili Tanıma & Mobil Sohbet Asistanı)

---

## 1. Giriş ve Karar Sorusu

İşaret Dili Tanıma (SLR) literatüründe eklem/nokta (landmark) çıkarımı model başarısının temel girdisini oluşturmaktadır. Bu projede ilk günden itibaren **Google MediaPipe** (33 vücut pozu + 21 sol el + 21 sağ el = toplam 75 landmark) kullanılmıştır. 

100 sınıfa geçiş aşamasında "Daha üstün bir iskelet kütüphanesi var mı, neden MediaPipe ile devam ediyoruz?" sorusu akademik literatür ve uç cihaz (edge-device) mühendislik gereksinimleri açısından incelenmiştir.

---

## 2. Alternatif İskelet Kütüphanelerinin Değerlendirilmesi

### 2.1. MMPose / RTMPose (OpenMMLab - CVPR 2023)
* **Karakteristik:** Güncel bilgisayarlı görü dünyasının en kapsamlı açık kaynak poz kestirim kütüphanesidir.
* **Avantajları:** 
  * Hareket bulanıklığı (motion blur), hızlı el hareketleri ve ellerin birbiriyle veya yüzle çakıştığı durumlarda MediaPipe'a göre daha yüksek tespit keskinliği sağlar.
  * WholeBody modeli ile vücut, el ve yüzü (133 nokta) tek bir modelde aynı anda çıkarabilir.
* **Dezavantajları ve Feda Edilenler (Trade-offs):**
  * PyTorch ve CUDA bağımlılığı yüksektir. CPU üzerinde gerçek zamanlı (30 FPS) çalışamaz (çoğunlukla 5–12 FPS bandında kalır).
  * Mobil cihazlarda (Flutter / Android / iOS) ek optimizasyon (ONNX Runtime / TensorRT / NCNN) yapılmadan yerleşik çalıştırılması çok zordur; sunucu tarafında GPU gerektirir.

### 2.2. DWPose (ECCV 2024)
* **Karakteristik:** Güncel üretken yapay zeka (ControlNet, AI video üretimi) ve akademik pose estimation'da SOTA doğruluk standardıdır.
* **Avantajları:** Parmak eklemleri arasındaki minör açıları ve mikroskobik temasları en yüksek doğrulukla yakalar.
* **Dezavantajları:** Çıkarım maliyeti aşırı yüksektir; güçlü bir sunucu GPU'su (NVIDIA RTX / A100 / T4) gerektirir. Canlı kamera akışında son kullanıcı cihazında yerel çalışmaya uygun değildir.

### 2.3. OpenPose (CMU)
* **Karakteristik:** Poz kestiriminin ilk meşhur öncüsü (tarihsel referans).
* **Dezavantajları:** Günümüz standartlarına göre son derece hantal ve ağırdır (CPU'da < 1-2 FPS). Ayrıca ticari kullanım için yıllık 25.000$ gibi ağır bir lisans kısıtlamasına sahiptir. Endüstri ve akademide yerini MediaPipe ve MMPose'a bırakmıştır.

### 2.4. YOLOv8 / YOLOv11 Pose (Ultralytics)
* **Karakteristik:** Nesne tespiti hızını poz kestirimine uyarlayan mimari.
* **Dezavantajları:** Standart pose modelleri yalnızca **17 vücut eklemini** bulur. İşaret dilinin asıl anlam taşıyıcısı olan **21 parmak eklemini** yerel olarak sunmaz; eller için ikincil bir algılama ve kırpma modeli zinciri gerektirir.

---

## 3. Karşılaştırma Matrisi

| Kriter | Google MediaPipe | MMPose / RTMPose | DWPose (ECCV 2024) | YOLOv8/11 Pose | OpenPose (CMU) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Gövde + Çift El (75+ Nokta)** | ✅ Dahili (33+21+21) | ✅ Dahili (WholeBody 133) | ✅ Dahili (133 nokta) | ❌ Yalnızca 17 gövde | ⚠️ Ayrı el modelleri |
| **CPU Çıkarım Hızı (FPS)** | **30+ FPS (Ultra Hızlı)** | 5–12 FPS (Yavaş) | 2–5 FPS (Çok Yavaş) | 25+ FPS (Hızlı) | < 2 FPS (Kullanılamaz) |
| **İstemci / Mobil Uyumu** | **Kusursuz (Android/iOS SDK)** | Zor (Özel port gerekir) | Çok Zor (Sunucu şart) | Orta (TFLite/ONNX) | İmkansız (Çok ağır) |
| **Tarayıcı / WASM Desteği** | **Mükemmel (JS/WASM)** | Sınırlı | Yok | Sınırlı | Yok |
| **Donanım / Sunucu Maliyeti** | **0 TL (İstemcide çalışır)** | Yüksek GPU maliyeti | Çok Yüksek GPU maliyeti | Orta GPU maliyeti | Yüksek GPU maliyeti |
| **Zor/Bulanık Poz Doğruluğu** | İyi / Orta | **Çok Yüksek** | **SOTA (En Yüksek)** | İyi (Gövde) | Orta |
| **Proje İçin Nihai Karar** | **✅ ASIL SEÇİM** | ❌ Uç cihaz için elendi | ❌ Maliyet için elendi | ❌ El noktası yok | ❌ Eski / Ağır |

---

## 4. Neden MediaPipe ile Devam Ediyoruz? (Mühendislik Gerekçesi)

1. **Uç Cihaz (Edge AI) ve Gerçek Zamanlılık İlkesi:**  
   Bu projenin nihai çıktısı bir sunucuya bağımlı olmaksızın işitme engelli bireyin akıllı telefonunda (Flutter / Android) kameradan anlık işaret tanıyan bir yardımcı asistan (SignAI) geliştirmektir. MediaPipe, harici bir GPU sunucu kiralamaya gerek kalmadan kullanıcının kendi telefonunda 30 FPS hızında çalışabilen tek olgun ekosistemdir.

2. **Zayıflığın Veri Mühendisliğiyle (Strict v1) Çözülmesi:**  
   MediaPipe'ın en büyük handikapı hızlı hareketlerde veya yüz önünde ellerin kısa süreli kaybolmasıdır. Projemizde bu zafiyeti daha ağır bir kütüphaneye geçerek değil; **1.5x bilek fallback'i ve sınırlı merkez onarımı (`centerrepair_strict v1`)** geliştirerek çözdük. Bu sayede sahte nokta uydurmadan (motion2'deki gerilemeyi önleyerek) hibrit modelde **%96.48**, pose-only modelde **%99.50** doğruluğa ulaşıldı.

3. **Özet:**  
   Eğer amacımız sadece sunucu üzerinde çalışan akademik bir benchmark yarışması olsaydı RTMPose/DWPose seçilebilirdi. Ancak amacımız **gerçek dünyada kullanılabilir, sıfır sunucu maliyetli, hafif ve gerçek zamanlı bir mobil ürün** olduğu için MediaPipe projenin teknik kısıtları için en uygun mimaridir.
