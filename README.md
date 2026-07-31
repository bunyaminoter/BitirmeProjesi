---
title: ASL API
emoji: 🤟
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
---

# 🤟 Hibrit Gerçek Zamanlı ASL Tanıma Sistemi

**RGB el özellikleri ve vücut landmark özelliklerini (işaretlerini) birlikte kullanarak Amerikan İşaret Dilini (ASL) tanıyan, araştırmaya yönelik ve üretime (production) hazır kalitede bir altyapı.**

![Python](https://img.shields.io/badge/Python-3.11-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-red)
![MediaPipe](https://img.shields.io/badge/MediaPipe-Tasks_API-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## 📋 Proje Özeti

Bu sistem, Amerikan İşaret Dili (ASL) hareketlerini videolardan tanımak için **hibrit bir mimari** kullanır:

- **RGB El Özellikleri** — Görüntüden kırpılmış eller bir CNN modeli (evrişimli sinir ağı) ile işlenerek parmak şekilleri, avuç içi yönelimi ve ince detaylar yakalanır.
- **Vücut Landmark Özellikleri** — Vücut iskeleti (Pose) ve (isteğe bağlı) yüz işaret noktaları ile kolların konumu, omuz açıları ve gövdenin yönü yakalanır.

### Mimari

```text
Video → Kare (Frame) Örnekleme → MediaPipe Tasks API
                              ↓
                    ┌─────────┴─────────┐
                    │                   │
               Boru Hattı A         Boru Hattı B
            (Vücut İskeleti)      (RGB El Kırpmaları)
                    │                   │
            Landmark Encoder     Hand CNN Encoder(s)
                    │                   │
                    └─────┬─────────────┘
                          │
                   Özellik Birleştirme (Fusion)
                          │
                    Zamana Bağlı Model (Temporal)
                          │
                     Sınıflandırma
                          │
                   İşaret Çıktısı (Kelime)
```

## 🏗️ Proje Yapısı

```text
├── configs/                    # YAML yapılandırma dosyaları
│   ├── experiment/            # Hazır deney yapılandırmaları
│   ├── dataset/               # Veri seti ayarları
│   ├── model/                 # Model bileşen ayarları
│   ├── training/              # Eğitim hiperparametreleri
│   └── augmentation/          # Veri artırma (Augmentation) ayarları
│
├── src/                       # Ana kaynak kodlar (Source)
│   ├── core/                  # Kayıt (Registry) sistemi, tip tanımlamaları
│   ├── data/                  # Veri setleri, örnekleyiciler (samplers)
│   ├── preprocessing/         # MediaPipe, el kırpma, önbellekleme (caching)
│   ├── models/                # Kodlayıcılar (Encoders), temporal ve hibrit modeller
│   ├── training/              # Eğitim döngüsü (Trainer), callback ve kayıp (loss) fonksiyonları
│   ├── evaluation/            # Metrik hesaplamaları (Accuracy, F1), değerlendirici
│   ├── tracking/              # TensorBoard ve CSV takipçileri
│   ├── export/                # ONNX formatında dışa aktarma (deploy için)
│   └── utils/                 # Loglama, rastgelelik kontrolü (seed), cihaz tespiti
│
├── scripts/                   # Çalıştırılabilir Ana Betikler
│   ├── train.py              # Model eğitimi
│   ├── evaluate.py           # Test seti üzerinde değerlendirme
│   ├── inference.py          # Tek bir video üzerinde test etme
│   ├── preprocess.py         # Veri setini önceden işleme (MediaPipe özelliklerini çıkarma)
│   └── export_onnx.py        # Modeli ONNX olarak kaydetme
│
├── tests/                     # PyTest birim testleri (Unit Tests)
├── notebooks/                 # Veri inceleme için Jupyter Notebook'lar
├── requirements.txt           # Gerekli kütüphaneler
└── pyproject.toml            # Proje ayarları ve linter yapılandırmaları
```

## 🚀 PyCharm İçin Hızlı Başlangıç

Bu projeyi **PyCharm** üzerinden çalıştırmak için aşağıdaki adımları sırayla izleyin. Komutları PyCharm'ın alt kısmında bulunan **Terminal (PowerShell)** penceresine yazacaksınız.

### 1. Sanal Ortam (Virtual Environment) Oluşturma ve Aktifleştirme

PyCharm genelde projeyi açtığınızda otomatik olarak bir sanal ortam (`venv`) oluşturur. Eğer oluşturmadıysa veya manuel yapmak isterseniz şu komutları sırayla çalıştırın:

```powershell
# Python sanal ortamını (venv) oluşturun
python -m venv venv

# Sanal ortamı aktifleştirin (Eğer "Yetkisiz Erişim" hatası alırsanız PowerShell'i yönetici olarak açıp: Set-ExecutionPolicy RemoteSigned -Scope CurrentUser komutunu bir kez çalıştırın)
.\venv\Scripts\Activate.ps1
```

*(Aktifleştirildiğinde terminal satırının başında `(venv)` yazısını görmelisiniz.)*

### 2. Gerekli Kütüphanelerin Kurulması

```powershell
pip install -r requirements.txt
```

### 3. Veri Setinin Hazırlanması (Ön İşleme - Preprocess)

MediaPipe işlemlerinin eğitim sırasında tekrar tekrar yapılması eğitimi çok yavaşlatır. Bu yüzden eğitimden **önce** verileri bir kez işleyip önbelleğe (`cache/`) kaydediyoruz.

```powershell
python scripts/preprocess.py --config configs/experiment/asl_citizen_100.yaml --cache_dir cache/features/
```

### 4. Eğitimi Başlatma

```powershell
python scripts/train.py --config configs/experiment/asl_citizen_100.yaml
```

### 5. Modeli Değerlendirme (Test)

```powershell
python scripts/evaluate.py --config configs/experiment/asl_citizen_100.yaml --checkpoint outputs/asl_citizen_100/checkpoints/best_model.pt
```

### 6. Tek Bir Video Üzerinde Deneme (Inference)

```powershell
python scripts/inference.py --config configs/experiment/asl_citizen_100.yaml --checkpoint outputs/asl_citizen_100/checkpoints/best_model.pt --video videolar_klasoru/ornek_video.mp4
```

---

## 🧪 Sistemin Çalışıp Çalışmadığını Test Etmek (Küçük Veri Seti İle Deneme)

Bütün bir veri setini (100 veya 1000 kelimelik versiyonları) doğrudan eğitmek bilgisayarınızın gücüne göre saatler hatta günler sürebilir. Sistemi tam anlamıyla eğitmeye başlamadan önce; kodların, veri akışının ve eğitim döngüsünün hatasız çalıştığını doğrulamak için **ufak bir alt kümeyle (örneğin sadece ilk 5 kelimeyle)** deneme yapmanız en iyi yoldur.

Sistem kodlaması (`src/data/datasets/asl_citizen_dataset.py`), YAML dosyasında belirttiğiniz sınıf (kelime) sayısına göre verileri otomatik olarak filtreleyecek şekilde tasarlanmıştır.

1. `configs/experiment/asl_citizen_100.yaml` dosyasını açın.
2. `dataset` ayarlarındaki `num_classes` değerini **5** olarak değiştirin:
   ```yaml
   dataset:
     name: "asl_citizen"
     annotation_file: "asl_citizen_100.json"
     num_classes: 5   # Sadece listedeki ilk 5 kelimenin videoları alınacak
   ```
3. Aynı şekilde `model` ayarlarındaki çıktı sayısını da güncelleyin:
   ```yaml
   model:
     num_classes: 5
   ```
4. Veri işleme ve eğitim komutlarını tekrar çalıştırın (`preprocess.py` ve `train.py`). Veri setinin çok küçük bir kısmı kullanılacağı için eğitim birkaç dakika içinde bitecektir. 

Eğer bu süreç hatasız tamamlanıyorsa projenizin mimarisi tamamen çalışıyor demektir. Sonrasında bu değerleri tekrar eski haline (`100`) çekerek orijinal tam eğitime geçebilirsiniz.

---

## 📂 JSON Dosyaları ve Veri Seti Mantığı

Eğitim sırasında verilerin hangi videoda olduğu ve bu videolarda hangi hareketin yapıldığı JSON ve TXT dosyaları üzerinden yönetilir. Modelin verileri nasıl okuduğunu anlamak için bu dosyaların mantığı önemlidir:

- **`asl_citizen_class_list.txt`**: Bu dosya, kelimelerin (sınıfların) sayısal indeks karşılıklarını tutar (Örnek: `0 book`, `1 drink`). Model metinlerden değil sayılardan anladığı için, tahminleri yaparken bu indeksleri (0, 1, 2...) kullanır.
- **`asl_citizen_100.json`**: Model eğitiminde **ana rehber olarak kullanılan** dosyadır. Dosyanın içeriği aşağıdaki gibidir:
  ```json
  "69241": {
      "subset": "train",
      "action": [0, 1, -1]
  }
  ```
  - **`69241`**: Bu ID, projenizde yer alan `videos/69241.mp4` adlı videoyu ifade eder.
  - **`subset`**: Bu videonun eğitim (`train`), doğrulama (`val`) veya test (`test`) verisi olarak mı kullanılacağını belirtir. Model eğitimde `train` videolarını kullanır ve `val` videoları ile başarısını ölçer.
  - **`action: [0, 1, -1]`**: 
    - `0`: Videodaki hareketin Sınıf İndeksini belirtir (`asl_citizen_class_list.txt` dosyasına göre 0 = "book").
    - `1`: İşaretin başladığı kare (frame) numarası.
    - `-1`: İşaretin bittiği kare numarası (-1 ise videonun sonuna kadar işaret dili devam ediyor demektir).

Model `asl_citizen_dataset.py` dosyası içerisinde bu JSON yapısını satır satır okur, `subset` bilgisiyle veriyi böler ve videoları `action` dizisindeki sınıf etiketine eşleyerek eğitime sokar. Eğer veri setinize kendi özel kelimelerinizi ve videolarınızı eklerseniz, yapmanız gereken tek şey JSON dosyasına bu formata uygun yeni bir kayıt girmek ve kelimeyi TXT listesine eklemektir.

---

## 🔧 Tasarım Prensipleri

| Prensip | Uygulanış Biçimi |
|:---|:---|
| **Modüler Yapı** | Her bileşen (kodlayıcılar, modeller vb.) `Registry` sistemi sayesinde kolayca değiştirilebilir. |
| **Konfigürasyon (YAML) Odaklı** | Kod içine sabitlenmiş (hardcoded) ayar yoktur; tüm eğitim ayarları YAML dosyalarından yönetilir. |
| **Araştırmaya Uygun** | Yeni birleştirme (fusion) stratejileri, zamansal (temporal) modeller eklemek için sadece yeni bir sınıf yazıp `@REGISTRY.register` demeniz yeterlidir. |
| **Üretime (Production) Hazır** | Modelin mobil veya web uygulamalarına aktarılması (deployment) için ONNX formatında dışa aktarma (export) desteği mevcuttur. |
| **SOLID Kodlama** | Soyut taban sınıfları (Abstract Base Classes) kullanılmış ve temiz kod mimarisi hedeflenmiştir. |

## ⚠️ Önemli Geliştirici Notları

- **MediaPipe Holistic Kullanımdan Kaldırıldı (Deprecated):** Bu sebeple proje Google'ın yeni ve güncel altyapısı olan **MediaPipe Tasks API** üzerine kurulmuştur. İşlemler PoseLandmarker, HandLandmarker ve FaceLandmarker üzerinden bağımsız yürütülür.
- Yüz işaret noktaları (Face Landmarks) kelime bazlı işaret dili tanımada çok fazla veri gürültüsü yarattığı için şimdilik kapalı durumdadır. İstenirse YAML yapılandırması üzerinden (`include_face: true`) aktif edilebilir.

## 📝 Yol Haritası ve Yapılacaklar (To-Do)

Projenin temel mimarisi, iskeleti ve veri akışı tamamen hazırdır. Komut dosyaları çalışmaktadır ancak "iç mantıkları" (derin öğrenme döngüleri ve görüntü işleme algoritmaları) bilerek sonraya bırakılmıştır.

Bundan sonraki süreçte sırasıyla tamamlanacak işler şunlardır:

- [x] **1. MediaPipe Veri Çıkarma Mantığı:** `src/preprocessing/mediapipe_extractor.py` içine videoları açıp MediaPipe Tasks API ile iskelet (pose) ve el (hand) koordinatlarını çıkaran kodun yazılması.
- [x] **2. El Kırpma (Hand Cropping):** `src/preprocessing/hand_cropper.py` içine RGB videodan ellerin, MediaPipe koordinatlarına göre dinamik olarak kırpılma algoritmalarının yazılması.
- [x] **3. Gerçek CNN Modellerinin Yüklenmesi:** `src/models/encoders/hand_cnn_encoder.py` içerisine `torchvision` üzerinden gerçek ResNet/MobileNet vb. derin öğrenme omurgalarının (backbone) dahil edilmesi.
- [x] **4. Hibrit Model İleri Beslemesi (Forward Pass):** `src/models/hybrid_model.py` içinde tüm alt dalların (sağ el, sol el, iskelet, fusion, temporal) birbiriyle veri alışverişini sağlayacak bağlantıların kurulması.
- [x] **5. Eğitim Döngüsü (Training Loop):** `src/training/trainer.py` içinde modelin ağırlıklarını güncelleyecek olan ileri besleme (forward), kayıp (loss) hesaplama ve optimizasyon adımının (`backward()`, `step()`) yazılması.
- [x] **6. Değerlendirme Döngüsü (Evaluation Loop):** `src/evaluation/evaluator.py` içinde modelin test seti üzerinde doğruluk (accuracy), F1 skoru gibi başarı oranlarını hesaplayacak döngünün yazılması.
- [x] **7. Önbellekleme Betiğinin Tamamlanması:** `scripts/preprocess.py` betiğinin içindeki TODO kısımlarının doldurularak, çıkartılan özelliklerin `.npz` formatında `cache/` dizinine kaydedilme işleminin aktif edilmesi.
- [x] **8. İlk Test Eğitimi:** Alt küme (örn. 5 sınıf/kelime) ile sistemin baştan sona çalıştırılıp hatasız eğitim yaptığının doğrulanması.

---

## 📄 Lisans

MIT License
