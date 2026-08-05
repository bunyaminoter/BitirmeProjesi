# 🚀 İşaret Dili Tanıma (SLR) Teknolojileri Gelişim ve Literatür Planı (2020 - 2026)

Bu doküman, İşaret Dili Tanıma (Sign Language Recognition - SLR) alanındaki teknolojilerin geçmişten günümüze (2024, 2025 ve öngörülen 2026 trendleri) evrimini ve **bu projeye entegre edilebilecek State-of-the-Art (SOTA) yaklaşımları** listelemektedir.

---

## 📅 1. SLR Teknolojilerinin Evrimi

### 🔙 Geçmiş Yaklaşımlar (2015 - 2021)
*   **Donanım Odaklı (Eldivenler vb.):** İlk çalışmalar sensörlü eldivenler veya derinlik (RGB-D) kameraları (Kinect) üzerinden yürütülüyordu. Kullanışlılık ve erişilebilirlik çok düşüktü.
*   **Geleneksel Makine Öğrenmesi (HMM):** Özelliklerin manuel çıkarılıp (el rengi takibi) Hidden Markov Model (HMM) veya SVM ile sınıflandırıldığı dönem.
*   **Sade 3D-CNN'ler (C3D, I3D):** Derin öğrenmenin yükselişiyle videoların doğrudan 3D CNN'lere verildiği dönem. Aşırı yüksek işlem gücü gerektiriyordu ve arka plan gürültüsünden (kıyafet, ışık) çok etkileniyordu.

### 🌟 Modern Dönem (2022 - 2024)
*   **İskelet ve Poz Odaklı Modeller:** MediaPipe, OpenPose gibi sistemlerin yaygınlaşmasıyla RGB görüntülerden ziyade eklem (landmark) verisinin kullanıldığı GCN (Graph Convolutional Networks) dönemi başladı (Mevcut projemizde entegre ettiğimiz **ST-GCN** bu dönemin en güçlü standartlarındandır).
*   **Attention (Dikkat) Mekanizmaları:** Klasik evrişimsel ağlar yerine Vision Transformer (ViT) ve Swin Transformer modellerinin kullanımı yaygınlaştı.
*   **Multimodal Fusion (Çoklu Modalite):** Sadece el değil, yüz mimikleri (non-manual features), iskelet ve ellerin bir arada (Hybrid) eğitilmesi standart hale geldi (Projemizdeki **Cross-Attention Fusion** mekanizması bu yaklaşıma dayanır).

### 🚀 Gelecek ve Güncel SOTA Teknolojileri (2025 - 2026)
Güncel makaleler ve araştırmalar, daha hafif ama daha bağlamsal modellere yönelmektedir:

1.  **Vision Mamba (SSM - State Space Models):** 
    *   2025 yılının en büyük yeniliklerinden biridir. Transformer'ların yüksek bellek tüketimi (kare sayısının karesiyle orantılı - $O(N^2)$) problemini çözen, çizgisel ($O(N)$) karmaşıklığa sahip Mamba mimarisi. Özellikle uzun cümle çevirilerinde (Continuous Sign Language) mükemmel sonuçlar vermektedir.
2.  **CBAM (Convolutional Block Attention Module):** 
    *   Özellik çıkarım ağlarının (örneğin projemizdeki EfficientNet) arka planı tamamen yok sayıp sadece elin anatomik olarak önemli yerlerine (parmak uçları) odaklanmasını sağlayan Kanal ve Uzamsal dikkat modülü.
3.  **TCN-Transformer Hibritleri (Local-Global Temporal):**
    *   Transformer'lar videonun tamamına aynı anda bakarak genel bağlamı anlar, ancak ardışık 2-3 frame arasındaki küçük hız veya yön değişimlerini kaçırabilir. 2025 mimarileri, Transformer'dan hemen önce **Temporal Convolutional Network (1D-CNN)** kullanarak yerel (local) hareket yönünü yakalar.
4.  **Contrastive Learning (SupCon):**
    *   Birbirine çok benzeyen el hareketlerinin vektör uzayında birbirinden yapay olarak uzaklaştırılmasını sağlayan modern bir kayıp fonksiyonu (Loss function) eğitimidir.

---

## 🛠️ Projemiz İçin Uygunluk ve Entegrasyon Analizi

Araştırmalar sonucunda elde edilen SOTA teknolojilerin mevcut kod tabanımıza uygulanabilirliği analiz edilmiştir:

| Teknoloji | SLR Etkisi | Projeye Uygunluk | Entegrasyon Kararı |
| :--- | :--- | :--- | :--- |
| **Vision Mamba (Vmamba)** | Çok Yüksek | Düşük (Yeni kütüphaneler CUDA uyumsuzlukları yaratabilir, Colab T4'te kurulumu zordur) | ❌ *Reddedildi (Riskli)* |
| **CBAM Attention** | Yüksek | Çok Yüksek (Mevcut EfficientNet encoder'ımızın sonuna rahatlıkla eklenebilir) | ✅ *Kabul Edildi* |
| **TCN + Transformer** | Çok Yüksek | Çok Yüksek (PyTorch standart kütüphaneleriyle yerel-genel temporal akış mükemmel yakalanır) | ✅ *Kabul Edildi* |
| **Pose Jittering (Veri Artırma)** | Orta/Yüksek | Çok Yüksek (İskelet modelinin ezberlemesini engelleyen modern bir veri artırma tekniğidir) | ✅ *Kabul Edildi* |
| **Yüz / Mimik Entegrasyonu**| Çok Yüksek | Orta (Modeli çok ağırlaştırır, 468 yüz noktası ST-GCN'i çok büyütür) | ❌ *Şimdilik Beklemede* |

---

## 🎯 Planlanan Geliştirmeler (2025 SLR Standartlarına Uyum)

Bu dokümandaki araştırma sonucunda projemizin **"2025 SOTA (State of the Art)"** standartlarına ulaşması için aşağıdaki **3 modülün** projeye kodlanarak eklenmesine karar verilmiştir:

### 1. Spatial-Channel Attention (CBAM) Modülü
*   **Dosya:** `src/models/attention/cbam.py`
*   **İşlev:** El resimlerinden özellik çıkaran CNN ağı (EfficientNet), arka planı ve gürültüyü filtreleyerek yalnızca parmakların veya elin işaret yapan kısımlarına odaklanacak.

### 2. Local-Global Temporal Modeli (TCN-Transformer)
*   **Dosya:** `src/models/temporal/tcn_transformer.py`
*   **İşlev:** Mevcut Transformer modelimizin önüne 1 boyutlu Temporal Convolution (TCN) katmanları eklenecek. Böylece sistem "El yukarı mı çıkıyor, aşağı mı iniyor?" gibi yerel akış yönünü matematiksel olarak daha net anlayacak.

### 3. Pose Jittering (İskelet Gürültü Artırımı)
*   **Dosya:** `src/data/augmentations.py`
*   **İşlev:** MediaPipe'ın hata payını simüle etmek için eğitim sırasında eklem (landmark) noktalarına rastgele küçük kaymalar (Gaussian Noise) uygulanacak. Bu, ST-GCN modelinin aşırı öğrenmesini (overfitting) engelleyecek.

*Not: Bu eklentiler Colab T4 (Ücretsiz) veya Pro versiyonunda herhangi bir VRAM darboğazı yaratmayacak hafiflikte ama mimari açıdan çok güçlü yeniliklerdir.*
