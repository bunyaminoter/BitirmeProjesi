"""Build the stand-alone AUTSL model training progress report."""

from pathlib import Path
from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_CELL_VERTICAL_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.style import WD_STYLE_TYPE
from docx.shared import Cm, Pt, RGBColor
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

ROOT = Path(__file__).resolve().parents[2]
PROJECT = ROOT / "BitirmeProjesi"
OUT = ROOT / "MODEL EĞİTİMİ İlerleme Raporu AUTSL.docx"

BLUE = RGBColor(25, 67, 114)
PALE = "EAF1F8"
GRID = "D9E0E8"


def cell_shade(cell, fill):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:fill"), fill)
    tcPr.append(shd)


def borders(cell):
    tcPr = cell._tc.get_or_add_tcPr()
    bd = OxmlElement("w:tcBorders")
    for edge in ("top", "left", "bottom", "right"):
        e = OxmlElement(f"w:{edge}")
        e.set(qn("w:val"), "single")
        e.set(qn("w:sz"), "5")
        e.set(qn("w:color"), GRID)
        bd.append(e)
    tcPr.append(bd)


def table(headers, rows, widths=None):
    t = doc.add_table(rows=1, cols=len(headers))
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    t.autofit = False
    for i, h in enumerate(headers):
        c = t.rows[0].cells[i]
        c.text = h
        cell_shade(c, PALE)
        for run in c.paragraphs[0].runs:
            run.bold = True
            run.font.color.rgb = BLUE
    for values in rows:
        cells = t.add_row().cells
        for i, value in enumerate(values):
            cells[i].text = str(value)
    for row in t.rows:
        row._tr.get_or_add_trPr().append(OxmlElement("w:cantSplit"))
        for i, c in enumerate(row.cells):
            c.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
            borders(c)
            if widths:
                c.width = Cm(widths[i])
            for p in c.paragraphs:
                p.paragraph_format.space_before = Pt(2)
                p.paragraph_format.space_after = Pt(2)
                for run in p.runs:
                    run.font.size = Pt(8.5)
    header = t.rows[0]._tr.get_or_add_trPr()
    rep = OxmlElement("w:tblHeader")
    rep.set(qn("w:val"), "true")
    header.append(rep)
    doc.add_paragraph()
    return t


def para(text, style=None):
    p = doc.add_paragraph(style=style)
    p.add_run(text)
    return p


def bullet(text):
    return para(text, "List Bullet")


def heading(text, level=1):
    return doc.add_heading(text, level=level)


def figure(path, caption):
    if not path.exists():
        return
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.add_run().add_picture(str(path), width=Cm(15.5))
    c = doc.add_paragraph(caption)
    c.style = "Caption"
    c.alignment = WD_ALIGN_PARAGRAPH.CENTER


doc = Document()
sec = doc.sections[0]
sec.top_margin = Cm(2.1)
sec.bottom_margin = Cm(2.0)
sec.left_margin = Cm(2.35)
sec.right_margin = Cm(2.35)
sec.header_distance = Cm(1)
sec.footer_distance = Cm(1)

styles = doc.styles
normal = styles["Normal"]
normal.font.name = "Calibri"
normal.font.size = Pt(10)
normal.font.color.rgb = RGBColor(38, 48, 58)
normal.paragraph_format.space_after = Pt(6)
normal.paragraph_format.line_spacing = 1.12
for style_name, size in (("Title", 23), ("Heading 1", 14), ("Heading 2", 11)):
    st = styles[style_name]
    st.font.name = "Calibri"
    st.font.size = Pt(size)
    st.font.bold = True
    st.font.color.rgb = BLUE if style_name != "Title" else RGBColor(26, 42, 60)
    st.paragraph_format.space_before = Pt(12 if style_name != "Title" else 0)
    st.paragraph_format.space_after = Pt(6)
    st.paragraph_format.keep_with_next = True
if "Report Lead" not in styles:
    lead = styles.add_style("Report Lead", WD_STYLE_TYPE.PARAGRAPH)
    lead.base_style = normal
    lead.font.size = Pt(11)
    lead.font.color.rgb = RGBColor(75, 91, 107)
    lead.paragraph_format.space_after = Pt(10)

title = doc.add_paragraph(style="Title")
title.alignment = WD_ALIGN_PARAGRAPH.CENTER
title.add_run("MODEL EĞİTİMİ\nİLERLEME RAPORU — AUTSL")
p = doc.add_paragraph(style="Report Lead")
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
p.add_run("10 sınıf deneyleri, doğrulanmış kararlar ve 100 sınıfa geçiş")
p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
p.add_run("22 Ekim 2026  |  Bitirme Projesi")
doc.add_paragraph()

heading("Yönetici özeti")
para("AUTSL için 10 sınıflı kişi bağımsız pilotta, resmî train/val ayrımı korunarak 16 karelik centerrepair_strict v1 ön işlemesi sabitlendi. İki seed'de aynı hibrit modelin base16 karşılaştırmasına göre doğrulama başarısı arttı. Aynı strict girdide pose-only model seed 7 ile 198/199 (%99,50) doğruya ulaştı. Yeni motion2 ön işleme adayı ise eş bütçeli eğitimde 176/199'a gerilediği için ölçeklenmeyecek.")
para("Bu yüzdeler seçilmiş 10 sınıfın 199 resmî val videosuna aittir. 100 ve 226 sınıf eğitimi ile resmî test değerlendirmesi henüz yapılmadı. Bundan sonraki pahalı koşu, sabit strict önbellekle 100 sınıf pose-only baz çizgisidir.")

heading("1. Veri ve deney sözleşmesi")
table(["Kapsam", "Veri / durum"], [
    ("AUTSL tam veri", "226 sınıf; 28.142 train, 4.418 val, 3.742 test video"),
    ("10 sınıf pilot", "1.245 train + 199 resmî val video; sınıf ID: 8, 14, 15, 20, 64, 106, 114, 143, 144, 196"),
    ("100 sınıf strict cache", "12.405 train + 1.963 val NPZ; sayım ve sıfır bayt denetimi tamam"),
    ("Değerlendirme", "Model ve checkpoint seçimi val üzerinde; resmî test kapalı"),
], [4.1, 11.4])
para("10 sınıf, ön işleme ve mimariyi düşük maliyetle sınamak için seçilmiş bir geliştirme kümesidir. Kişi bağımsız resmî train/val protokolü korunur; aynı val üzerinde çok sayıda karar verildiğinden bu pilotun doğruluğu nihai genelleme tahmini değildir.")

heading("2. Sabit giriş ve model seçenekleri")
para("centerrepair_strict v1 her videodan 16 kare örnekler. MediaPipe, 33 gövde ve her elden 21 nokta çıkarır: toplam 75 nokta ve kare başına 225 koordinat. El görüntüleri 160×160 piksele kırpılır; 1,5 kat kırpım ve sınırlı merkez onarımı kullanılır. Tespit kaybında canlı bilek merkezli kırpım, donuk son kareyi taşımayı önler. Bu giriş biçimi 100 sınıf koşusunda değişmeyecek.")

heading("Neden MediaPipe ile devam ediyoruz? Alternatif iskelet kütüphaneleri karşılaştırması", 2)
para("İşaret Dili Tanıma (SLR) literatüründe eklem çıkarımı için güncel alternatifler (MMPose/RTMPose, DWPose, OpenPose ve YOLO-Pose) ile projedeki mühendislik tercihleri incelenmiştir:")
bullet("MMPose / RTMPose (OpenMMLab): Bulanık karelerde ve zor el örtüşmelerinde yüksek keskinliğe sahiptir (WholeBody 133 nokta). Ancak ağır PyTorch/CUDA bağımlılığı, CPU üzerinde 5–12 FPS'e düşmesi ve mobil dağıtımdaki zorluğu nedeniyle uç cihaz hedefimize uymaz.")
bullet("DWPose (ECCV 2024): Üretken yapay zeka ve animasyonda SOTA eklem hassasiyetine sahiptir; fakat güçlü sunucu GPU'su gerektirir ve gerçek zamanlı istemci akışı için aşırı maliyetlidir.")
bullet("OpenPose (CMU): Tarihsel öncüdür ancak CPU'da < 1–2 FPS verecek kadar hantal ve ticari lisansı kısıtlayıcıdır (yıllık 25.000$). Yerini modern kütüphanelere bırakmıştır.")
bullet("YOLOv8 / YOLOv11 Pose: Hızlıdır fakat standart olarak yalnızca 17 gövde noktasını verir; işaret dili için şart olan 21 parmak eklemini yerel olarak içermez (ekstra el modeli kaskadı ister).")

table(["Kütüphane / Model", "Zor Poz Doğruluğu", "CPU Hızı", "Mobil / Web Uyumu", "Donanım / Maliyet", "Proje Kararı"], [
    ("MediaPipe (Google)", "İyi / Dengeli", "30+ FPS (Çok hızlı)", "Kusursuz (Android/iOS SDK, WASM)", "0 TL (İstemci tarafında)", "Kabul (Uç cihaz / gerçek zamanlı)"),
    ("MMPose / RTMPose", "Çok Yüksek", "5–12 FPS (Yavaş)", "Zor (Sunucu / GPU portu gerekir)", "Yüksek GPU maliyeti", "Elendi (Mobil gecikme / bağımlılık)"),
    ("DWPose (ECCV 2024)", "SOTA (En Yüksek)", "2–5 FPS (Çok Yavaş)", "Çok Zor (Büyük model boyutu)", "Çok Yüksek GPU maliyeti", "Elendi (Canlı kamera için aşırı ağır)"),
    ("YOLOv8 / v11 Pose", "İyi (Yalnız Gövde)", "25+ FPS (Hızlı)", "İyi (ONNX / TFLite)", "Düşük / Orta", "Elendi (21 parmak noktası yok)"),
    ("OpenPose (CMU)", "Orta", "< 2 FPS (Çok hantal)", "Kullanılamaz", "Yüksek + 25.000$ lisans", "Elendi (Eski / Lisans engeli)"),
], [2.8, 2.5, 2.4, 2.8, 2.7, 3.1])

para("Mühendislik gerekçesi: Bu projenin ve TÜBİTAK 2209-A başvurusunun nihai çıktısı, harici GPU sunucusuna bağımlı olmadan kullanıcının akıllı telefonunda (Flutter / Android) gerçek zamanlı çalışan iki yönlü bir sohbet asistanıdır. MediaPipe bu çalışma zarfında sıfır maliyetle 30+ FPS hızında çalışabilen tek pratik çözümdür. MediaPipe'ın hızlı el hareketlerindeki eksikleri ise kütüphane değiştirmeden, projede geliştirdiğimiz Strict v1 (1,5 kat bilek fallback ve merkez onarımı) veri mühendisliğiyle başarıyla çözülmüş; hibritte %96,48 ve pose-only'de %99,50 doğruluğa ulaşılmıştır.")

heading("Model mimarisi seçenekleri", 2)
table(["Model", "İşleyen bilgi", "10 sınıf araştırmasındaki rol"], [
    ("Hibrit RGB + pose", "Dondurulmuş ImageNet MobileNetV3-Small el RGB kolu; landmark MLP; çapraz dikkat füzyonu; TCN + Transformer zaman modeli", "base16/strict ve motion2 etkisini eş koşullarda ölçmek"),
    ("Pose-only", "75 noktanın koordinatları; landmark MLP + zaman modeli; RGB kolu kapalı", "strict girdinin düşük maliyetli, güçlü 100 sınıf adayı"),
], [3.3, 7.1, 5.1])
para("Pose-only model el ve parmak noktalarını içerir; ham el görüntüsünü işlemez. Bu yüzden 10 sınıfta yüksek puan alması, RGB'nin 100 veya 226 sınıfta gereksiz olduğunu göstermez.")

heading("3. Kontrollü 10 sınıf eğitim sonuçları")
para("Karşılaştırmalı koşular 20 epoch, batch 16, AdamW (öğrenme oranı 0,0003), cosine zamanlayıcı, üç epoch warmup ve 0,05 label smoothing ayarlarıyla yürütüldü. Tüm satırlarda aynı 199 resmî val videosu kullanıldı; raporlanan değerler val doğruluğuna göre seçilen en iyi checkpoint'e aittir.")
table(["Model / girdi", "Seed", "Doğru", "Top-1", "Macro-F1", "Top-5"], [
    ("Hibrit / base16", "42", "174/199", "%87,44", "0,87185", "198/199"),
    ("Hibrit / strict v1", "42", "177/199", "%88,94", "0,88852", "199/199"),
    ("Hibrit / base16", "7", "188/199", "%94,47", "0,94517", "199/199"),
    ("Hibrit / strict v1", "7", "192/199", "%96,48", "0,96455", "199/199"),
    ("Pose-only / strict v1", "7", "198/199", "%99,50", "0,99500", "199/199"),
    ("Hibrit / motion2", "7", "176/199", "%88,44", "0,88414", "199/199"),
], [4.2, 1.4, 2.1, 2.1, 2.6, 2.0])
para("Strict v1, base16'yı seed 42'de 3 video (+1,51 yüzde puan), seed 7'de 4 video (+2,01 puan) geçti. Ancak seed 42'den 7'ye geçişte iki modelde de 14–15 videoluk fark oluştu. Seed numarasının küçük olması performans nedeni değildir; başlangıç ve veri sırası duyarlılığına işaret eder.")
para("Strict hibritte seed 7'nin en iyi sonucu 19. epoch'ta 192/199; 20. epoch'ta 187/199 idi. Pose-only seed 7, 17. epoch'ta 198/199'a ulaştı. En iyi checkpoint kullanımı ve val ile model seçimi sonuçların yorumunda önemlidir.")

heading("4. Hata analizi ve mimari kararı")
para("Aynı strict cache ve seed 7'de pose-only, hibritten net 6 video fazla bildi (+3,02 puan). Eşleştirilmiş karşılaştırmada hibritin yedi hatasını düzeltti, bir yeni hata üretti: signer16_sample317_color, gerçek sınıf 114, tahmin 106. İşaretçi 11 kırılımı hibritte 26/30 iken pose-only'de 30/30 oldu.")
para("Bu tek ablation, farkın RGB bilgisi, dondurulmuş omurga, füzyon ya da optimizasyon dinamiğinin hangisinden kaynaklandığını ayıramaz. İşaretçi 11'deki giysi ve arka planın tek başına MediaPipe tespitini bozduğu hipotezi de doğrulanmış değil. 100 sınıfta benzer parmak biçimli işaretlerin karışıp karışmadığı doğrudan ölçülecek.")
figure(PROJECT / "outputs/colab_poseonly_seed7_review/poseonly_seed7_curves.png", "Şekil 1. Pose-only, strict v1, seed 7: 20 epoch eğitim/doğrulama eğrileri. En iyi val checkpoint'i 17. epoch'ta.")

heading("5. Motion2 denemesi: neden ölçeklenmiyor?")
para("Sıfır el görülen karelerde üst gövde ROI ve en az iki bitişik örnek kare koşulu uygulayan motion2, train'de 82 videonun 269 karesine, val'de 9 videonun 26 karesine el tespiti ekledi. Görsel kalite denetimindeki bu artışın sınıflandırmaya yansıması eş bütçeli seed 7 hibrit eğitiminde sınandı.")
table(["Aynı koşul, farklı cache", "Doğru / 199", "Top-1", "Macro-F1"], [
    ("centerrepair_strict v1", "192", "%96,48", "0,96455"),
    ("motion2", "176", "%88,44", "0,88414"),
    ("Fark", "−16 video", "−8,04 puan", "−0,08041"),
], [6.1, 3.0, 3.1, 3.3])
para("Strict'in doğru bildiği 17 video motion2'de yanlış, ters yönde yalnız bir video doğru oldu. Değişen 9 val videosunda sonuç 9/9'dan 7/9'a düştü; değişmeyen 190 videoda da net 14 doğru kaybedildi. Bu, train girdilerinin değişmesinin modelin genel öğrenmesini etkileyebildiğini gösterir. Yerel bağımsız yeniden değerlendirme 199 video etiketinin tamamında Drive çıktısıyla aynı sonucu verdi.")
para("Motion2 doğrulama eğrisi 20. epoch sonunda hâlâ yükseliyordu. Daha uzun eğitimle hiçbir zaman yetişemeyeceği iddia edilmiyor; mevcut eşit 20 epoch ve sınırlı Colab bütçesinde ikinci pahalı deneye gerekçe oluşturmadı. Karar: strict v1 korunur, motion2 yalnız inceleme kaydı olarak kalır.")

heading("6. 100 sınıf yol haritası ve karar kapıları")
table(["Aşama", "Yapılacak iş", "Geçiş ölçütü"], [
    ("1. Ön kontrol", "Sabit 100 sınıf listesi/manifesti, 12.405 + 1.963 strict NPZ, resmî ID ve split, tek batch bellek/süre denetimi", "Eksik/bozuk giriş yok; koşu süresi ve üst epoch sınırı bütçeye sığıyor"),
    ("2. İlk tam koşu", "Strict v1 ile 100 sınıf pose-only; en iyi/son checkpoint ve kaldığı yerden devam kaydı", "Tüm val videoları için olasılık ve tahminler korunuyor"),
    ("3. Hata inceleme", "Sınıf/işaretçi matrisi, karışan çiftler, parmak ayrıntısı ve düşük nokta kalitesi kırılımı", "İkinci model için somut eksik bilgi türü gösteriliyor"),
    ("4. İkinci koşu kararı", "RGB ayrıntısı açıkça eksikse eğitilebilir RGB+pose; aksi halde bütçeyi 226 sınıfa sakla", "Eş val üzerinde pose yanlışlarını kurtarma ve pose doğrularını bozma sayısı ölçülüyor"),
    ("5. Tam hedef", "Seçilen tarifi 226 sınıf resmî train/val üzerinde eğit; kararları dondurunca test et", "226 sınıf doğrulama ve kilitli test ayrı raporlanıyor"),
], [2.6, 7.3, 5.6])
para("100 sınıf için en fazla iki tam eğitim bütçesi var. İlk koşuyu 10 sınıftaki 20 epoch'a otomatik bağlamayacağız; bir batch maliyeti ve eğri davranışıyla üst sınır önceden belirlenecek. İkinci koşu varsayılan değil. Seçici uzman/kaskad da iki modelin aynı videolardaki olasılıkları ve cihaz gecikmesi görülmeden ana mimari olarak kabul edilmeyecek.")

heading("7. Sonuç ve raporlama sınırı")
para("Bugünkü somut karar: Ön işleme centerrepair_strict v1; 10 sınıftaki en güçlü çalışma adayı pose-only seed 7; 100 sınıfta ilk tam koşu strict pose-only. Hibrit ve motion2 sonuçları karşılaştırma kanıtı olarak saklanır. 10 sınıfın %99,50'si 100/226 sınıfa veya canlı kullanıcı başarısına yansıtılmaz. Canlı uygulama için bilinmeyen işaret reddi, sürekli işaret bölütleme, yeni kişiler/kameralar ve cihaz gecikmesi ayrıca sınanmalıdır.")

heading("Kaynak kayıtları", 2)
for item in [
    "docs/neden_mediapipe_iskelet_karsilastirmasi.md — iskelet omurgası tercihi ve alternatif kütüphaneler karşılaştırması",
    "docs/colab_seed7_sonuclari_2026-10-20.md — iki seed hibrit karşılaştırması ve 199 val tahmin denetimi",
    "docs/poseonly_seed7_sonuclari_2026-10-20.md — pose-only eşleşmiş analiz ve sınırlamalar",
    "docs/AUTSL_10_motion2_egitim_karari_2026-10-22.md — motion2 yeniden değerlendirme ve karar",
    "docs/100_sinif_iki_kosu_butce_plani_2026-10-20.md — 100 sınıf bütçe/karar planı",
]:
    bullet(item)

footer = sec.footer.paragraphs[0]
footer.alignment = WD_ALIGN_PARAGRAPH.CENTER
footer.add_run("AUTSL • Model eğitimi ilerleme raporu • 22.10.2026")

doc.save(OUT)
print(OUT)
