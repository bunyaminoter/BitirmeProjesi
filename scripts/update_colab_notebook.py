import json

with open("colab_egitim.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

# Update cell 8 display code so it accepts base16_val.json and base16_val_v2.json
for cell in nb["cells"]:
    if cell["cell_type"] == "code" and "base_json = " in "".join(cell["source"]):
        cell["source"] = [
            "import os, json\n",
            "import pandas as pd\n",
            "\n",
            "base_path = 'outputs/base16_val.json' if os.path.exists('outputs/base16_val.json') else 'outputs/base16_val_v2.json'\n",
            "cand_path = 'outputs/centerrepair_val.json' if os.path.exists('outputs/centerrepair_val.json') else 'outputs/centerrepair_val_v2.json'\n",
            "\n",
            "if os.path.exists(base_path) and os.path.exists(cand_path):\n",
            "    b_res = json.load(open(base_path))\n",
            "    c_res = json.load(open(cand_path))\n",
            "    \n",
            "    df = pd.DataFrame([\n",
            "        {'Model / Varyant': '1. Temel 16 Kare (base16)', 'Val Top-1 Doğruluk': f\"%{b_res['accuracy']*100:.2f}\", 'Val Macro F1': f\"{b_res['f1_macro']:.4f}\", 'Top-5': f\"%{b_res.get('top5_accuracy', 0)*100:.2f}\"},\n",
            "        {'Model / Varyant': '2. Merkez Onarımlı (centerrepair_strict)', 'Val Top-1 Doğruluk': f\"%{c_res['accuracy']*100:.2f}\", 'Val Macro F1': f\"{c_res['f1_macro']:.4f}\", 'Top-5': f\"%{c_res.get('top5_accuracy', 0)*100:.2f}\"}\n",
            "    ])\n",
            "    display(df)\n",
            "else:\n",
            "    print('Sonuç dosyaları henüz oluşmadı. Lütfen 7. adımı tamamlayın.')\n",
        ]

new_cells = [
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### 9. 🔍 Seed 42: Video Bazlı Tahminlerin Çıkarılması (Hata Analizi İçin)\n",
            "*Mevcut en iyi modellerin her video için doğru/yanlış tahminlerini JSON olarak kaydeder (~1-2 dakika sürer).*",
        ],
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "%cd /content/BitirmeProjesi\n",
            "import os\n",
            "metadata_dir = '/content/BitirmeProjesi/data/autsl'\n",
            "if not os.path.exists(os.path.join(metadata_dir, 'val_labels.csv')):\n",
            "    metadata_dir = '/content/drive/MyDrive/BitirmeProjesi/outputs/colab_training_metadata'\n",
            "os.environ['AUTSL_DATA_DIR'] = metadata_dir\n",
            "\n",
            "# 1. Temel Model (Seed 42) Video Tahminleri\n",
            "!python scripts/evaluate.py \\\n",
            "    --config configs/experiment/autsl_10_base16_screen.yaml \\\n",
            "    --cache_dir /content/BitirmeProjesi/cache/autsl_10_features \\\n",
            "    --checkpoint /content/drive/MyDrive/BitirmeProjesi/outputs/autsl_10_base16_screen/checkpoints/best_model.pt \\\n",
            "    --split val \\\n",
            "    --output outputs/base16_seed42_val_verified.json \\\n",
            "    --predictions_output outputs/base16_seed42_predictions.json\n",
            "\n",
            "# 2. Merkez Onarımlı Model (Seed 42) Video Tahminleri\n",
            "!python scripts/evaluate.py \\\n",
            "    --config configs/experiment/autsl_10_centerrepair_screen.yaml \\\n",
            "    --cache_dir /content/BitirmeProjesi/cache/autsl_10_features_centerrepair_strict \\\n",
            "    --checkpoint /content/drive/MyDrive/BitirmeProjesi/outputs/autsl_10_centerrepair_screen/checkpoints/best_model.pt \\\n",
            "    --split val \\\n",
            "    --output outputs/centerrepair_seed42_val_verified.json \\\n",
            "    --predictions_output outputs/centerrepair_seed42_predictions.json\n",
            "\n",
            'print("\\n✅ Seed 42 video bazlı tahminler başarıyla kaydedildi!")\n',
        ],
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### 10. 🎲 Seed 7 Doğrulama Eğitimi (Temel ve Merkez Onarımlı)\n",
            "*Başarının tesadüf olmadığını kanıtlamak için model farklı bir rastgele tohumla (`seed=7`) 20 epoch eğitilir.*",
        ],
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "%cd /content/BitirmeProjesi\n",
            "\n",
            "# Preflight kontrolü\n",
            "!python scripts/check_colab_inputs.py \\\n",
            "    --base_config configs/experiment/autsl_10_base16_seed7.yaml \\\n",
            "    --candidate_config configs/experiment/autsl_10_centerrepair_seed7.yaml \\\n",
            "    --base_cache /content/BitirmeProjesi/cache/autsl_10_features \\\n",
            "    --candidate_cache /content/BitirmeProjesi/cache/autsl_10_features_centerrepair_strict \\\n",
            '    --metadata_dir "$AUTSL_DATA_DIR" \\\n',
            "    --output outputs/seed7_preflight_colab.json\n",
            "\n",
            "# 1. Temel Model (Seed 7) Eğitimi\n",
            "!python scripts/train.py \\\n",
            "    --config configs/experiment/autsl_10_base16_seed7.yaml \\\n",
            "    --cache_dir /content/BitirmeProjesi/cache/autsl_10_features\n",
            "\n",
            "# 2. Merkez Onarımlı Model (Seed 7) Eğitimi\n",
            "!python scripts/train.py \\\n",
            "    --config configs/experiment/autsl_10_centerrepair_seed7.yaml \\\n",
            "    --cache_dir /content/BitirmeProjesi/cache/autsl_10_features_centerrepair_strict\n",
        ],
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### 11. 📊 Seed 7 Resmi Val Seti Değerlendirmesi ve Tahminleri"
        ],
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "%cd /content/BitirmeProjesi\n",
            "\n",
            "# 1. Temel Model (Seed 7) Değerlendirmesi\n",
            "!python scripts/evaluate.py \\\n",
            "    --config configs/experiment/autsl_10_base16_seed7.yaml \\\n",
            "    --checkpoint /content/drive/MyDrive/BitirmeProjesi/outputs/autsl_10_base16_seed7/checkpoints/best_model.pt \\\n",
            "    --split val \\\n",
            "    --cache_dir /content/BitirmeProjesi/cache/autsl_10_features \\\n",
            "    --output outputs/base16_seed7_val.json \\\n",
            "    --predictions_output outputs/base16_seed7_predictions.json\n",
            "\n",
            "# 2. Merkez Onarımlı Model (Seed 7) Değerlendirmesi\n",
            "!python scripts/evaluate.py \\\n",
            "    --config configs/experiment/autsl_10_centerrepair_seed7.yaml \\\n",
            "    --checkpoint /content/drive/MyDrive/BitirmeProjesi/outputs/autsl_10_centerrepair_seed7/checkpoints/best_model.pt \\\n",
            "    --split val \\\n",
            "    --cache_dir /content/BitirmeProjesi/cache/autsl_10_features_centerrepair_strict \\\n",
            "    --output outputs/centerrepair_seed7_val.json \\\n",
            "    --predictions_output outputs/centerrepair_seed7_predictions.json\n",
        ],
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### 12. 📈 Seed 7 Karşılaştırmalı Sonuç Tablosu"
        ],
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import os, json\n",
            "import pandas as pd\n",
            "\n",
            "b_path = 'outputs/base16_seed7_val.json'\n",
            "c_path = 'outputs/centerrepair_seed7_val.json'\n",
            "\n",
            "if os.path.exists(b_path) and os.path.exists(c_path):\n",
            "    b7 = json.load(open(b_path))\n",
            "    c7 = json.load(open(c_path))\n",
            "    \n",
            "    df_seed7 = pd.DataFrame([\n",
            "        {'Model (Seed 7)': '1. Temel 16 Kare (base16)', 'Val Top-1 Doğruluk': f\"%{b7['accuracy']*100:.2f}\", 'Val Macro F1': f\"{b7['f1_macro']:.4f}\", 'Top-5': f\"%{b7.get('top5_accuracy', 0)*100:.2f}\"},\n",
            "        {'Model (Seed 7)': '2. Merkez Onarımlı (centerrepair_strict)', 'Val Top-1 Doğruluk': f\"%{c7['accuracy']*100:.2f}\", 'Val Macro F1': f\"{c7['f1_macro']:.4f}\", 'Top-5': f\"%{c7.get('top5_accuracy', 0)*100:.2f}\"}\n",
            "    ])\n",
            "    display(df_seed7)\n",
            "else:\n",
            "    print('Seed 7 sonuç dosyaları henüz oluşmadı. Lütfen 11. adımı tamamlayın.')\n",
        ],
    },
]

nb["cells"].extend(new_cells)

with open("colab_egitim.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("Updated colab_egitim.ipynb successfully. Total cells:", len(nb["cells"]))
