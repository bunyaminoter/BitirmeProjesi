import json

with open("colab_egitim.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

sources = ["".join(c.get("source", [])) for c in nb["cells"]]
if not any("autsl_10_poseonly_seed7" in s for s in sources):
    new_cells = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 13. 🔬 Teşhis Deneyi: Pose-Only (Sadece İskelet) Seed 7 Eğitimi\n",
                "*RGB el görüntülerini kapatıp modeli SADECE 225 iskelet noktasıyla eğitir (~4-5 dk sürer).*\n",
                "*Amaç: İşaretçi 11 hatalarının el görüntüsünden mi yoksa iskeletten mi kaynaklandığını kesin olarak teşhis etmektir.*",
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
                "os.environ['AUTSL_DATA_DIR'] = '/content/drive/MyDrive/BitirmeProjesi/outputs/colab_training_metadata'\n",
                "\n",
                "# 1. Pose-Only Eğitimi (Seed 7) - 20 Epoch\n",
                "!python scripts/train.py \\\n",
                "    --config configs/experiment/autsl_10_poseonly_seed7.yaml \\\n",
                "    --cache_dir /content/BitirmeProjesi/cache/autsl_10_features_centerrepair_strict\n",
                "\n",
                "# 2. Pose-Only Değerlendirmesi ve Tahminleri\n",
                "!python scripts/evaluate.py \\\n",
                "    --config configs/experiment/autsl_10_poseonly_seed7.yaml \\\n",
                "    --cache_dir /content/BitirmeProjesi/cache/autsl_10_features_centerrepair_strict \\\n",
                "    --checkpoint /content/drive/MyDrive/BitirmeProjesi/outputs/autsl_10_poseonly_seed7/checkpoints/best_model.pt \\\n",
                "    --split val \\\n",
                "    --output outputs/poseonly_seed7_val.json \\\n",
                "    --predictions_output outputs/poseonly_seed7_predictions.json\n",
                "\n",
                "print('\\n✅ Pose-Only modeli eğitildi ve değerlendirildi!')\n",
            ],
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 14. 📊 Teşhis Analizi: Hibrit vs Pose-Only Karşılaştırması\n",
                "*İşaretçi 11'in hatalarının kaçının iskelet modeliyle düzeldiğini raporlar.*",
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
                "# Eşleştirilmiş ikili analiz\n",
                "!python scripts/analyze_val_pairs.py \\\n",
                "    --base outputs/centerrepair_seed7_predictions.json \\\n",
                "    --candidate outputs/poseonly_seed7_predictions.json \\\n",
                "    --official_labels \"$AUTSL_DATA_DIR/val_labels.csv\" \\\n",
                "    --output outputs/hybrid_vs_poseonly_seed7_pairs.json\n",
                "\n",
                "import os, json\n",
                "import pandas as pd\n",
                "\n",
                "p_path = 'outputs/poseonly_seed7_val.json'\n",
                "c_path = 'outputs/centerrepair_seed7_val.json'\n",
                "\n",
                "if os.path.exists(p_path) and os.path.exists(c_path):\n",
                "    p_res = json.load(open(p_path))\n",
                "    c_res = json.load(open(c_path))\n",
                "    \n",
                "    df_cmp = pd.DataFrame([\n",
                "        {'Model Modu': '1. Hibrit Model (RGB El Kırpımı + İskelet)', 'Val Doğruluk': f\"%{c_res['accuracy']*100:.2f}\", 'Macro F1': f\"{c_res['f1_macro']:.4f}\", 'Top-5': f\"%{c_res.get('top5_accuracy', 0)*100:.2f}\"},\n",
                "        {'Model Modu': '2. Pose-Only (Sadece İskelet Koordinatları)', 'Val Doğruluk': f\"%{p_res['accuracy']*100:.2f}\", 'Macro F1': f\"{p_res['f1_macro']:.4f}\", 'Top-5': f\"%{p_res.get('top5_accuracy', 0)*100:.2f}\"}\n",
                "    ])\n",
                "    display(df_cmp)\n",
            ],
        },
    ]
    nb["cells"].extend(new_cells)
    with open("colab_egitim.ipynb", "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print("Added pose-only cells to colab_egitim.ipynb successfully. Total cells:", len(nb["cells"]))
else:
    print("Pose-only cells already present.")
