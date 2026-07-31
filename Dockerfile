FROM python:3.11-slim

WORKDIR /app

# Gerekli sistem kütüphanelerini kur
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1-mesa-glx \
    git \
    && rm -rf /var/lib/apt/lists/*

# Bağımlılıkları yükle
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Tüm proje kodlarını kopyala
COPY . .

# Hugging Face Spaces varsayılan portu 7860'tır
EXPOSE 7860

# Sunucuyu 7860 portunda çalıştır
CMD ["python", "scripts/server.py", "--config", "configs/experiment/asl_citizen_baseline.yaml", "--checkpoint", "outputs/asl_citizen_baseline/checkpoints/best_model.pt", "--port", "7860"]
