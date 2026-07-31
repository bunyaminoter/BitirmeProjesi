FROM python:3.11-slim

WORKDIR /app

# Gerekli sistem paketleri ve Git LFS kur
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1-mesa-glx \
    git \
    git-lfs \
    && rm -rf /var/lib/apt/lists/*

# Proje dosyalarını kopyala ve LFS büyük dosyalarını indir
COPY . .
RUN git lfs install && git lfs pull || true

# Python bağımlılıklarını kur
RUN pip install --no-cache-dir -r requirements.txt

# Hugging Face Spaces portu
EXPOSE 7860

# Sunucuyu başlat
CMD ["python", "scripts/server.py", "--config", "configs/experiment/asl_citizen_baseline.yaml", "--checkpoint", "outputs/asl_citizen_baseline/checkpoints/best_model.pt", "--port", "7860"]
