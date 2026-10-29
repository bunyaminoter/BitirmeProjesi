import os
import sys
import time
import tempfile
from pathlib import Path
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from src.core.config import load_config
from src.models.hybrid_model import HybridASLModel
from src.preprocessing.inference_pipeline import prepare_inference_batch, sample_video_frames
from src.utils.device import get_device

# Varsayılan yapılandırma ve model yolları
CONFIG_PATH = project_root / "configs" / "experiment" / "autsl_10_mobilenet.yaml"
CHECKPOINT_PATH = project_root / "outputs" / "autsl_10_mobilenet" / "checkpoints" / "best_model.pt"

# Global değişkenler (modelin RAM'de sürekli açık kalması için)
model = None
config = None
device = None
idx_to_class = {}

def load_class_names(class_list_file: str) -> dict:
    path = Path(class_list_file)
    if not path.is_absolute():
        path = project_root / path
    if not path.exists():
        return {}

    mapping = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(maxsplit=1)
            if len(parts) == 2:
                mapping[int(parts[0])] = parts[1]
    return mapping

def sample_frames_from_video(video_path: str, num_frames: int) -> list:
    return sample_video_frames(video_path, num_frames)

# Sunucu başlarken modeli belleğe yükle
@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, config, device, idx_to_class
    
    print(f"Loading configuration from {CONFIG_PATH}...")
    try:
        config = load_config(str(CONFIG_PATH))
        device = get_device(config.device)
        print(f"Using device: {device}")
        
        idx_to_class = load_class_names(config.dataset.class_list_file)
        print(f"Loaded {len(idx_to_class)} classes.")
        
        model = HybridASLModel(config.model)
        model.to(device)
        
        if CHECKPOINT_PATH.exists():
            print(f"Loading checkpoint from {CHECKPOINT_PATH}...")
            checkpoint = torch.load(str(CHECKPOINT_PATH), map_location=device, weights_only=False)
            model.load_state_dict(checkpoint["model_state_dict"])
            print("Model loaded successfully!")
        else:
            raise FileNotFoundError(CHECKPOINT_PATH)
            
        model.eval()
    except Exception as e:
        model = None
        print(f"Error initializing model: {e}")
        
    yield
    print("Shutting down model server...")
    model = None

# FastAPI uygulamasını oluştur
app = FastAPI(title="Hybrid ASL Backend", lifespan=lifespan)

# CORS ayarları (Telefondan gelecek isteklere izin vermek için)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
async def health():
    if model is None:
        return {"status": "offline", "message": "Model could not be loaded."}
    return {"status": "online"}

@app.get("/classes")
async def get_classes():
    return {"classes": idx_to_class}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if model is None or config is None:
        raise HTTPException(status_code=503, detail="Model server is not ready")
        
    start_time = time.time()
    
    # 1. Videoyu geçici bir dosyaya kaydet
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
        content = await file.read()
        temp_video.write(content)
        temp_video_path = temp_video.name
        
    try:
        # 2. Videodan frame'leri çek
        frames = sample_frames_from_video(temp_video_path, config.dataset.num_frames)
        if not frames:
            raise HTTPException(status_code=400, detail="Could not read video frames.")
            
        batch = prepare_inference_batch(frames, config, device)

        # 5. Modeli çalıştır
        with torch.no_grad():
            logits = model(batch)
            probs = torch.softmax(logits, dim=-1)
            
        top_k = min(5, config.model.num_classes)
        top_probs, top_indices = probs.topk(top_k, dim=-1)
        
        top_probs = top_probs.squeeze(0).cpu().numpy()
        top_indices = top_indices.squeeze(0).cpu().numpy()
        
        predictions = []
        for idx, prob in zip(top_indices, top_probs):
            class_idx = int(idx)
            class_name = idx_to_class.get(class_idx, f"class_{class_idx}")
            predictions.append({
                "label": class_name,
                "confidence": float(prob),
                "classIndex": class_idx
            })
            
        inference_time_ms = int((time.time() - start_time) * 1000)
        
        return JSONResponse({
            "predictions": predictions,
            "inferenceTimeMs": inference_time_ms
        })
        
    finally:
        # Geçici video dosyasını temizle
        if os.path.exists(temp_video_path):
            try:
                os.remove(temp_video_path)
            except:
                pass

if __name__ == "__main__":
    import uvicorn
    # Doğrudan python src/api/main.py ile de çalıştırabilmek için:
    uvicorn.run("src.api.main:app", host="0.0.0.0", port=8000, reload=True)
