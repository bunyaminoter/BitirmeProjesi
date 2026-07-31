import os
import sys
import time
import tempfile
from pathlib import Path
from contextlib import asynccontextmanager

import cv2
import numpy as np
import torch
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from src.core.config import load_config
from src.models.hybrid_model import HybridASLModel
from src.preprocessing.hand_cropper import HandCropper
from src.preprocessing.mediapipe_extractor import MediaPipeLandmarkExtractor
from src.utils.device import get_device

# Varsayılan yapılandırma ve model yolları
CONFIG_PATH = project_root / "configs" / "experiment" / "asl_citizen_100.yaml"
CHECKPOINT_PATH = project_root / "outputs" / "asl_citizen_100" / "checkpoints" / "best_model.pt"

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
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return []

    if total <= num_frames:
        indices = list(range(total))
    else:
        indices = np.linspace(0, total - 1, num_frames, dtype=int).tolist()

    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    cap.release()
    return frames

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
            print(f"WARNING: Checkpoint not found at {CHECKPOINT_PATH}. Model is randomly initialized!")
            
        model.eval()
    except Exception as e:
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
            
        # 3. MediaPipe ile özellikleri (iskelet & el görüntüleri) çıkart
        extractor = MediaPipeLandmarkExtractor(enable_face=config.model.landmark_encoder.include_face)
        cropper = HandCropper(output_size=tuple(config.model.hand_encoder.input_size), padding_ratio=0.2)
        
        all_pose, all_left_crops, all_right_crops = [], [], []
        
        for frame in frames:
            result = extractor.extract(frame)
            
            if result.pose is not None:
                all_pose.append(result.pose.reshape(-1))
            else:
                all_pose.append(np.zeros(33 * 3, dtype=np.float32))
                
            left_crop, right_crop = cropper.crop_hands(frame, result)
            all_left_crops.append(left_crop.image)
            all_right_crops.append(right_crop.image)
            
        extractor.close()
        
        # 4. Veriyi tensor formatına getir ve pad (doldurma) yap
        def pad_list(lst, target_len, default_fn):
            while len(lst) < target_len:
                lst.append(default_fn())
            return lst[:target_len]

        num_frames = config.dataset.num_frames
        lm_dim = config.model.landmark_encoder.input_dim
        
        all_pose = pad_list(all_pose, num_frames, lambda: np.zeros(lm_dim, dtype=np.float32))
        
        h, w = config.model.hand_encoder.input_size
        all_left_crops = pad_list(all_left_crops, num_frames, lambda: np.zeros((h, w, 3), dtype=np.uint8))
        all_right_crops = pad_list(all_right_crops, num_frames, lambda: np.zeros((h, w, 3), dtype=np.uint8))
        
        pose_tensor = torch.from_numpy(np.stack(all_pose, axis=0)).float().unsqueeze(0).to(device)
        
        left_imgs = np.stack(all_left_crops, axis=0).astype(np.float32) / 255.0
        left_imgs = left_imgs.transpose(0, 3, 1, 2)
        left_tensor = torch.from_numpy(left_imgs).float().unsqueeze(0).to(device)
        
        right_imgs = np.stack(all_right_crops, axis=0).astype(np.float32) / 255.0
        right_imgs = right_imgs.transpose(0, 3, 1, 2)
        right_tensor = torch.from_numpy(right_imgs).float().unsqueeze(0).to(device)
        
        mask = torch.ones(1, num_frames, dtype=torch.bool, device=device)
        mask[0, len(frames):] = False
        
        batch = {
            "pose_landmarks": pose_tensor,
            "left_hand_images": left_tensor,
            "right_hand_images": right_tensor,
            "mask": mask,
        }
        
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
