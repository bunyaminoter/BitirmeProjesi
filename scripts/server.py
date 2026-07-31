"""
FastAPI Server for ASL Recognition Model Inference.

Provides REST API endpoints for the Flutter mobile app to run predictions
using the trained HybridASLModel.

Usage:
    python scripts/server.py --config configs/experiment/asl_citizen_baseline.yaml --checkpoint outputs/asl_citizen_baseline/checkpoints/best_model.pt --port 8000
"""

from __future__ import annotations

import argparse
import io
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, List, Dict, Optional

import cv2
import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.core.config import load_config
from src.models.hybrid_model import HybridASLModel
from src.preprocessing.hand_cropper import HandCropper
from src.preprocessing.mediapipe_extractor import MediaPipeLandmarkExtractor
from src.utils.device import get_device
from src.utils.logging import get_logger, setup_logging


app = FastAPI(
    title="ASL Sign Language Recognition API",
    description="Backend API serving PyTorch HybridASLModel for mobile app predictions.",
    version="1.0.0",
)

# Enable CORS for Flutter mobile/web clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for model state
model: Optional[HybridASLModel] = None
config: Any = None
device: torch.device = torch.device("cpu")
idx_to_class: Dict[int, str] = {}


class PredictionItem(BaseModel):
    label: str
    confidence: float
    classIndex: int


class PredictionResponse(BaseModel):
    predictions: List[PredictionItem]
    inferenceTimeMs: int


def load_class_names(class_list_file: str) -> Dict[int, str]:
    """Load class index to class name mapping with fallback defaults."""
    default_10_classes = [
        "BITE1",
        "BREAKFAST1",
        "DARK1",
        "DEAF1",
        "DECIDE1",
        "DEMAND1",
        "DOG1",
        "HURDLE/TRIP1",
        "ROCKINGCHAIR1",
        "WHATFOR1",
    ]
    path = Path(class_list_file)
    if not path.exists():
        # Fallback to default 10 baseline classes if file is not found
        return {idx: name for idx, name in enumerate(default_10_classes)}

    mapping = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(maxsplit=1)
            if len(parts) == 2:
                mapping[int(parts[0])] = parts[1]

    if not mapping:
        return {idx: name for idx, name in enumerate(default_10_classes)}

    return mapping


def sample_frames_from_bytes(video_bytes: bytes, num_frames: int) -> List[np.ndarray]:
    """Sample frames from uploaded video bytes using a temporary file."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(video_bytes)
        tmp_path = tmp.name

    try:
        cap = cv2.VideoCapture(tmp_path)
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
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def run_model_inference(frames: List[np.ndarray], top_k_val: int = 5) -> List[Dict[str, Any]]:
    """Run model inference pipeline on extracted RGB frames."""
    if not frames or model is None or config is None:
        return []

    num_frames = config.dataset.num_frames

    extractor = MediaPipeLandmarkExtractor(
        enable_face=config.model.landmark_encoder.include_face,
    )
    cropper = HandCropper(
        output_size=tuple(config.model.hand_encoder.input_size),
        padding_ratio=0.2,
    )

    all_pose = []
    all_left_crops = []
    all_right_crops = []

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

    def pad_list(lst: list, target_len: int, default_fn):
        while len(lst) < target_len:
            lst.append(default_fn())
        return lst[:target_len]

    lm_dim = config.model.landmark_encoder.input_dim
    all_pose = pad_list(all_pose, num_frames, lambda: np.zeros(lm_dim, dtype=np.float32))

    h, w = config.model.hand_encoder.input_size
    all_left_crops = pad_list(all_left_crops, num_frames, lambda: np.zeros((h, w, 3), dtype=np.uint8))
    all_right_crops = pad_list(all_right_crops, num_frames, lambda: np.zeros((h, w, 3), dtype=np.uint8))

    pose_tensor = torch.from_numpy(
        np.stack(all_pose, axis=0)
    ).float().unsqueeze(0).to(device)

    left_imgs = np.stack(all_left_crops, axis=0).astype(np.float32) / 255.0
    left_imgs = left_imgs.transpose(0, 3, 1, 2)
    left_tensor = torch.from_numpy(left_imgs).float().unsqueeze(0).to(device)

    right_imgs = np.stack(all_right_crops, axis=0).astype(np.float32) / 255.0
    right_imgs = right_imgs.transpose(0, 3, 1, 2)
    right_tensor = torch.from_numpy(right_imgs).float().unsqueeze(0).to(device)

    mask = torch.ones(1, num_frames, dtype=torch.bool, device=device)
    if len(frames) < num_frames:
        mask[0, len(frames):] = False

    batch = {
        "pose_landmarks": pose_tensor,
        "left_hand_images": left_tensor,
        "right_hand_images": right_tensor,
        "mask": mask,
    }

    with torch.no_grad():
        logits = model(batch)
        probs = torch.softmax(logits, dim=-1)

    top_k = min(top_k_val, config.model.num_classes)
    top_probs, top_indices = probs.topk(top_k, dim=-1)

    top_probs = top_probs.squeeze(0).cpu().numpy()
    top_indices = top_indices.squeeze(0).cpu().numpy()

    results = []
    for idx, prob in zip(top_indices, top_probs):
        idx_int = int(idx)
        label_name = idx_to_class.get(idx_int, f"class_{idx_int}")
        results.append({
            "label": label_name,
            "confidence": float(prob),
            "classIndex": idx_int,
        })

    return results


@app.get("/health")
def health_check():
    """Health check status endpoint."""
    return {
        "status": "online",
        "model_loaded": model is not None,
        "device": str(device),
        "experiment": config.name if config else None,
    }


@app.get("/classes")
def get_classes():
    """Return all supported class names."""
    return {
        "total_classes": len(idx_to_class),
        "classes": idx_to_class,
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict_video(
    file: UploadFile = File(...),
    top_k: int = 5,
):
    """Predict sign language label from an uploaded video file."""
    if model is None:
        raise HTTPException(status_code=500, detail="Model yüklü değil (Server initialization failed).")

    start_time = time.time()
    contents = await file.read()
    file_size = len(contents)
    
    print(f"\n[SERVER API] 📥 İstek alındı! Dosya: {file.filename}, Boyut: {file_size} bytes ({file_size / 1024:.1f} KB)")

    if not contents or file_size == 0:
        print("[SERVER API] ❌ HATA: Boş video dosyası alındı (0 bytes)!")
        raise HTTPException(
            status_code=400, 
            detail="Boş video dosyası alındı (0 bytes). Kameradan video kaydı alınamamış olabilir."
        )

    frames = sample_frames_from_bytes(contents, config.dataset.num_frames)
    if not frames:
        print(f"[SERVER API] ❌ HATA: Video kareleri okunamadı ({file_size} bytes)! OpenCV MP4 formatını çözemedi.")
        raise HTTPException(
            status_code=400, 
            detail=f"Video kareleri okunamadı ({file_size / 1024:.1f} KB). OpenCV MP4 formatını çözemedi."
        )

    print(f"[SERVER API] 🎞️ {len(frames)} adet kare başarıyla okundu. Görsel boyutu: {frames[0].shape}. Model çıkarımı başlatılıyor...")
    results = run_model_inference(frames, top_k_val=top_k)
    elapsed_ms = int((time.time() - start_time) * 1000)

    if results:
        top_res = results[0]
        print(f"[SERVER API] ✅ Tahmin Tamamlandı! En yüksek: {top_res['label']} (%{top_res['confidence']*100:.1f}) - Süre: {elapsed_ms}ms\n")

    return PredictionResponse(
        predictions=[PredictionItem(**item) for item in results],
        inferenceTimeMs=elapsed_ms,
    )


def parse_args():
    parser = argparse.ArgumentParser(description="ASL Inference Server")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/experiment/asl_citizen_baseline.yaml",
        help="Path to experiment config YAML.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="outputs/asl_citizen_baseline/checkpoints/best_model.pt",
        help="Path to trained model checkpoint.",
    )
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host address.")
    parser.add_argument("--port", type=int, default=8000, help="Port number.")
    return parser.parse_args()


def init_server(config_path: str, checkpoint_path: str):
    global model, config, device, idx_to_class

    setup_logging(level="INFO")
    logger = get_logger("server")

    logger.info(f"Loading config from: {config_path}")
    config = load_config(config_path)

    device = get_device(config.device)
    logger.info(f"Using device: {device}")

    idx_to_class = load_class_names(config.dataset.class_list_file)
    logger.info(f"Loaded {len(idx_to_class)} class names.")

    logger.info("Initializing HybridASLModel architecture...")
    model = HybridASLModel(config.model)
    model.to(device)

    chk_path = Path(checkpoint_path)
    if not chk_path.exists():
        logger.warning(f"Checkpoint file not found at: {checkpoint_path}. Searching default paths...")
        fallback = Path("outputs/asl_citizen_baseline/checkpoints/best_model.pt")
        if fallback.exists():
            chk_path = fallback

    logger.info(f"Loading model weights from: {chk_path}")
    checkpoint = torch.load(chk_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    logger.info("Model loaded and ready for predictions!")


def main():
    args = parse_args()
    init_server(args.config, args.checkpoint)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
