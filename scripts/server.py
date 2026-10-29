"""
FastAPI Server for ASL Recognition Model Inference.

Provides REST API endpoints for the Flutter mobile app to run predictions
using the trained HybridASLModel.

Usage:
    python scripts/server.py --config configs/experiment/autsl_10_mobilenet.yaml --checkpoint outputs/autsl_10_mobilenet/checkpoints/best_model.pt --port 8000
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
from src.preprocessing.inference_pipeline import prepare_inference_batch, sample_video_frames
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
    """Load AUTSL class names and stop if their mapping is unavailable."""
    path = Path(class_list_file)
    if not path.is_absolute():
        path = project_root / path
    if not path.is_file():
        raise FileNotFoundError(f"AUTSL class list missing: {path}")

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
        raise ValueError(f"Empty AUTSL class list: {path}")

    return mapping


def sample_frames_from_bytes(video_bytes: bytes, num_frames: int) -> List[np.ndarray]:
    """Sample frames from uploaded video bytes using a temporary file."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(video_bytes)
        tmp_path = tmp.name

    try:
        return sample_video_frames(tmp_path, num_frames)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def run_model_inference(frames: List[np.ndarray], top_k_val: int = 5) -> List[Dict[str, Any]]:
    """Run model inference pipeline on extracted RGB frames."""
    if not frames or model is None or config is None:
        return []

    batch = prepare_inference_batch(frames, config, device)

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
    
    print(f"\n[SERVER API] [+] Istek alindi! Dosya: {file.filename}, Boyut: {file_size} bytes ({file_size / 1024:.1f} KB)")

    if not contents or file_size == 0:
        print("[SERVER API] [ERROR] Bos video dosyasi alindi (0 bytes)!")
        raise HTTPException(
            status_code=400, 
            detail="Boş video dosyası alındı (0 bytes). Kameradan video kaydı alınamamış olabilir."
        )

    frames = sample_frames_from_bytes(contents, config.dataset.num_frames)
    if not frames:
        print(f"[SERVER API] [ERROR] Video kareleri okunamadi ({file_size} bytes)! OpenCV MP4 formatini cozemedi.")
        raise HTTPException(
            status_code=400, 
            detail=f"Video kareleri okunamadı ({file_size / 1024:.1f} KB). OpenCV MP4 formatını çözemedi."
        )

    print(f"[SERVER API] [*] {len(frames)} adet kare basariyla okundu. Gorsel boyutu: {frames[0].shape}. Model cikarimi baslatiliyor...")
    results = run_model_inference(frames, top_k_val=top_k)
    elapsed_ms = int((time.time() - start_time) * 1000)

    if results:
        top_res = results[0]
        print(f"[SERVER API] [SUCCESS] Tahmin Tamamlandi! En yuksek: {top_res['label']} (%{top_res['confidence']*100:.1f}) - Sure: {elapsed_ms}ms\n")

    return PredictionResponse(
        predictions=[PredictionItem(**item) for item in results],
        inferenceTimeMs=elapsed_ms,
    )


@app.post("/predict_frames", response_model=PredictionResponse)
async def predict_from_frames(
    frames: list[UploadFile] = File(...),
    top_k: int = 5,
):
    """Predict sign language label from uploaded JPEG frame images.

    This endpoint is optimized for the continuous recognition mode where
    the mobile app buffers frames during motion and sends them directly
    as JPEG images, bypassing video encoding/decoding overhead.
    """
    if model is None:
        raise HTTPException(status_code=500, detail="Model yuklu degil (Server initialization failed).")

    start_time = time.time()

    print(f"\n[SERVER API] [+] Frame-based istek alindi! {len(frames)} frame")

    if not frames:
        raise HTTPException(status_code=400, detail="Bos frame listesi alindi.")

    # JPEG frame'leri RGB numpy array'lere donustur
    rgb_frames = []
    for i, frame_file in enumerate(frames):
        try:
            content = await frame_file.read()
            if not content:
                continue

            # JPEG bytes -> numpy array (OpenCV ile decode)
            nparr = np.frombuffer(content, np.uint8)
            img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img_bgr is not None:
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                rgb_frames.append(img_rgb)
        except Exception as e:
            print(f"[SERVER API] [WARN] Frame {i} decode hatasi: {e}")
            continue

    if not rgb_frames:
        raise HTTPException(status_code=400, detail="Hicbir frame decode edilemedi.")

    # Model'in beklediği frame sayısına uydur (linspace sampling)
    num_model_frames = config.dataset.num_frames
    if len(rgb_frames) > num_model_frames:
        indices = np.linspace(0, len(rgb_frames) - 1, num_model_frames, dtype=int).tolist()
        rgb_frames = [rgb_frames[i] for i in indices]

    print(f"[SERVER API] [*] {len(rgb_frames)} frame hazir. Model cikarimi baslatiliyor...")
    results = run_model_inference(rgb_frames, top_k_val=top_k)
    elapsed_ms = int((time.time() - start_time) * 1000)

    if results:
        top_res = results[0]
        print(f"[SERVER API] [SUCCESS] Tahmin Tamamlandi! En yuksek: {top_res['label']} (%{top_res['confidence']*100:.1f}) - Sure: {elapsed_ms}ms\n")

    return PredictionResponse(
        predictions=[PredictionItem(**item) for item in results],
        inferenceTimeMs=elapsed_ms,
    )

def parse_args():
    parser = argparse.ArgumentParser(description="ASL Inference Server")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/experiment/autsl_10_mobilenet.yaml",
        help="Path to experiment config YAML.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="outputs/autsl_10_mobilenet/checkpoints/best_model.pt",
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
    try:
        if not chk_path.is_file():
            raise FileNotFoundError(chk_path)
        checkpoint = torch.load(chk_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        logger.info("Model loaded and ready for predictions!")
    except Exception:
        model = None
        raise


def main():
    args = parse_args()
    init_server(args.config, args.checkpoint)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
