"""
Single-video inference script.

Runs the full pipeline on a single video file and outputs the
predicted sign label.

Usage:
    python scripts/inference.py --config configs/experiment/autsl_10_mobilenet.yaml --checkpoint checkpoints/best_model.pt --video path/to/video.mp4
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.core.config import load_config
from src.models.hybrid_model import HybridASLModel
from src.preprocessing.inference_pipeline import prepare_inference_batch, sample_video_frames
from src.utils.device import get_device
from src.utils.logging import get_logger, setup_logging


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Run inference on a video")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--video", type=str, required=True)
    parser.add_argument("--top_k", type=int, default=5)
    return parser.parse_args()


def load_class_names(class_list_file: str) -> dict[int, str]:
    """Load class index to name mapping.

    Args:
        class_list_file: Path to class list txt file.

    Returns:
        Dictionary mapping class index to gloss name.
    """
    path = Path(class_list_file)
    if not path.exists():
        return {}

    idx_to_class = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(maxsplit=1)
            if len(parts) == 2:
                idx_to_class[int(parts[0])] = parts[1]
    return idx_to_class


def sample_frames_from_video(
    video_path: str, num_frames: int
) -> list[np.ndarray]:
    """Load and uniformly sample frames from a video.

    Args:
        video_path: Path to video file.
        num_frames: Number of frames to sample.

    Returns:
        List of (H, W, 3) uint8 RGB frames.
    """
    return sample_video_frames(video_path, num_frames)


def main() -> None:
    """Main inference entry point."""
    args = parse_args()
    config = load_config(args.config)

    logger = setup_logging(level="INFO")
    log = get_logger("inference")
    device = get_device(config.device)

    log.info(f"Running inference on: {args.video}")
    log.info(f"Device: {device}")

    # --- 1. Load video frames ---
    frames = sample_frames_from_video(args.video, config.dataset.num_frames)
    if not frames:
        log.error(f"Could not load video: {args.video}")
        return

    log.info(f"Loaded {len(frames)} frames from video")
    batch = prepare_inference_batch(frames, config, device)

    # --- 4. Load model and checkpoint ---
    log.info("Loading model...")
    model = HybridASLModel(config.model)
    model.to(device)

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    log.info(
        f"Loaded checkpoint (epoch={checkpoint.get('epoch', '?')}, "
        f"best_metric={checkpoint.get('best_metric', '?'):.4f})"
    )

    # --- 5. Forward pass ---
    with torch.no_grad():
        logits = model(batch)  # (1, num_classes)
        probs = torch.softmax(logits, dim=-1)

    # --- 6. Get top-K predictions ---
    top_k = min(args.top_k, config.model.num_classes)
    top_probs, top_indices = probs.topk(top_k, dim=-1)

    top_probs = top_probs.squeeze(0).cpu().numpy()
    top_indices = top_indices.squeeze(0).cpu().numpy()

    # Load class names
    idx_to_class = load_class_names(config.dataset.class_list_file)

    # --- 7. Print results ---
    log.info("=" * 50)
    log.info("Prediction Results")
    log.info("=" * 50)

    for rank, (idx, prob) in enumerate(zip(top_indices, top_probs), 1):
        class_name = idx_to_class.get(int(idx), f"class_{idx}")
        log.info(f"  #{rank}: {class_name} (index={idx}, prob={prob:.4f})")

    log.info("=" * 50)


if __name__ == "__main__":
    main()
