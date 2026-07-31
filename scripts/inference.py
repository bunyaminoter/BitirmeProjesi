"""
Single-video inference script.

Runs the full pipeline on a single video file and outputs the
predicted sign label.

Usage:
    python scripts/inference.py --config configs/experiment/asl_citizen_100.yaml --checkpoint checkpoints/best_model.pt --video path/to/video.mp4
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.core.config import load_config
from src.models.hybrid_model import HybridASLModel
from src.preprocessing.hand_cropper import HandCropper
from src.preprocessing.mediapipe_extractor import MediaPipeLandmarkExtractor
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
    num_frames = config.dataset.num_frames

    # --- 2. Extract landmarks ---
    log.info("Extracting landmarks...")
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

        # Pose landmarks
        if result.pose is not None:
            all_pose.append(result.pose.reshape(-1))  # (99,)
        else:
            all_pose.append(np.zeros(33 * 3, dtype=np.float32))

        # Hand crops
        left_crop, right_crop = cropper.crop_hands(frame, result)
        all_left_crops.append(left_crop.image)
        all_right_crops.append(right_crop.image)

    extractor.close()

    # --- 3. Build input tensors ---
    # Pad/truncate to num_frames
    def pad_list(lst, target_len, default_fn):
        while len(lst) < target_len:
            lst.append(default_fn())
        return lst[:target_len]

    lm_dim = config.model.landmark_encoder.input_dim
    all_pose = pad_list(
        all_pose, num_frames,
        lambda: np.zeros(lm_dim, dtype=np.float32)
    )

    h, w = config.model.hand_encoder.input_size
    all_left_crops = pad_list(
        all_left_crops, num_frames,
        lambda: np.zeros((h, w, 3), dtype=np.uint8)
    )
    all_right_crops = pad_list(
        all_right_crops, num_frames,
        lambda: np.zeros((h, w, 3), dtype=np.uint8)
    )

    # Stack and convert
    pose_tensor = torch.from_numpy(
        np.stack(all_pose, axis=0)  # (T, lm_dim)
    ).float().unsqueeze(0).to(device)  # (1, T, lm_dim)

    left_imgs = np.stack(all_left_crops, axis=0).astype(np.float32) / 255.0
    left_imgs = left_imgs.transpose(0, 3, 1, 2)  # (T, 3, H, W)
    left_tensor = torch.from_numpy(left_imgs).float().unsqueeze(0).to(device)

    right_imgs = np.stack(all_right_crops, axis=0).astype(np.float32) / 255.0
    right_imgs = right_imgs.transpose(0, 3, 1, 2)  # (T, 3, H, W)
    right_tensor = torch.from_numpy(right_imgs).float().unsqueeze(0).to(device)

    mask = torch.ones(1, num_frames, dtype=torch.bool, device=device)
    mask[0, len(frames):] = False

    batch = {
        "pose_landmarks": pose_tensor,
        "left_hand_images": left_tensor,
        "right_hand_images": right_tensor,
        "mask": mask,
    }

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
