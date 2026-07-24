"""
ASL Citizen Preprocessor (Kaggle Compatible)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.core.config import load_config
from src.preprocessing.feature_cache import FeatureCache
from src.preprocessing.hand_cropper import HandCropper
from src.preprocessing.mediapipe_extractor import MediaPipeLandmarkExtractor
from src.utils.logging import setup_logging, get_logger
from src.data.datasets.asl_citizen_dataset import ASLCitizenDataset


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--cache_dir", type=str, default="cache/asl_citizen_features/")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--top_n_classes", type=int, default=None,
                        help="Override config num_classes for preprocessing")
    return parser.parse_args()


def sample_frames(video_path: str, num_frames: int) -> list[np.ndarray]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened(): return []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        return []

    indices = np.linspace(0, total_frames - 1, min(total_frames, num_frames), dtype=int).tolist()
    if total_frames < num_frames:
        indices = list(range(total_frames))

    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        else:
            if frames: frames.append(np.zeros_like(frames[-1]))
            else: frames.append(np.zeros((480, 640, 3), dtype=np.uint8))
    cap.release()
    return frames


def main():
    args = parse_args()
    config = load_config(args.config)
    log = setup_logging(level="INFO")
    log = get_logger("preprocess")

    # CLI override for num_classes
    num_classes = args.top_n_classes or config.dataset.num_classes
    config.dataset.num_classes = num_classes
    config.model.num_classes = num_classes

    # --- Ortam tespiti ve yol düzeltme ---
    import os
    class_list_filename = Path(config.dataset.class_list_file).name  # e.g. class_list_100.txt
    if os.path.exists("/content/drive"):
        # Google Colab ortamı
        drive_project = "/content/drive/MyDrive/BitirmeProjesi"
        asl_data = f"{drive_project}/asl_citizen_data"
        if os.path.isdir(f"{asl_data}/splits"):
            config.dataset.annotation_file = f"{asl_data}/splits"
        asl_vids = f"{drive_project}/asl_citizen_videos/ASL_Citizen/videos"
        if os.path.isdir(asl_vids):
            config.dataset.video_dir = asl_vids
        else:
            config.dataset.video_dir = f"{drive_project}/videos"
        config.dataset.class_list_file = f"{drive_project}/cache/{class_list_filename}"
        log.info("Ortam: Google Colab (Drive)")
    elif os.path.exists("/kaggle/input"):
        # Kaggle ortamı
        config.dataset.annotation_file = "/kaggle/input/datasets/abd0kamel/asl-citizen/ASL_Citizen/splits"
        config.dataset.video_dir = "/kaggle/input/datasets/abd0kamel/asl-citizen/ASL_Citizen/videos"
        config.dataset.class_list_file = f"/kaggle/working/cache/{class_list_filename}"
        log.info("Ortam: Kaggle")
    else:
        log.info("Ortam: Local")

    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    log.info(f"Using cache dir: {cache_dir}")
    log.info(f"Video dir: {config.dataset.video_dir}")
    log.info(f"Annotation file: {config.dataset.annotation_file}")
    log.info(f"Num classes: {config.dataset.num_classes}")

    cache = FeatureCache(cache_dir=str(cache_dir), cache_format="npz", store_hand_crops=True)
    extractor = MediaPipeLandmarkExtractor(enable_face=False, model_complexity=1)
    cropper = HandCropper(output_size=tuple(config.model.hand_encoder.input_size), padding_ratio=0.2)

    # Use dataset class to get valid files (num_classes from config)
    ds = ASLCitizenDataset(config.dataset, split="train", top_n_classes=num_classes)
    
    # Process train, val, test subsets
    from tqdm import tqdm
    for split in ["train", "val", "test"]:
        ds.split = split
        samples = ds.load_annotations()
        log.info(f"{split} split için {len(samples)} örnek işlenecek.")
        
        skipped = 0
        processed = 0
        for i, sample in enumerate(tqdm(samples, desc=f"{split}", leave=True)):
            if cache.exists(sample.video_id):
                skipped += 1
                continue
                
            vid_path = ds.get_video_path(i)
            if not vid_path.exists():
                continue
                
            frames = sample_frames(str(vid_path), config.dataset.num_frames)
            if not frames: continue

            all_pose, all_left_hand, all_right_hand, all_left_crop, all_right_crop = [], [], [], [], []

            for frame in frames:
                res = extractor.extract(frame)
                all_pose.append(res.pose if res.pose is not None else np.zeros((33, 3), dtype=np.float32))
                all_left_hand.append(res.left_hand if res.left_hand is not None else np.zeros((21, 3), dtype=np.float32))
                all_right_hand.append(res.right_hand if res.right_hand is not None else np.zeros((21, 3), dtype=np.float32))
                
                lc, rc = cropper.crop_hands(frame, res)
                all_left_crop.append(lc.image)
                all_right_crop.append(rc.image)

            features = {
                "pose_landmarks": np.stack(all_pose, axis=0),
                "left_hand_landmarks": np.stack(all_left_hand, axis=0),
                "right_hand_landmarks": np.stack(all_right_hand, axis=0),
                "left_hand_crops": np.stack(all_left_crop, axis=0),
                "right_hand_crops": np.stack(all_right_crop, axis=0),
                "label": np.array(sample.label, dtype=np.int64),
                "num_frames": np.array(len(frames), dtype=np.int64),
            }
            cache.save(sample.video_id, features)
            processed += 1

        log.info(f"{split} split: {processed} işlendi, {skipped} cache'den atlandı.")

    extractor.close()
    log.info("Preprocessing tamamlandı!")

if __name__ == "__main__":
    main()
