"""
Visualize Grad-CAM for a given video.

Usage:
    python scripts/visualize_gradcam.py --config configs/experiment/colab_100.yaml --checkpoint checkpoints/best_model.pt --video path/to/video.mp4
"""

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.core.config import load_config
from src.models.hybrid_model import HybridASLModel
from src.evaluation.gradcam import GradCAM, overlay_cam_on_image
from scripts.inference import sample_frames_from_video, load_class_names
from src.preprocessing.mediapipe_extractor import MediaPipeLandmarkExtractor
from src.preprocessing.hand_cropper import HandCropper
from src.utils.device import get_device

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--video", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="outputs/gradcam")
    return parser.parse_args()

def main():
    args = parse_args()
    config = load_config(args.config)
    device = get_device(config.device)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 1. Load model
    print("Loading model...")
    model = HybridASLModel(config.model)
    model.to(device)
    
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    # 2. Extract frames & features
    print(f"Processing video: {args.video}")
    frames = sample_frames_from_video(args.video, config.dataset.num_frames)
    if not frames:
        print("Failed to load frames.")
        return
        
    extractor = MediaPipeLandmarkExtractor(enable_face=config.model.landmark_encoder.include_face)
    cropper = HandCropper(output_size=tuple(config.model.hand_encoder.input_size))
    
    all_pose, all_left_crops, all_right_crops = [], [], []
    for frame in frames:
        res = extractor.extract(frame)
        if res.pose is not None:
            all_pose.append(res.pose.reshape(-1))
        else:
            all_pose.append(np.zeros(33*3, dtype=np.float32))
            
        lc, rc = cropper.crop_hands(frame, res)
        all_left_crops.append(lc.image)
        all_right_crops.append(rc.image)
    extractor.close()
    
    # Padding if needed
    while len(all_pose) < config.dataset.num_frames:
        all_pose.append(np.zeros(config.model.landmark_encoder.input_dim, dtype=np.float32))
        h, w = config.model.hand_encoder.input_size
        all_left_crops.append(np.zeros((h, w, 3), dtype=np.uint8))
        all_right_crops.append(np.zeros((h, w, 3), dtype=np.uint8))
        
    all_pose = all_pose[:config.dataset.num_frames]
    all_left_crops = all_left_crops[:config.dataset.num_frames]
    all_right_crops = all_right_crops[:config.dataset.num_frames]
    
    # To tensors
    pose_tensor = torch.from_numpy(np.stack(all_pose)).float().unsqueeze(0).to(device)
    
    left_imgs = np.stack(all_left_crops).astype(np.float32) / 255.0
    left_tensor = torch.from_numpy(left_imgs.transpose(0, 3, 1, 2)).float().unsqueeze(0).to(device)
    
    right_imgs = np.stack(all_right_crops).astype(np.float32) / 255.0
    right_tensor = torch.from_numpy(right_imgs.transpose(0, 3, 1, 2)).float().unsqueeze(0).to(device)
    
    mask = torch.ones(1, config.dataset.num_frames, dtype=torch.bool, device=device)
    mask[0, len(frames):] = False
    
    batch = {
        "pose_landmarks": pose_tensor,
        "left_hand_images": left_tensor,
        "right_hand_images": right_tensor,
        "mask": mask
    }
    
    # 3. Predict
    with torch.no_grad():
        logits = model(batch)
        pred_class = logits.argmax(dim=-1).item()
        
    idx_to_class = load_class_names(config.dataset.class_list_file)
    pred_name = idx_to_class.get(pred_class, f"Class {pred_class}")
    print(f"Predicted class: {pred_name}")
    
    # 4. Grad-CAM on right hand (for example)
    print("Computing Grad-CAM...")
    # Find a suitable target layer in the backbone
    # For EfficientNet, it might be model.hand_encoder_right.backbone[-3] or similar.
    # Let's try to get the last conv layer from the backbone
    if hasattr(model.hand_encoder_right.backbone, "features"):
        # MobileNet/EfficientNet/ConvNeXt
        target_layer = model.hand_encoder_right.backbone.features[-1]
    else:
        # ResNet
        target_layer = model.hand_encoder_right.backbone.layer4[-1]
        
    cam_extractor = GradCAM(model, target_layer)
    cams = cam_extractor(batch, pred_class) # (1, T, 224, 224)
    
    # 5. Visualize and save
    print(f"Saving visualizations to {args.output_dir}...")
    T = cams.shape[1]
    for t in range(min(T, len(frames))):
        img = left_imgs[t] # Or right_imgs, depending on which hand we hooked
        # We hooked right hand in this example, let's overlay on right hand
        img = right_imgs[t] 
        cam_mask = cams[0, t]
        
        overlay = overlay_cam_on_image(img, cam_mask)
        
        out_path = os.path.join(args.output_dir, f"frame_{t:02d}.png")
        # OpenCV expects BGR
        overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
        cv2.imwrite(out_path, overlay_bgr)
        
    print("Done!")

if __name__ == "__main__":
    main()
