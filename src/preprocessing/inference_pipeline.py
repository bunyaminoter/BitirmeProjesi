"""AUTSL inference features matching the offline cache and validation loader."""
from __future__ import annotations

import cv2
import numpy as np
import torch

from src.preprocessing.gap_repair import repair_hand_gaps
from src.preprocessing.hand_cropper import HandCropper
from src.preprocessing.mediapipe_extractor import MediaPipeLandmarkExtractor
from src.preprocessing.wrist_fallback_crop import fallback_crops, fallback_geometry, center_repair_policy


def sample_video_frames(video_path: str, num_frames: int) -> list[np.ndarray]:
    cap = cv2.VideoCapture(video_path)
    try:
        if not cap.isOpened():
            return []
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            return []
        frames = []
        for index in np.linspace(0, total - 1, min(total, num_frames), dtype=int):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
            ok, frame = cap.read()
            if not ok:
                raise ValueError(f"Video frame {index} could not be decoded: {video_path}")
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        return frames
    finally:
        cap.release()


def extract_features(frames, config, extractor=None) -> dict:
    """Match preprocess_tid carry-forward, then apply the selected derivation."""
    if not frames:
        raise ValueError("No video frames")
    if config.model.landmark_encoder.input_dim != 225:
        raise ValueError("AUTSL inference requires 33 pose + 21 + 21 hand joints (225 values)")
    method = config.dataset.cache_derivation
    if method not in (None, "pose_wrist_fallback_crop", "short_hand_gap_linear"):
        raise ValueError(f"Unsupported preprocessing derivation: {method}")
    versions = {None: (None,), "pose_wrist_fallback_crop": (None, 1, 2, 3),
                "short_hand_gap_linear": (None, 2)}
    version = config.dataset.cache_derivation_version
    if version not in versions[method]:
        raise ValueError("Unsupported preprocessing derivation version")
    if method == "pose_wrist_fallback_crop" and config.dataset.fallback_box_scale != 1.0 and version not in (2, 3):
        raise ValueError("Scaled fallback crops require derivation version 2 or 3")
    policy = center_repair_policy(config.dataset.fallback_center_repair)
    if (policy is not None) != (method == "pose_wrist_fallback_crop" and version == 3):
        raise ValueError("Center repair requires explicit derivation version 3 and policy")
    owns_extractor = extractor is None
    extractor = extractor or MediaPipeLandmarkExtractor(enable_face=False, model_complexity=1)
    cropper = HandCropper(tuple(config.model.hand_encoder.input_size), padding_ratio=.2)
    values = {key: [] for key in ("pose_landmarks", "left_hand_landmarks", "right_hand_landmarks",
                                  "pose_detected", "left_hand_detected", "right_hand_detected",
                                  "left_hand_crops", "right_hand_crops")}
    previous = {"left": None, "right": None}
    try:
        extractor.ensure_required_models()
        for frame in frames:
            detected = extractor.extract(frame)
            values["pose_landmarks"].append(detected.pose if detected.pose is not None else np.zeros((33, 3), np.float32))
            values["pose_detected"].append(detected.pose is not None)
            crops = cropper.crop_hands(frame, detected)
            for side, crop in zip(("left", "right"), crops):
                hand = getattr(detected, f"{side}_hand")
                values[f"{side}_hand_detected"].append(hand is not None)
                values[f"{side}_hand_landmarks"].append(hand if hand is not None else np.zeros((21, 3), np.float32))
                if hand is not None:
                    previous[side] = crop.image
                values[f"{side}_hand_crops"].append(previous[side] if previous[side] is not None else crop.image)
    finally:
        if owns_extractor:
            extractor.close()
    data = {key: np.asarray(value) for key, value in values.items()}
    data["num_frames"] = np.array(len(frames))
    for side, own, other in (("left", 15, 16), ("right", 16, 15)):
        if method == "pose_wrist_fallback_crop":
            data[f"{side}_hand_crops"], data[f"{side}_crop_source"] = fallback_crops(
                frames, data[f"{side}_hand_crops"], data[f"{side}_hand_landmarks"],
                data[f"{side}_hand_detected"], data["pose_landmarks"], data["pose_detected"], side,
                box_scale=config.dataset.fallback_box_scale, center_repair=policy)
            if policy is not None:
                height, width = frames[0].shape[:2]
                _, centers, boxes = fallback_geometry(data[f"{side}_hand_landmarks"],
                    data[f"{side}_hand_detected"], data["pose_landmarks"], data["pose_detected"],
                    side, width, height, box_scale=config.dataset.fallback_box_scale, center_repair=policy)
                data[f"{side}_crop_center"] = centers
                data[f"{side}_crop_box"] = boxes
                data[f"{side}_crop_center_imputed"] = data[f"{side}_crop_source"] == 3
        elif method == "short_hand_gap_linear":
            data[f"{side}_hand_landmarks"], data[f"{side}_hand_imputed"] = repair_hand_gaps(
                data[f"{side}_hand_landmarks"], data[f"{side}_hand_detected"],
                data["pose_landmarks"], data["pose_detected"], wrist_index=own,
                other_wrist_index=other, max_gap=config.dataset.max_repair_gap or 2)
    return data


def features_to_batch(data, config, device="cpu") -> dict[str, torch.Tensor]:
    """Same joint order, RGB scaling, ImageNet normalization and padding as val."""
    target = config.dataset.num_frames
    count = min(int(data["num_frames"]), target)
    combined = np.concatenate([data[key] for key in (
        "pose_landmarks", "left_hand_landmarks", "right_hand_landmarks")], axis=1)
    pose = np.zeros((target, 225), dtype=np.float32)
    pose[:count] = combined[:count].reshape(count, 225)
    batch = {"pose_landmarks": torch.from_numpy(pose).unsqueeze(0),
             "mask": (torch.arange(target) < count).unsqueeze(0)}
    height, width = config.model.hand_encoder.input_size
    mean = torch.tensor([.485, .456, .406]).view(1, 3, 1, 1)
    std = torch.tensor([.229, .224, .225]).view(1, 3, 1, 1)
    for side in ("left", "right"):
        images = np.zeros((target, height, width, 3), dtype=np.float32)
        for i in range(count):
            crop = data[f"{side}_hand_crops"][i]
            if crop.shape[:2] != (height, width):
                crop = cv2.resize(crop, (width, height), interpolation=cv2.INTER_LINEAR)
            images[i] = crop.astype(np.float32) / 255.
        tensor = torch.from_numpy(images.transpose(0, 3, 1, 2))
        batch[f"{side}_hand_images"] = ((tensor - mean) / std).unsqueeze(0)
    return {key: value.to(device) for key, value in batch.items()}


def prepare_inference_batch(frames, config, device="cpu"):
    return features_to_batch(extract_features(frames, config), config, device)
