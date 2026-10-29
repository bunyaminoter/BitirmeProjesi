"""Colab GPU smoke test for the paired 100-class trainable RGB+pose candidate.

Usage:
  python scripts/preflight_100_rgb_pose.py \
    --config configs/experiment/autsl_100_rgb_mobilenet_v3_large_seed7.yaml \
    --cache_dir /content/BitirmeProjesi/cache/autsl_100_features_centerrepair_strict

This does one optimizer-sized micro-batch forward/backward only; it does not
train a model or write a checkpoint.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from statistics import median

import torch
import torch.nn.functional as F

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))

from src.core.config import load_config
from src.data.tid_datamodule import TIDDataModule, collate_fn
from src.models.hybrid_model import HybridASLModel


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--cache_dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    cfg = load_config(str(args.config))
    if not cfg.model.use_rgb or cfg.dataset.num_classes != 100:
        raise ValueError("Expected 100-class RGB+pose config")
    cache = args.cache_dir
    manifest = json.loads((cache / "manifest.json").read_text(encoding="utf-8"))
    if (manifest.get("num_classes"), manifest.get("num_frames")) != (100, 16):
        raise ValueError("Cache class/frame manifest mismatch")
    if manifest.get("class_ids") != cfg.dataset.class_ids:
        raise ValueError("Cache class order differs from config")
    derivation = manifest.get("derivation", {})
    if derivation.get("method") != "pose_wrist_fallback_crop" or derivation.get("version") != 3:
        raise ValueError("Expected strict v1 crop derivation")
    counts = {s: sum(1 for _ in (cache / s).glob("*.npz")) for s in ("train", "val")}
    if counts != {"train": 12405, "val": 1963}:
        raise ValueError(f"Incomplete 100-class cache: {counts}")

    dm = TIDDataModule(
        dataset_config=cfg.dataset,
        training_config=cfg.training,
        cache_dir=str(cache),
        augmentation_config=cfg.augmentation,
        hand_image_size=tuple(cfg.model.hand_encoder.input_size),
        landmark_dim=cfg.model.landmark_encoder.input_dim,
        load_rgb=True,
    )
    dm.setup(stage="fit")
    train_signers = {s.signer_id for s in dm._train_dataset.base_dataset.samples}
    val_signers = {s.signer_id for s in dm._val_dataset.base_dataset.samples}
    if not train_signers.isdisjoint(val_signers):
        raise ValueError("Train/val signer overlap")

    data_start = time.perf_counter()
    batch = collate_fn([dm._train_dataset[i] for i in range(cfg.training.batch_size)])
    data_load_seconds = time.perf_counter() - data_start
    for key in ("pose_landmarks", "left_hand_images", "right_hand_images", "labels", "mask"):
        if key not in batch:
            raise ValueError(f"Missing model input: {key}")
    model = HybridASLModel(cfg.model)
    if cfg.training.gradient_checkpointing:
        model.hand_encoder_left.enable_gradient_checkpointing()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("This memory/speed preflight requires a Colab GPU")
    model.to(device).train()
    tensors = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=cfg.training.learning_rate)
    torch.cuda.reset_peak_memory_stats()
    scaler = torch.amp.GradScaler(enabled=cfg.training.mixed_precision)
    timings = []
    for _ in range(4):  # First step warms kernels and optimizer; 3 steps estimate steady compute.
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.amp.autocast(device_type="cuda", enabled=cfg.training.mixed_precision):
            logits = model(tensors)
            loss = F.cross_entropy(logits, tensors["labels"])
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        torch.cuda.synchronize()
        timings.append(time.perf_counter() - start)
    result = {
        "status": "passed", "cache_counts": counts, "batch_size": cfg.training.batch_size,
        "effective_batch": cfg.training.batch_size * cfg.training.gradient_accumulation_steps,
        "logit_shape": list(logits.shape), "loss": round(float(loss.item()), 5),
        "data_load_seconds_for_batch_npz": round(data_load_seconds, 3),
        "first_step_seconds": round(timings[0], 3),
        "steady_step_seconds_median": round(median(timings[1:]), 3),
        "peak_gpu_gib": round(torch.cuda.max_memory_allocated() / 2**30, 3),
        "peak_reserved_gpu_gib": round(torch.cuda.max_memory_reserved() / 2**30, 3),
        "total_gpu_gib": round(torch.cuda.get_device_properties(device).total_memory / 2**30, 3),
        "gpu_name": torch.cuda.get_device_name(),
    }
    serialized = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized + "\n", encoding="utf-8")
    print(serialized)


if __name__ == "__main__":
    main()
