"""
Evaluation entry point.

Loads a trained model checkpoint and evaluates on a selected AUTSL split.

Usage:
    python scripts/evaluate.py --config configs/experiment/autsl_10_mobilenet.yaml --checkpoint checkpoints/best_model.pt
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.core.config import load_config
from src.data.tid_datamodule import TIDDataModule
from src.evaluation.evaluator import Evaluator
from src.models.hybrid_model import HybridASLModel
from src.utils.device import get_device
from src.utils.logging import get_logger, setup_logging
from src.utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate the AUTSL model")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--split", type=str, default="val", choices=["val", "test"])
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--output", type=str, default=None,
                        help="Optional path to save evaluation results as JSON.")
    parser.add_argument("--predictions_output", type=str, default=None,
                        help="Optional JSON path for per-video labels and predictions.")
    return parser.parse_args()


def main() -> None:
    """Main evaluation entry point."""
    args = parse_args()
    config = load_config(args.config)

    logger = setup_logging(level="INFO")
    log = get_logger("evaluate")
    set_seed(config.seed)
    device = get_device(config.device)

    log.info(f"Evaluating: {config.name} on {args.split} split")
    log.info(f"Checkpoint: {args.checkpoint}")

    cache_dir = args.cache_dir or config.dataset.cache_dir
    if Path("/content/drive").exists():
        cache_name = Path(cache_dir).name
        local_cache = Path(f"/content/BitirmeProjesi/cache/{cache_name}")
        if local_cache.is_dir() and any(local_cache.iterdir()):
            cache_dir = str(local_cache)

    # --- 1. Build DataModule ---
    datamodule = TIDDataModule(
        dataset_config=config.dataset,
        training_config=config.training,
        cache_dir=cache_dir,
        hand_image_size=tuple(config.model.hand_encoder.input_size),
        landmark_dim=config.model.landmark_encoder.input_dim,
        load_rgb=config.model.use_rgb,
    )
    datamodule.setup(stage="test" if args.split == "test" else "validate")

    if args.split == "test":
        dataloader = datamodule.test_dataloader()
    else:
        dataloader = datamodule.val_dataloader()

    log.info(f"Data loaded — {args.split} batches: {len(dataloader)}")

    # --- 2. Build Model ---
    model = HybridASLModel(config.model)
    model.to(device)

    # --- 3. Load Checkpoint ---
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    log.info(
        f"Loaded checkpoint from epoch {checkpoint.get('epoch', '?')}, "
        f"best_metric={checkpoint.get('best_metric', '?')}"
    )

    # --- 4. Run Evaluation ---
    evaluator = Evaluator(
        model=model,
        num_classes=config.model.num_classes,
        device=device,
        mixed_precision=config.training.mixed_precision,
        collect_predictions=bool(args.predictions_output),
    )

    result = evaluator.evaluate(dataloader)

    # --- 5. Print Results ---
    log.info("=" * 60)
    log.info("Evaluation Results")
    log.info("=" * 60)

    metrics = result.to_dict()
    for name, value in metrics.items():
        log.info(f"  {name}: {value:.4f}")

    # --- 6. Optionally save results ---
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        log.info(f"Results saved to: {output_path}")

    if args.predictions_output:
        predictions_path = Path(args.predictions_output)
        predictions_path.parent.mkdir(parents=True, exist_ok=True)
        with checkpoint_path.open("rb") as checkpoint_file:
            digest = hashlib.file_digest(checkpoint_file, "sha256").hexdigest()
        payload = {
            "config": config.name,
            "split": args.split,
            "checkpoint_sha256": digest,
            "class_ids": config.dataset.class_ids,
            "predictions": evaluator.prediction_records,
        }
        predictions_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        log.info(f"Predictions saved to: {predictions_path}")


if __name__ == "__main__":
    main()
