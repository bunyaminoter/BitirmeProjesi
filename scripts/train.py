"""
Training entry point.

Loads configuration, builds all components, and starts training.

Usage:
    python scripts/train.py --config configs/experiment/asl_citizen_100.yaml
    python scripts/train.py --config configs/experiment/asl_citizen_100.yaml --resume checkpoints/last.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.core.config import ExperimentConfig, load_config
from src.data.asl_citizen_datamodule import ASLCitizenDataModule
from src.evaluation.evaluator import Evaluator
from src.models.hybrid_model import HybridASLModel
from src.training.callbacks.checkpoint import CheckpointCallback
from src.training.callbacks.early_stopping import EarlyStoppingCallback
from src.training.callbacks.logging_callback import LoggingCallback
from src.training.losses import CrossEntropyLoss
from src.training.optimizers import build_optimizer, build_scheduler
from src.training.trainer import Trainer
from src.utils.device import get_device, get_device_info
from src.utils.logging import get_logger, setup_logging
from src.utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Train the Hybrid ASL Recognition Model"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to experiment YAML configuration file.",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume training from.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="cache/features/",
        help="Path to cached preprocessed features.",
    )
    return parser.parse_args()


def main() -> None:
    """Main training entry point."""
    args = parse_args()

    # Load configuration
    config = load_config(args.config)

    # --- Ortam tespiti ve yol düzeltme ---
    # Colab ortamında Drive yollarını, Kaggle'da input yollarını otomatik çöz
    import os
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
        config.dataset.class_list_file = f"{drive_project}/cache/class_list.txt"
        config.output_dir = f"{drive_project}/outputs/asl_citizen_baseline/"
        config.training.checkpoint_dir = f"{drive_project}/outputs/asl_citizen_baseline/checkpoints/"
        log_msg = "Colab"
    elif os.path.exists("/kaggle/input"):
        # Kaggle ortamı
        config.dataset.annotation_file = "/kaggle/input/datasets/abd0kamel/asl-citizen/ASL_Citizen/splits"
        config.dataset.video_dir = "/kaggle/input/datasets/abd0kamel/asl-citizen/ASL_Citizen/videos"
        config.dataset.class_list_file = "/kaggle/working/cache/class_list.txt"
        config.output_dir = "/kaggle/working/outputs/asl_citizen_baseline/"
        config.training.checkpoint_dir = "/kaggle/working/outputs/asl_citizen_baseline/checkpoints/"
        log_msg = "Kaggle"
    else:
        log_msg = "Local"

    # Setup
    logger = setup_logging(level="INFO")
    log = get_logger("train")
    set_seed(config.seed)
    device = get_device(config.device)

    log.info(f"Experiment: {config.name}")
    log.info(f"Environment: {log_msg}")
    log.info(f"Device: {get_device_info(device)}")
    log.info(f"Seed: {config.seed}")

    # --- 1. Build DataModule ---
    log.info("Building DataModule...")
    datamodule = ASLCitizenDataModule(
        dataset_config=config.dataset,
        training_config=config.training,
        cache_dir=args.cache_dir,
    )
    datamodule.setup(stage="fit")

    train_loader = datamodule.train_dataloader()
    val_loader = datamodule.val_dataloader()

    log.info(
        f"Data loaded — Train batches: {len(train_loader)}, "
        f"Val batches: {len(val_loader)}"
    )

    # --- 2. Build Model ---
    log.info("Building HybridASLModel...")
    model = HybridASLModel(config.model)

    param_counts = model.get_num_parameters()
    log.info(f"Model parameters: {param_counts}")

    # --- 3. Build Optimizer and Scheduler ---
    optimizer = build_optimizer(
        name=config.training.optimizer,
        parameters=model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )
    log.info(f"Optimizer: {config.training.optimizer}")

    scheduler = build_scheduler(
        name=config.training.scheduler,
        optimizer=optimizer,
        epochs=config.training.epochs,
        **config.training.scheduler_params,
    )
    log.info(f"Scheduler: {config.training.scheduler}")

    # --- 4. Build Loss Function ---
    criterion = CrossEntropyLoss(label_smoothing=0.1)
    log.info("Loss: CrossEntropyLoss (label_smoothing=0.1)")

    # --- 5. Build Callbacks ---
    callbacks = []

    # Checkpoint callback
    checkpoint_dir = Path(config.training.checkpoint_dir)
    checkpoint_callback = CheckpointCallback(
        save_dir=str(checkpoint_dir),
        monitor="accuracy",
        mode="max",
    )
    callbacks.append(checkpoint_callback)

    # Early stopping callback
    if config.training.early_stopping_patience > 0:
        early_stop = EarlyStoppingCallback(
            monitor="accuracy",
            patience=config.training.early_stopping_patience,
            mode="max",
        )
        callbacks.append(early_stop)

    # Logging callback
    callbacks.append(LoggingCallback())

    log.info(f"Callbacks: {[type(c).__name__ for c in callbacks]}")

    # --- 6. Create Trainer ---
    trainer = Trainer(
        config=config,
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        scheduler=scheduler,
        callbacks=callbacks,
        device=device,
    )

    # --- 7. Resume from checkpoint if specified ---
    if args.resume:
        log.info(f"Resuming from checkpoint: {args.resume}")
        trainer.load_checkpoint(args.resume)

    # --- 8. Start Training ---
    log.info("=" * 60)
    log.info("Starting training...")
    log.info("=" * 60)

    results = trainer.fit(train_loader, val_loader)

    # --- 9. Save final checkpoint ---
    final_path = checkpoint_dir / "final_model.pt"
    trainer.save_checkpoint(final_path)
    log.info(f"Final checkpoint saved: {final_path}")

    log.info("=" * 60)
    log.info("Training complete!")
    log.info(f"Best validation accuracy: {trainer.best_metric:.4f}")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
