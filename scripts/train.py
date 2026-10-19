"""
Training entry point.

Loads configuration, builds all components, and starts training.
Supports advanced training techniques: EMA, gradient checkpointing,
warmup scheduling, Mixup augmentation, and gradient accumulation.

Usage:
    python scripts/train.py --config configs/experiment/autsl_10_mobilenet.yaml
    python scripts/train.py --config configs/experiment/autsl_226_mobilenet_frozen.yaml --resume checkpoints/last.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.core.config import ExperimentConfig, load_config
from src.data.tid_datamodule import TIDDataModule
from src.evaluation.evaluator import Evaluator
from src.models.hybrid_model import HybridASLModel
from src.training.callbacks.checkpoint import CheckpointCallback
from src.training.callbacks.early_stopping import EarlyStoppingCallback
from src.training.callbacks.logging_callback import LoggingCallback
from src.training.ema import ModelEMA
from src.training.losses import CrossEntropyLoss
from src.training.optimizers import build_optimizer, build_scheduler, LinearWarmupScheduler
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


def resolve_environment(config: ExperimentConfig, args: argparse.Namespace) -> str:
    """Ortamı (Colab / Kaggle / Local) tespit edip config ve cache yollarını düzeltir.

    `config` ve `args.cache_dir` yerinde güncellenir. train.py ve train_cv.py ortak kullanır.

    Returns:
        Ortam adı ("Colab", "Kaggle" veya "Local").
    """
    # --- Ortam tespiti ve yol düzeltme ---
    # Colab ortamında Drive yollarını, Kaggle'da input yollarını otomatik çöz
    import os
    from pathlib import Path as _Path

    if config.dataset.name == "autsl":
        if args.cache_dir == "cache/features/":
            args.cache_dir = config.dataset.cache_dir
        if os.path.exists("/content/drive"):
            drive_project = "/content/drive/MyDrive/BitirmeProjesi"
            if args.cache_dir == config.dataset.cache_dir:
                cache_name = os.path.basename(os.path.normpath(config.dataset.cache_dir))
                local_cache = f"/content/BitirmeProjesi/cache/{cache_name}"
                drive_cache = f"{drive_project}/cache/{cache_name}"
                args.cache_dir = local_cache if os.path.isdir(local_cache) else drive_cache
            config.output_dir = f"{drive_project}/outputs/{config.name}/"
            config.training.checkpoint_dir = f"{config.output_dir}checkpoints/"
            return "Colab"
        return "Local"

    raise ValueError("Only dataset.name='autsl' is supported by the current training flow.")

    # Config'deki class_list dosya adını koru (class_list.txt / class_list_100.txt vb.)
    _class_list_filename = _Path(config.dataset.class_list_file).name

    if os.path.exists("/content/drive"):
        # Google Colab ortamı — tüm çıktılar doğrudan Drive'a kaydedilir
        drive_project = "/content/drive/MyDrive/BitirmeProjesi"
        
        if config.dataset.name == "tid":
            if os.path.exists("/content/tid_10_classes/labels.csv"):
                config.dataset.annotation_file = "/content/tid_10_classes/labels.csv"
                config.dataset.data_dir = "/content/tid_10_classes/videos"
            elif os.path.exists(f"{drive_project}/tid_10_classes/labels.csv"):
                config.dataset.annotation_file = f"{drive_project}/tid_10_classes/labels.csv"
                config.dataset.data_dir = f"{drive_project}/tid_10_classes/videos"
            
            cache_dir_override = args.cache_dir not in ["cache/features/", "cache/asl_citizen_features/", "cache/tid_10_features/"]
            if not cache_dir_override:
                if os.path.exists("/content/BitirmeProjesi/cache/tid_10_features"):
                    args.cache_dir = "/content/BitirmeProjesi/cache/tid_10_features/"
                elif os.path.exists(f"{drive_project}/cache/tid_10_features"):
                    args.cache_dir = f"{drive_project}/cache/tid_10_features/"
                else:
                    args.cache_dir = "/content/BitirmeProjesi/cache/tid_10_features/"
        else:
            asl_data = f"{drive_project}/asl_citizen_data"
            if os.path.isdir(f"{asl_data}/splits"):
                config.dataset.annotation_file = f"{asl_data}/splits"
            asl_vids = f"{drive_project}/asl_citizen_videos/ASL_Citizen/videos"
            if os.path.isdir(asl_vids):
                config.dataset.video_dir = asl_vids
            else:
                config.dataset.video_dir = f"{drive_project}/videos"
            config.dataset.class_list_file = f"{drive_project}/cache/{_class_list_filename}"

            cache_dir_override = args.cache_dir not in ["cache/features/", "cache/asl_citizen_features/"]
            if not cache_dir_override:
                if os.path.exists("/content/BitirmeProjesi/cache/asl_citizen_features"):
                    args.cache_dir = "/content/BitirmeProjesi/cache/asl_citizen_features/"
                elif os.path.exists("/content/cache/asl_citizen_features"):
                    args.cache_dir = "/content/cache/asl_citizen_features/"
                elif os.path.exists(f"{drive_project}/cache/asl_citizen_features"):
                    args.cache_dir = f"{drive_project}/cache/asl_citizen_features/"
                else:
                    args.cache_dir = f"{drive_project}/cache/features/"

        # Çıktılar doğrudan Drive'a (config adına göre dinamik)
        _run_name = config.name  # örn: "tid_10_mobilenet"
        config.output_dir = f"{drive_project}/outputs/{_run_name}/"
        config.training.checkpoint_dir = f"{drive_project}/outputs/{_run_name}/checkpoints/"
        log_msg = "Colab"

    elif os.path.exists("/kaggle/input"):
        # Kaggle ortamı
        config.dataset.annotation_file = "/kaggle/input/datasets/abd0kamel/asl-citizen/ASL_Citizen/splits"
        config.dataset.video_dir = "/kaggle/input/datasets/abd0kamel/asl-citizen/ASL_Citizen/videos"
        config.dataset.class_list_file = f"/kaggle/working/cache/{_class_list_filename}"
        args.cache_dir = "/kaggle/working/cache/asl_citizen_features/"
        config.output_dir = f"/kaggle/working/outputs/{config.name}/"
        config.training.checkpoint_dir = f"/kaggle/working/outputs/{config.name}/checkpoints/"
        log_msg = "Kaggle"
    else:
        log_msg = "Local"

    return log_msg


def main() -> None:
    """Main training entry point."""
    args = parse_args()

    # Load configuration
    config = load_config(args.config)
    log_msg = resolve_environment(config, args)

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
    log.info("Building TIDDataModule...")
    datamodule = TIDDataModule(
        dataset_config=config.dataset,
        training_config=config.training,
        cache_dir=args.cache_dir,
        augmentation_config=config.augmentation,
        hand_image_size=tuple(config.model.hand_encoder.input_size),
        landmark_dim=config.model.landmark_encoder.input_dim,
    )
    datamodule.setup(stage="fit")

    train_loader = datamodule.train_dataloader()
    val_loader = datamodule.val_dataloader()

    if hasattr(datamodule, "_train_dataset") and datamodule._train_dataset is not None:
        base_ds = datamodule._train_dataset.base_dataset
        log.info(f"Loaded {len(base_ds.class_to_idx)} classes: {sorted(base_ds.class_to_idx.keys())}")

    log.info(
        f"Data loaded — Train batches: {len(train_loader)}, "
        f"Val batches: {len(val_loader)}"
    )

    # --- 2. Build Model ---
    log.info("Building HybridASLModel...")
    model = HybridASLModel(config.model)

    param_counts = model.get_num_parameters()
    log.info(f"Model parameters: {param_counts}")

    # --- 2a. Enable Gradient Checkpointing (if configured) ---
    if config.training.gradient_checkpointing:
        # Enable on hand encoders (the heaviest components)
        if hasattr(model.hand_encoder_left, "enable_gradient_checkpointing"):
            model.hand_encoder_left.enable_gradient_checkpointing()
        if (
            model.hand_encoder_right is not model.hand_encoder_left
            and hasattr(model.hand_encoder_right, "enable_gradient_checkpointing")
        ):
            model.hand_encoder_right.enable_gradient_checkpointing()
        log.info("Gradient checkpointing enabled on hand encoders")

    # --- 3. Build Optimizer and Scheduler ---
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    total_params = sum(p.numel() for p in model.parameters())
    trainable_count = sum(p.numel() for p in trainable_params)
    log.info(f"Trainable parameters: {trainable_count:,} / {total_params:,} ({100*trainable_count/total_params:.1f}%)")

    optimizer = build_optimizer(
        name=config.training.optimizer,
        parameters=trainable_params,
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

    # --- 3a. Wrap with Warmup (if configured) ---
    warmup_epochs = config.training.warmup_epochs
    if warmup_epochs > 0:
        scheduler = LinearWarmupScheduler(
            optimizer=optimizer,
            warmup_epochs=warmup_epochs,
            wrapped_scheduler=scheduler,
        )
        log.info(f"Warmup: {warmup_epochs} epochs (linear)")

    # --- 4. Build Loss Function ---
    label_smoothing = getattr(config.training, "label_smoothing", 0.0)
    criterion = CrossEntropyLoss(label_smoothing=label_smoothing)
    log.info(f"Loss: CrossEntropyLoss (label_smoothing={label_smoothing})")

    # --- 4a. Build EMA (if configured) ---
    ema = None
    if config.training.use_ema:
        # EMA needs to be created after model is built but before training
        # We'll pass it to trainer which will create it after model.to(device)
        log.info(f"EMA: decay={config.training.ema_decay}")

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

    # --- 5a. Log active training techniques ---
    techniques = []
    if config.training.gradient_accumulation_steps > 1:
        effective_batch = config.training.batch_size * config.training.gradient_accumulation_steps
        techniques.append(f"GradAccum({config.training.gradient_accumulation_steps}x → eff.batch={effective_batch})")
    if config.training.gradient_checkpointing:
        techniques.append("GradCheckpoint")
    if config.training.use_ema:
        techniques.append(f"EMA(decay={config.training.ema_decay})")
    if warmup_epochs > 0:
        techniques.append(f"Warmup({warmup_epochs}ep)")
    if config.training.mixup_alpha > 0:
        techniques.append(f"Mixup(α={config.training.mixup_alpha})")

    if techniques:
        log.info(f"Active techniques: {', '.join(techniques)}")
    else:
        log.info("Active techniques: None (baseline)")

    # --- 6. Create Trainer ---
    # Move model to device first, then create EMA
    model.to(device)
    if config.training.use_ema:
        ema = ModelEMA(model, decay=config.training.ema_decay)

    trainer = Trainer(
        config=config,
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        scheduler=scheduler,
        callbacks=callbacks,
        device=device,
        ema=ema,
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

    # --- 10. Run Test Set Evaluation if available ---
    try:
        test_loader = datamodule.test_dataloader()
        if test_loader is not None and len(test_loader) > 0:
            log.info("=" * 60)
            log.info("Evaluating Best Model on TEST set...")
            log.info("=" * 60)
            best_path = checkpoint_dir / "best_model.pt"
            if best_path.exists():
                trainer.load_checkpoint(best_path)
            test_metrics = trainer.evaluate(test_loader)
            log.info(f"🏆 Final TEST Accuracy: {test_metrics.get('accuracy', 0.0):.4f}")
            log.info(f"🏆 Final TEST Loss:     {test_metrics.get('loss', 0.0):.4f}")
            log.info("=" * 60)
    except Exception as e:
        log.warning(f"Test evaluation skipped: {e}")



if __name__ == "__main__":
    main()
