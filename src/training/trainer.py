"""
Main Trainer class for the training loop.

Handles the full training lifecycle: epoch loop, loss computation,
optimization, validation, checkpointing, and callback dispatch.
Supports mixed precision, gradient clipping, and resume from checkpoint.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.core.config import ExperimentConfig
from src.training.callbacks.base_callback import BaseCallback

logger = logging.getLogger(__name__)


class Trainer:
    """Main training orchestrator.

    Manages the complete training lifecycle with support for:
        - Mixed precision training (torch.amp)
        - Gradient clipping
        - Callback system (checkpoint, early stopping, logging)
        - Resume from checkpoint
        - Multi-device support (CPU, CUDA, MPS)

    Attributes:
        config: Full experiment configuration.
        model: The model to train.
        optimizer: Optimizer instance.
        scheduler: Learning rate scheduler instance (optional).
        criterion: Loss function.
        callbacks: List of training callbacks.
        device: Target device for training.
        scaler: GradScaler for mixed precision.
        current_epoch: Current epoch number.
        global_step: Global step counter.
    """

    def __init__(
        self,
        config: ExperimentConfig,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        scheduler: Optional[Any] = None,
        callbacks: Optional[List[BaseCallback]] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        """Initialize the Trainer.

        Args:
            config: Full experiment configuration.
            model: Model to train.
            optimizer: Optimizer instance.
            criterion: Loss function.
            scheduler: Optional LR scheduler.
            callbacks: Optional list of callbacks.
            device: Target device (auto-detected if None).
        """
        self.config = config
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        self.callbacks = callbacks or []
        self.device = device or self._detect_device()

        # Mixed precision
        self.scaler = torch.amp.GradScaler(
            enabled=config.training.mixed_precision
        )

        # State
        self.current_epoch = 0
        self.global_step = 0
        self.best_metric = 0.0

        # Move model to device
        self.model.to(self.device)

    def _detect_device(self) -> torch.device:
        """Auto-detect the best available device.

        Returns:
            torch.device for CUDA, MPS, or CPU.
        """
        if self.config.device != "auto":
            return torch.device(self.config.device)
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
    ) -> Dict[str, Any]:
        """Run the full training loop.

        Args:
            train_loader: Training DataLoader.
            val_loader: Optional validation DataLoader.

        Returns:
            Dictionary of final training metrics.
        """
        # Notify callbacks
        self._dispatch("on_train_start", trainer=self)

        train_metrics: Dict[str, float] = {}
        val_metrics: Dict[str, float] = {}

        for epoch in range(self.current_epoch, self.config.training.epochs):
            self.current_epoch = epoch

            # --- Training epoch ---
            self._dispatch("on_epoch_start", trainer=self, epoch=epoch)
            train_metrics = self._train_epoch(train_loader)

            logger.info(
                f"Epoch {epoch + 1}/{self.config.training.epochs} — "
                f"Train Loss: {train_metrics['loss']:.4f}, "
                f"Train Acc: {train_metrics['accuracy']:.4f}"
            )

            self._dispatch(
                "on_epoch_end",
                trainer=self,
                epoch=epoch,
                metrics=train_metrics,
            )

            # --- Validation ---
            val_metrics = {}
            if val_loader and (epoch + 1) % self.config.training.val_every_n_epochs == 0:
                val_metrics = self._validate_epoch(val_loader)

                logger.info(
                    f"  Val Loss: {val_metrics['loss']:.4f}, "
                    f"  Val Acc: {val_metrics['accuracy']:.4f}"
                )

                self._dispatch(
                    "on_validation_end",
                    trainer=self,
                    epoch=epoch,
                    metrics=val_metrics,
                )

                # Track best metric
                val_acc = val_metrics.get("accuracy", 0.0)
                if val_acc > self.best_metric:
                    self.best_metric = val_acc
                    logger.info(
                        f"  New best val accuracy: {self.best_metric:.4f}"
                    )

            # --- LR Scheduler step (epoch-level) ---
            if self.scheduler is not None:
                if hasattr(self.scheduler, "step"):
                    # ReduceLROnPlateau needs metric
                    if isinstance(
                        self.scheduler,
                        torch.optim.lr_scheduler.ReduceLROnPlateau,
                    ):
                        metric = val_metrics.get(
                            "accuracy", train_metrics.get("loss", 0.0)
                        )
                        self.scheduler.step(metric)
                    else:
                        self.scheduler.step()

            # --- Check for early stopping ---
            should_stop = self._check_early_stopping()
            if should_stop:
                logger.info(
                    f"Early stopping triggered at epoch {epoch + 1}"
                )
                break

        self._dispatch("on_train_end", trainer=self)
        return {"train": train_metrics, "val": val_metrics}

    def _train_epoch(self, loader: DataLoader) -> Dict[str, float]:
        """Run one training epoch.

        Args:
            loader: Training DataLoader.

        Returns:
            Dictionary of training metrics for this epoch.
        """
        self.model.train()

        running_loss = 0.0
        correct = 0
        total = 0

        # Determine autocast device type
        amp_device = "cuda" if self.device.type == "cuda" else "cpu"
        use_amp = self.config.training.mixed_precision and self.device.type == "cuda"

        pbar = tqdm(loader, desc=f"Epoch {self.current_epoch + 1}/{self.config.training.epochs} [Train]", leave=False)
        for batch_idx, batch in enumerate(pbar):
            # Move batch tensors to device
            batch = self._move_to_device(batch)

            labels = batch["labels"]
            batch_size = labels.size(0)

            self._dispatch("on_batch_start", trainer=self, batch_idx=batch_idx)

            # --- Forward pass with autocast ---
            self.optimizer.zero_grad()

            with torch.amp.autocast(device_type=amp_device, enabled=use_amp):
                logits = self.model(batch)  # (B, num_classes)
                loss = self.criterion(logits, labels)

            # --- Backward pass with scaler ---
            self.scaler.scale(loss).backward()

            # --- Gradient clipping ---
            if self.config.training.gradient_clip > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.training.gradient_clip,
                )

            # --- Optimizer step ---
            self.scaler.step(self.optimizer)
            self.scaler.update()

            # --- Track metrics ---
            running_loss += loss.item() * batch_size
            preds = logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += batch_size
            self.global_step += 1

            self._dispatch(
                "on_batch_end",
                trainer=self,
                batch_idx=batch_idx,
                loss=loss.item(),
            )

            # Periodic logging
            if (
                self.config.training.log_every_n_steps > 0
                and (batch_idx + 1) % self.config.training.log_every_n_steps == 0
            ):
                logger.info(
                    f"  Step {batch_idx + 1}/{len(loader)} — "
                    f"Loss: {loss.item():.4f}"
                )
            
            # Update progress bar
            pbar.set_postfix({"loss": f"{loss.item():.4f}", "acc": f"{(correct/max(total,1)):.4f}"})

        epoch_loss = running_loss / max(total, 1)
        epoch_acc = correct / max(total, 1)

        return {"loss": epoch_loss, "accuracy": epoch_acc}

    def _validate_epoch(self, loader: DataLoader) -> Dict[str, float]:
        """Run one validation epoch.

        Args:
            loader: Validation DataLoader.

        Returns:
            Dictionary of validation metrics.
        """
        self.model.eval()

        running_loss = 0.0
        correct = 0
        total = 0

        amp_device = "cuda" if self.device.type == "cuda" else "cpu"
        use_amp = self.config.training.mixed_precision and self.device.type == "cuda"

        pbar = tqdm(loader, desc=f"Epoch {self.current_epoch + 1}/{self.config.training.epochs} [Val]", leave=False)
        with torch.no_grad():
            for batch in pbar:
                batch = self._move_to_device(batch)
                labels = batch["labels"]
                batch_size = labels.size(0)

                with torch.amp.autocast(device_type=amp_device, enabled=use_amp):
                    logits = self.model(batch)
                    loss = self.criterion(logits, labels)

                running_loss += loss.item() * batch_size
                preds = logits.argmax(dim=-1)
                correct += (preds == labels).sum().item()
                total += batch_size
                
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        epoch_loss = running_loss / max(total, 1)
        epoch_acc = correct / max(total, 1)

        return {"loss": epoch_loss, "accuracy": epoch_acc}

    def _move_to_device(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Move batch tensors to the target device.

        Args:
            batch: Dictionary of batch data.

        Returns:
            Batch with tensors moved to device.
        """
        moved = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                moved[key] = value.to(self.device)
            else:
                moved[key] = value
        return moved

    def _check_early_stopping(self) -> bool:
        """Check if training should stop early.

        Returns:
            True if early stopping criteria are met.
        """
        for callback in self.callbacks:
            if hasattr(callback, "should_stop") and callback.should_stop:
                return True
        return False

    def save_checkpoint(self, path: str | Path) -> None:
        """Save a training checkpoint.

        Args:
            path: Path to save the checkpoint file.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "epoch": self.current_epoch,
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scaler_state_dict": self.scaler.state_dict(),
            "best_metric": self.best_metric,
            "config": self.config,
        }
        if self.scheduler is not None:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()

        torch.save(checkpoint, path)

    def load_checkpoint(self, path: str | Path) -> None:
        """Load a training checkpoint and resume training.

        Args:
            path: Path to the checkpoint file.

        Raises:
            FileNotFoundError: If the checkpoint file does not exist.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scaler.load_state_dict(checkpoint["scaler_state_dict"])
        self.current_epoch = checkpoint["epoch"] + 1
        self.global_step = checkpoint["global_step"]
        self.best_metric = checkpoint.get("best_metric", 0.0)

        if self.scheduler is not None and "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    def _dispatch(self, event: str, **kwargs: Any) -> None:
        """Dispatch a callback event to all registered callbacks.

        Args:
            event: Name of the callback method to invoke.
            **kwargs: Arguments to pass to the callback method.
        """
        for callback in self.callbacks:
            method = getattr(callback, event, None)
            if method is not None:
                method(**kwargs)
