"""
Logging callback for training progress.

Logs training metrics to console using Rich formatting.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, Optional

from src.training.callbacks.base_callback import BaseCallback


class LoggingCallback(BaseCallback):
    """Log training progress to console and history.csv.

    Uses Rich library for formatted console output with tables
    and progress indicators, and saves metrics to a CSV for plotting.

    Attributes:
        log_every_n_steps: Log every N training steps.
        history: List of dictionaries storing epoch metrics.
    """

    def __init__(self, log_every_n_steps: int = 10) -> None:
        """Initialize the logging callback.

        Args:
            log_every_n_steps: Logging frequency in steps.
        """
        super().__init__()
        self.log_every_n_steps = log_every_n_steps
        self.history = []
        self.csv_path = None

    def on_train_start(self, trainer: Any) -> None:
        """Log training start information."""
        checkpoint_dir = Path(trainer.config.training.checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = checkpoint_dir / "history.csv"

        is_resuming = getattr(trainer, "current_epoch", 0) > 0 and self.csv_path.exists()
        if is_resuming:
            # Önceki epochların geçmişini yükle
            try:
                self.history = []
                with open(self.csv_path, mode='r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        self.history.append({
                            'epoch': int(row.get('epoch', len(self.history) + 1)),
                            'train_loss': float(row.get('train_loss', 0.0)),
                            'train_acc': float(row.get('train_acc', 0.0)),
                            'val_loss': float(row.get('val_loss', 0.0)),
                            'val_acc': float(row.get('val_acc', 0.0)),
                        })
            except Exception:
                pass
        else:
            # Sıfırdan başlama: CSV'yi başlıklarla başlat
            self.history = []
            with open(self.csv_path, mode='w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['epoch', 'train_loss', 'train_acc', 'val_loss', 'val_acc'])

    def on_epoch_end(
        self,
        trainer: Any,
        epoch: int,
        metrics: Optional[Dict[str, float]] = None,
    ) -> None:
        """Log epoch training metrics."""
        if metrics:
            # Add a new entry for this epoch, padding if necessary (e.g. when resuming)
            while len(self.history) <= epoch:
                self.history.append({'epoch': len(self.history) + 1})
            self.history[epoch].update({
                'train_loss': metrics.get('loss', 0.0),
                'train_acc': metrics.get('accuracy', 0.0)
            })

    def on_validation_end(
        self,
        trainer: Any,
        epoch: int,
        metrics: Optional[Dict[str, float]] = None,
    ) -> None:
        """Log validation metrics."""
        if metrics:
            while len(self.history) <= epoch:
                self.history.append({'epoch': len(self.history) + 1})
            self.history[epoch].update({
                'val_loss': metrics.get('loss', 0.0),
                'val_acc': metrics.get('accuracy', 0.0)
            })
            
            # Write the current epoch to CSV since validation is the end of the epoch step
            entry = self.history[epoch]
            with open(self.csv_path, mode='a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    entry.get('epoch', epoch + 1),
                    entry.get('train_loss', 0.0),
                    entry.get('train_acc', 0.0),
                    entry.get('val_loss', 0.0),
                    entry.get('val_acc', 0.0)
                ])

    def on_train_end(self, trainer: Any) -> None:
        """Plot and save training curves to checkpoint directory."""
        if not self.history or self.csv_path is None:
            return

        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            epochs = [entry.get('epoch', i + 1) for i, entry in enumerate(self.history)]
            train_loss = [entry.get('train_loss', 0.0) for entry in self.history]
            val_loss = [entry.get('val_loss', 0.0) for entry in self.history]
            train_acc = [entry.get('train_acc', 0.0) for entry in self.history]
            val_acc = [entry.get('val_acc', 0.0) for entry in self.history]

            fig, axes = plt.subplots(1, 2, figsize=(14, 5))

            # Loss curve
            axes[0].plot(epochs, train_loss, label='Train Loss', color='#2563eb', linewidth=2)
            axes[0].plot(epochs, val_loss, label='Val Loss', color='#dc2626', linewidth=2)
            axes[0].set_title('Loss Eğrisi (Train vs Val)', fontsize=12, fontweight='bold')
            axes[0].set_xlabel('Epoch')
            axes[0].set_ylabel('Loss')
            axes[0].legend()
            axes[0].grid(True, linestyle='--', alpha=0.6)

            # Accuracy curve
            axes[1].plot(epochs, train_acc, label='Train Acc', color='#2563eb', linewidth=2)
            axes[1].plot(epochs, val_acc, label='Val Acc', color='#dc2626', linewidth=2)
            axes[1].set_title('Accuracy Eğrisi (Train vs Val)', fontsize=12, fontweight='bold')
            axes[1].set_xlabel('Epoch')
            axes[1].set_ylabel('Accuracy')
            axes[1].legend()
            axes[1].grid(True, linestyle='--', alpha=0.6)

            plt.tight_layout()

            save_path = self.csv_path.parent / "training_curves.png"
            fig.savefig(save_path, dpi=150)
            plt.close(fig)
            print(f"📊 Eğitim grafiği başarıyla kaydedildi: {save_path}")
        except Exception as e:
            print(f"⚠️ Grafik kaydedilirken hata oluştu: {e}")

