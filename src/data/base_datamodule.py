"""
Abstract DataModule for managing train/val/test DataLoaders.

Encapsulates dataset instantiation and DataLoader creation,
providing a single interface for the Trainer to consume.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable, Optional

from torch.utils.data import DataLoader

from src.core.config import DatasetConfig, TrainingConfig
from src.utils.paths import get_effective_num_workers


class BaseDataModule(ABC):
    """Abstract DataModule that provides DataLoaders for each split.

    Subclasses must implement setup() to create dataset instances
    for each split, and optionally override DataLoader construction.

    Attributes:
        dataset_config: Dataset configuration.
        training_config: Training configuration (for batch_size, num_workers).
    """

    def __init__(
        self,
        dataset_config: DatasetConfig,
        training_config: TrainingConfig,
        collate_fn: Optional[Callable[[list[Any]], Any]] = None,
    ) -> None:
        """Initialize the DataModule.

        Args:
            dataset_config: Configuration for the dataset.
            training_config: Configuration for training parameters.
            collate_fn: Optional custom collate function for DataLoaders.
        """
        self.dataset_config = dataset_config
        self.training_config = training_config
        self.collate_fn = collate_fn

        self._train_dataset = None
        self._val_dataset = None
        self._test_dataset = None

    @abstractmethod
    def setup(self, stage: Optional[str] = None) -> None:
        """Create dataset instances for each split.

        Args:
            stage: Optional stage ('fit', 'test', 'predict').
                   If None, sets up all splits.
        """
        ...

    def _loader_kwargs(self) -> dict[str, Any]:
        """Shared DataLoader kwargs."""
        return {
            "batch_size": self.training_config.batch_size,
            "num_workers": get_effective_num_workers(self.training_config.num_workers),
            "pin_memory": self.training_config.pin_memory,
            "collate_fn": self.collate_fn,
        }

    def train_dataloader(self) -> DataLoader:
        """Create and return the training DataLoader.

        Returns:
            DataLoader for the training split.

        Raises:
            RuntimeError: If setup() has not been called.
        """
        if self._train_dataset is None:
            raise RuntimeError("Call setup() before requesting DataLoaders.")
        return DataLoader(
            self._train_dataset,
            shuffle=True,
            drop_last=True,
            **self._loader_kwargs(),
        )

    def val_dataloader(self) -> DataLoader:
        """Create and return the validation DataLoader.

        Returns:
            DataLoader for the validation split.

        Raises:
            RuntimeError: If setup() has not been called.
        """
        if self._val_dataset is None:
            raise RuntimeError("Call setup() before requesting DataLoaders.")
        return DataLoader(
            self._val_dataset,
            shuffle=False,
            drop_last=False,
            **self._loader_kwargs(),
        )

    def test_dataloader(self) -> DataLoader:
        """Create and return the test DataLoader.

        Returns:
            DataLoader for the test split.

        Raises:
            RuntimeError: If setup() has not been called.
        """
        if self._test_dataset is None:
            raise RuntimeError("Call setup() before requesting DataLoaders.")
        return DataLoader(
            self._test_dataset,
            shuffle=False,
            drop_last=False,
            **self._loader_kwargs(),
        )
