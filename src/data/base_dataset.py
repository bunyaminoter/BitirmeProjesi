"""
Abstract base class for sign language video datasets.

All dataset implementations must inherit from BaseSignLanguageDataset
and implement the abstract methods. This ensures that training code
never depends on a specific dataset format.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from torch.utils.data import Dataset

from src.core.config import DatasetConfig
from src.core.types import SampleMetadata


class BaseSignLanguageDataset(Dataset, ABC):
    """Abstract base class for all sign language video datasets.

    Subclasses must implement:
        - load_annotations(): Parse the dataset-specific annotation format.
        - get_video_path(index): Return the video file path for a sample.
        - get_label(index): Return the integer class label for a sample.
        - get_metadata(index): Return full metadata for a sample.

    Attributes:
        config: Dataset configuration dataclass.
        split: Current split ('train', 'val', 'test').
        samples: List of sample metadata after loading annotations.
        class_to_idx: Mapping from gloss string to integer label.
        idx_to_class: Mapping from integer label to gloss string.
    """

    def __init__(
        self,
        config: DatasetConfig,
        split: str,
        transform: Optional[Any] = None,
    ) -> None:
        """Initialize the dataset.

        Args:
            config: Dataset configuration.
            split: Which split to load ('train', 'val', 'test').
            transform: Optional transform/augmentation pipeline.
        """
        super().__init__()
        self.config = config
        self.split = split
        self.transform = transform
        self.samples: List[SampleMetadata] = []
        self.class_to_idx: Dict[str, int] = {}
        self.idx_to_class: Dict[int, str] = {}

        # Load annotations on initialization
        self.samples = self.load_annotations()

    @abstractmethod
    def load_annotations(self) -> List[SampleMetadata]:
        """Parse the dataset annotation file and return sample metadata.

        Returns:
            List of SampleMetadata for all samples in the current split.

        Raises:
            FileNotFoundError: If the annotation file does not exist.
        """
        ...

    @abstractmethod
    def get_video_path(self, index: int) -> Path:
        """Return the path to the video file for the given sample index.

        Args:
            index: Sample index.

        Returns:
            Path to the video file.
        """
        ...

    @abstractmethod
    def get_label(self, index: int) -> int:
        """Return the integer class label for the given sample index.

        Args:
            index: Sample index.

        Returns:
            Integer class label.
        """
        ...

    @abstractmethod
    def get_metadata(self, index: int) -> SampleMetadata:
        """Return full metadata for the given sample index.

        Args:
            index: Sample index.

        Returns:
            SampleMetadata dataclass.
        """
        ...

    def __len__(self) -> int:
        """Return the number of samples in this split."""
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        """Get a single sample by index.

        This default implementation returns raw metadata. Subclasses should
        override to include video loading, preprocessing, and augmentation.

        Args:
            index: Sample index.

        Returns:
            Dictionary with sample data.
        """
        metadata = self.get_metadata(index)
        return {
            "video_id": metadata.video_id,
            "label": metadata.label,
            "gloss": metadata.gloss,
            "split": metadata.split,
            "video_path": str(self.get_video_path(index)),
        }

    def get_class_names(self) -> List[str]:
        """Return an ordered list of class names.

        Returns:
            List of gloss strings ordered by class index.
        """
        return [
            self.idx_to_class[i]
            for i in range(len(self.idx_to_class))
        ]

    @property
    def num_classes(self) -> int:
        """Return the number of classes in this dataset."""
        return len(self.class_to_idx)
