"""
WLASL (Word-Level American Sign Language) dataset implementation.

Parses the WLASL nslt_*.json annotation format where each entry maps
a video_id to {subset, action: [class_idx, start_frame, end_frame]}.

Reference: https://github.com/dxli94/WLASL
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.core.config import DatasetConfig
from src.core.registry import DATASET_REGISTRY
from src.core.types import SampleMetadata
from src.data.base_dataset import BaseSignLanguageDataset
from src.utils.class_list import load_class_list


@DATASET_REGISTRY.register("wlasl")
class WLASLDataset(BaseSignLanguageDataset):
    """WLASL dataset for isolated sign language recognition.

    Expects annotation files in the nslt_*.json format:
    {
        "video_id": {
            "subset": "train" | "val" | "test",
            "action": [class_index, start_frame, end_frame]
        },
        ...
    }

    And a class list file with one gloss per line.

    Attributes:
        annotations: Raw parsed JSON annotations.
    """

    def __init__(
        self,
        config: DatasetConfig,
        split: str,
        transform: Optional[Any] = None,
    ) -> None:
        """Initialize the WLASL dataset.

        Args:
            config: Dataset configuration.
            split: Which split to load ('train', 'val', 'test').
            transform: Optional transform/augmentation pipeline.
        """
        self.annotations: Dict[str, Any] = {}
        super().__init__(config=config, split=split, transform=transform)

    def load_annotations(self) -> List[SampleMetadata]:
        """Parse the WLASL nslt JSON annotation file.

        Returns:
            List of SampleMetadata for samples in the current split.

        Raises:
            FileNotFoundError: If annotation or class list file is missing.
        """
        annotation_path = Path(self.config.annotation_file)
        if not annotation_path.exists():
            raise FileNotFoundError(
                f"WLASL annotation file not found: {annotation_path}"
            )

        # Load class list (supports "0 book" and gloss-only formats)
        class_list_path = Path(self.config.class_list_file)
        if class_list_path.exists():
            self.class_to_idx, self.idx_to_class = load_class_list(class_list_path)

        # Load annotations
        with open(annotation_path, "r", encoding="utf-8") as f:
            self.annotations = json.load(f)

        # Filter by split and num_classes
        samples: List[SampleMetadata] = []
        for video_id, info in self.annotations.items():
            subset = info.get("subset", "")
            action = info.get("action", [0, 1, -1])
            class_idx = action[0]
            start_frame = action[1]
            end_frame = action[2]

            # Filter by split
            if subset != self.split:
                continue

            # Filter by num_classes
            if class_idx >= self.config.num_classes:
                continue

            # Get gloss name if available
            gloss = self.idx_to_class.get(class_idx, str(class_idx))

            samples.append(
                SampleMetadata(
                    video_id=video_id,
                    label=class_idx,
                    gloss=gloss,
                    split=subset,
                    start_frame=start_frame,
                    end_frame=end_frame,
                )
            )

        return samples

    def get_video_path(self, index: int) -> Path:
        """Return the video file path for the given index.

        Args:
            index: Sample index.

        Returns:
            Path to the .mp4 video file.
        """
        video_id = self.samples[index].video_id
        return Path(self.config.video_dir) / f"{video_id}.mp4"

    def get_label(self, index: int) -> int:
        """Return the class label for the given index.

        Args:
            index: Sample index.

        Returns:
            Integer class label.
        """
        return self.samples[index].label

    def get_metadata(self, index: int) -> SampleMetadata:
        """Return the full metadata for the given index.

        Args:
            index: Sample index.

        Returns:
            SampleMetadata dataclass.
        """
        return self.samples[index]
