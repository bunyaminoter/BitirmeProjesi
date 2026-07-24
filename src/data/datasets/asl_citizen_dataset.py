"""
ASL Citizen dataset implementation.

Parses the ASL Citizen metadata format (CSV/JSON) where each entry maps
a video to its sign label, signer ID, and split assignment.
"""

from __future__ import annotations

import csv
import json
import logging
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.core.config import DatasetConfig
from src.core.registry import DATASET_REGISTRY
from src.core.types import SampleMetadata
from src.data.base_dataset import BaseSignLanguageDataset

logger = logging.getLogger(__name__)


@DATASET_REGISTRY.register("asl_citizen")
class ASLCitizenDataset(BaseSignLanguageDataset):
    """ASL Citizen dataset for isolated sign language recognition."""

    def __init__(
        self,
        config: DatasetConfig,
        split: str,
        transform: Optional[Any] = None,
        top_n_classes: Optional[int] = None,
    ) -> None:
        self.annotations: Dict[str, Any] = {}
        self.top_n_classes = top_n_classes
        super().__init__(config=config, split=split, transform=transform)

    def load_annotations(self) -> List[SampleMetadata]:
        annotation_path = Path(self.config.annotation_file)
        if annotation_path.is_dir():
            annotation_path = annotation_path / f"{self.split}.csv"

        if not annotation_path.exists():
            # Kaggle environment fallback
            kaggle_path = Path("/kaggle/input/datasets/abd0kamel/asl-citizen/ASL_Citizen/splits") / f"{self.split}.csv"
            if kaggle_path.exists():
                annotation_path = kaggle_path
            else:
                logger.warning(f"Annotation file not found: {annotation_path}. Returning empty list (expected during initialization without data).")
                return []

        class_list_path = Path(self.config.class_list_file)
        if class_list_path.exists():
            self._load_class_list(class_list_path)

        suffix = annotation_path.suffix.lower()
        if suffix == ".csv":
            raw_entries = self._parse_csv(annotation_path)
        elif suffix == ".json":
            raw_entries = self._parse_json(annotation_path)
        else:
            raise ValueError(f"Unsupported format: {suffix}")

        if self.top_n_classes and not self.class_to_idx:
            self._build_top_n_class_list(raw_entries)

        samples: List[SampleMetadata] = []
        for entry in raw_entries:
            if entry["split"] != self.split:
                continue

            gloss = entry["gloss"]
            if self.class_to_idx and gloss not in self.class_to_idx:
                continue

            if gloss in self.class_to_idx:
                class_idx = self.class_to_idx[gloss]
            else:
                class_idx = len(self.class_to_idx)
                self.class_to_idx[gloss] = class_idx
                self.idx_to_class[class_idx] = gloss

            if class_idx >= self.config.num_classes:
                continue

            samples.append(
                SampleMetadata(
                    video_id=entry["video_id"],
                    label=class_idx,
                    gloss=gloss,
                    split=entry["split"],
                    start_frame=1,
                    end_frame=-1,
                    signer_id=entry.get("signer_id"),
                )
            )

        logger.info(
            f"ASLCitizenDataset ({self.split}): "
            f"{len(samples)} samples, "
            f"{len(set(s.label for s in samples))} classes"
        )
        return samples

    def _parse_csv(self, path: Path) -> List[Dict[str, str]]:
        entries = []
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            headers = {h.strip().lower() for h in reader.fieldnames or []}

            vid_col = self._find_column(headers, {"video_id", "filename", "video", "id", "video file"}, reader.fieldnames)
            gloss_col = self._find_column(headers, {"gloss", "label", "sign"}, reader.fieldnames)
            signer_col = self._find_column(headers, {"signer_id", "participant_id", "signer", "participant id"}, reader.fieldnames)
            split_col = self._find_column(headers, {"split", "subset", "set"}, reader.fieldnames)

            if vid_col is None or gloss_col is None:
                raise ValueError("CSV must have video_id and gloss columns.")

            for row in reader:
                video_id = row[vid_col].strip()
                if video_id.endswith((".mp4", ".avi", ".mov", ".webm")):
                    video_id = Path(video_id).stem

                entries.append({
                    "video_id": video_id,
                    "gloss": row[gloss_col].strip(),
                    "signer_id": row[signer_col].strip() if signer_col else None,
                    "split": row[split_col].strip() if split_col else self.split,
                })
        return entries

    @staticmethod
    def _find_column(headers: set[str], candidates: set[str], originals: list[str] | None) -> Optional[str]:
        if not originals: return None
        for field in originals:
            if field.strip().lower() in candidates: return field
        return None

    def _parse_json(self, path: Path) -> List[Dict[str, str]]:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        entries = []
        if isinstance(data, dict):
            for vid, info in data.items():
                entries.append({
                    "video_id": vid,
                    "gloss": info.get("gloss", info.get("label", "")),
                    "signer_id": info.get("signer_id"),
                    "split": info.get("split", "train"),
                })
        elif isinstance(data, list):
            for item in data:
                vid = item.get("video_id", item.get("filename", ""))
                if vid.endswith(".mp4"): vid = Path(vid).stem
                entries.append({
                    "video_id": vid,
                    "gloss": item.get("gloss", item.get("label", "")),
                    "signer_id": item.get("signer_id"),
                    "split": item.get("split", "train"),
                })
        return entries

    def _build_top_n_class_list(self, entries: List[Dict[str, str]]) -> None:
        gloss_counts = Counter(e["gloss"] for e in entries)
        n = self.top_n_classes or len(gloss_counts)
        top_glosses = [gloss for gloss, _ in gloss_counts.most_common(n)]
        self.class_to_idx = {gloss: idx for idx, gloss in enumerate(sorted(top_glosses))}
        self.idx_to_class = {idx: gloss for gloss, idx in self.class_to_idx.items()}

    def _load_class_list(self, path: Path) -> None:
        from src.utils.class_list import load_class_list
        self.class_to_idx, self.idx_to_class = load_class_list(path)

    def get_video_path(self, index: int) -> Path:
        video_id = self.samples[index].video_id
        video_dir = Path(self.config.video_dir)
        
        # Kaggle environment fallback for videos
        if not video_dir.exists():
            kaggle_dir = Path("/kaggle/input/datasets/abd0kamel/asl-citizen/ASL_Citizen/videos")
            if kaggle_dir.exists():
                video_dir = kaggle_dir
                
        return video_dir / f"{video_id}.mp4"

    def get_label(self, index: int) -> int:
        return self.samples[index].label

    def get_metadata(self, index: int) -> SampleMetadata:
        return self.samples[index]
