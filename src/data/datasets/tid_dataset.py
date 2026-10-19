"""
TİD (Türk İşaret Dili / AUTSL) Dataset Implementation.
Compatible with BaseSignLanguageDataset and our MediaPipe + MobileNetV3 pipeline.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional
import pandas as pd

from src.core.config import DatasetConfig
from src.core.registry import DATASET_REGISTRY
from src.core.types import SampleMetadata
from src.data.base_dataset import BaseSignLanguageDataset

logger = logging.getLogger(__name__)


@DATASET_REGISTRY.register("tid")
@DATASET_REGISTRY.register("autsl")
class TIDDataset(BaseSignLanguageDataset):
    """Turkish Sign Language (TİD) dataset loader for isolated sign recognition."""

    def __init__(
        self,
        config: DatasetConfig,
        split: str,
        transform: Optional[Any] = None,
        top_n_classes: Optional[int] = None,
    ) -> None:
        self.top_n_classes = top_n_classes
        super().__init__(config=config, split=split, transform=transform)

    def load_annotations(self) -> List[SampleMetadata]:
        if self.config.name == "autsl":
            return self._load_official_autsl()

        annotation_path = Path(self.config.annotation_file)

        # Fallback paths for Colab and local setups
        candidate_paths = [
            annotation_path,
            Path("/content/tid_10_classes/labels.csv"),
            Path("/content/drive/MyDrive/BitirmeProjesi/tid_10_classes/labels.csv"),
            Path("data/tid_10_classes/labels.csv"),
            Path(r"C:\Users\Bünyamin\Desktop\AUTSL_Dataset\labels.csv"),
            Path(r"C:\Users\Bünyamin\Desktop\AUTSL_Dataset") / f"{self.split}_labels.csv",
        ]

        resolved_path = None
        for p in candidate_paths:
            if p.exists() and p.is_file():
                resolved_path = p
                break

        if not resolved_path:
            logger.warning(f"TİD annotation file not found: {annotation_path}. Returning empty list.")
            return []

        df = pd.read_csv(resolved_path)

        # Filter by split. "val+test" gibi '+' ile birleştirilmiş adlar desteklenir
        # (örn. 5-Fold CV'de eski val+test kümesini tek bir kilitli test kümesi yapmak için).
        wanted_splits = [s.strip() for s in str(self.split).split("+") if s.strip()]
        df_split = df[df["split"].isin(wanted_splits)].copy()

        # Build class index mappings
        unique_classes = sorted(df["tr_label"].unique())
        for idx, cls_name in enumerate(unique_classes):
            self.class_to_idx[cls_name] = idx
            self.idx_to_class[idx] = cls_name

        samples: List[SampleMetadata] = []
        for _, row in df_split.iterrows():
            vname = Path(str(row["video_filename"])).stem  # signerX_sampleY_color
            label_idx = int(row["class_id"])
            tr_label = str(row["tr_label"])
            signer = str(row.get("signer", "unknown"))

            samples.append(
                SampleMetadata(
                    video_id=vname,
                    label=label_idx,
                    gloss=tr_label,
                    split=self.split,
                    start_frame=1,
                    end_frame=-1,
                    signer_id=signer,
                )
            )

        logger.info(f"Loaded {len(samples)} TİD samples for split '{self.split}' across {len(self.class_to_idx)} classes.")
        return samples

    def _autsl_root(self) -> Path:
        """Resolve the official split directory without changing class or split semantics."""
        override = os.environ.get("AUTSL_DATA_DIR")
        required = (
            "train_labels.csv", "val_labels.csv", "test_labels.csv",
            "SignList_ClassId_TR_EN.csv",
        )
        if override:
            root = Path(override)
            missing = [name for name in required if not (root / name).is_file()]
            if missing:
                raise FileNotFoundError(f"AUTSL_DATA_DIR={root} is missing {missing}")
            return root
        candidates = [
            self.config.data_dir,
            self.config.annotation_file,
            str(Path.home() / "Desktop" / "AUTSL_Dataset"),
            "/content/drive/MyDrive/BitirmeProjesi/AUTSL_Dataset",
        ]
        for candidate in candidates:
            if candidate and all(
                (Path(candidate) / filename).is_file()
                for filename in required
            ):
                return Path(candidate)
        raise FileNotFoundError(
            "Official AUTSL root not found. Set AUTSL_DATA_DIR to the folder "
            "containing train_labels.csv, val_labels.csv and test_labels.csv."
        )

    def _load_official_autsl(self) -> List[SampleMetadata]:
        root = self._autsl_root()
        self._official_root = root
        if self.split not in {"train", "val", "test"}:
            raise ValueError(f"Official AUTSL needs a single split, got {self.split!r}")

        labels_file = root / f"{self.split}_labels.csv"
        if not labels_file.is_file():
            raise FileNotFoundError(labels_file)
        names_file = root / "SignList_ClassId_TR_EN.csv"
        if not names_file.is_file():
            raise FileNotFoundError(names_file)

        names = pd.read_csv(names_file, encoding="utf-8-sig")
        if not {"ClassId", "TR"}.issubset(names.columns):
            raise ValueError(f"Invalid AUTSL class list: {names_file}")
        names["ClassId"] = pd.to_numeric(names["ClassId"], errors="raise").astype(int)
        if names["ClassId"].duplicated().any():
            raise ValueError(f"Duplicate class IDs in {names_file}")
        source_names = dict(zip(names["ClassId"], names["TR"].astype(str)))
        configured_ids = self.config.class_ids
        source_ids = list(range(len(source_names))) if configured_ids is None else list(configured_ids)
        if len(source_ids) != len(set(source_ids)):
            raise ValueError("AUTSL class_ids must be unique")
        if any(class_id not in source_names for class_id in source_ids):
            raise ValueError("AUTSL class_ids contains an unknown source class ID")
        if self.config.num_classes is not None and len(source_ids) != int(self.config.num_classes):
            raise ValueError("AUTSL num_classes must equal the number of configured class_ids")

        source_to_model = {source_id: index for index, source_id in enumerate(source_ids)}
        self.idx_to_class = {index: source_names[source_id] for source_id, index in source_to_model.items()}
        self.class_to_idx = {name: index for index, name in self.idx_to_class.items()}
        if len(self.class_to_idx) != len(self.idx_to_class):
            raise ValueError(f"Duplicate class names in {names_file}")

        labels = pd.read_csv(
            labels_file, header=None, names=["video_id", "label"],
            dtype={"video_id": str}, encoding="utf-8-sig",
        )
        if labels.isna().any().any() or labels["video_id"].duplicated().any():
            raise ValueError(f"Missing or duplicate AUTSL labels in {labels_file}")
        labels["label"] = pd.to_numeric(labels["label"], errors="raise").astype(int)
        if not set(labels["label"]).issubset(source_names):
            raise ValueError("AUTSL labels contain an unknown source class ID")
        labels = labels[labels["label"].isin(source_to_model)].copy()

        samples: List[SampleMetadata] = []
        for video_id, source_label in labels.itertuples(index=False, name=None):
            if not video_id.startswith("signer") or "_sample" not in video_id:
                raise ValueError(f"Invalid AUTSL video ID: {video_id}")
            samples.append(SampleMetadata(
                video_id=f"{video_id}_color",
                label=source_to_model[int(source_label)],
                gloss=source_names[int(source_label)],
                split=self.split,
                start_frame=1,
                end_frame=-1,
                signer_id=video_id.split("_", 1)[0],
            ))
        logger.info("Loaded %d official AUTSL %s samples", len(samples), self.split)
        return samples

    def get_video_path(self, index: int) -> Path:
        video_id = self.samples[index].video_id
        fname = video_id if video_id.endswith(".mp4") else f"{video_id}.mp4"
        split_name = str(self.samples[index].split)

        if self.config.name == "autsl":
            return self._official_root / split_name / fname

        # Primary data_dir if configured
        if hasattr(self.config, "data_dir"):
            candidate = Path(self.config.data_dir) / fname
            if candidate.exists():
                return candidate

        # Check standard and desktop fallback locations
        autsl_base = Path(r"C:\Users\Bünyamin\Desktop\AUTSL_Dataset")
        candidate_dirs = [
            Path("/content/tid_10_classes/videos"),
            Path("/content/drive/MyDrive/BitirmeProjesi/tid_10_classes/videos"),
            Path("data/tid_10_classes/videos"),
            autsl_base / split_name,
            autsl_base / "train",
            autsl_base / "val",
            autsl_base / "test",
        ]

        for d in candidate_dirs:
            p = d / fname
            if p.exists():
                return p

        # Fallback to default
        video_dir = Path(self.config.data_dir if hasattr(self.config, "data_dir") else "videos")
        return video_dir / fname

    def get_label(self, index: int) -> int:
        return self.samples[index].label

    def get_metadata(self, index: int) -> SampleMetadata:
        return self.samples[index]
