"""
Configuration dataclasses and YAML loading utilities.

All configuration is defined as frozen dataclasses with sensible defaults.
Configs are loaded from YAML files and merged with CLI overrides.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


# ============================================================================
# Component-level configs
# ============================================================================


@dataclass
class HandEncoderConfig:
    """Configuration for the hand CNN encoder.

    Attributes:
        backbone: Torchvision model name (e.g., 'resnet18', 'efficientnet_b0').
        pretrained: Whether to load ImageNet-pretrained weights.
        output_dim: Dimensionality of the output feature vector.
        input_size: Expected input image size (H, W).
        shared_weights: If True, left and right hand share encoder weights.
        dropout: Dropout rate after the backbone's feature extractor.
    """

    backbone: str = "resnet18"
    pretrained: bool = True
    output_dim: int = 256
    input_size: List[int] = field(default_factory=lambda: [224, 224])
    shared_weights: bool = True
    dropout: float = 0.3


@dataclass
class LandmarkEncoderConfig:
    """Configuration for the landmark feature encoder.

    Attributes:
        input_dim: Flattened landmark dimension (num_landmarks * 3).
        hidden_dims: List of hidden layer dimensions.
        output_dim: Dimensionality of the output feature vector.
        dropout: Dropout rate between layers.
        use_batch_norm: Whether to apply batch normalization.
        include_face: Whether to include face landmarks in input.
        num_pose_landmarks: Number of pose landmarks to use.
        num_face_landmarks: Number of face landmarks to use (if enabled).
    """

    input_dim: int = 99  # 33 pose * 3
    hidden_dims: List[int] = field(default_factory=lambda: [256, 128])
    output_dim: int = 256
    dropout: float = 0.3
    use_batch_norm: bool = True
    include_face: bool = False
    num_pose_landmarks: int = 33
    num_face_landmarks: int = 468


@dataclass
class FusionConfig:
    """Configuration for the feature fusion module.

    Attributes:
        method: Fusion strategy name (e.g., 'concat', 'gated', 'attention').
        output_dim: Dimensionality of the fused output.
        dropout: Dropout rate in the fusion layer.
        num_heads: Number of attention heads (for attention-based fusion).
    """

    method: str = "concat"
    output_dim: int = 512
    dropout: float = 0.3
    num_heads: int = 4


@dataclass
class TemporalConfig:
    """Configuration for the temporal sequence model.

    Attributes:
        method: Temporal model name (e.g., 'bilstm', 'transformer', 'tcn').
        hidden_dim: Hidden dimension of the temporal model.
        num_layers: Number of recurrent/transformer layers.
        dropout: Dropout rate.
        num_heads: Number of attention heads (for transformer).
        max_seq_len: Maximum sequence length for positional encoding.
    """

    method: str = "bilstm"
    hidden_dim: int = 256
    num_layers: int = 2
    dropout: float = 0.3
    num_heads: int = 4
    max_seq_len: int = 64


@dataclass
class ModelConfig:
    """Top-level model architecture configuration.

    Attributes:
        hand_encoder: Hand CNN encoder configuration.
        landmark_encoder: Landmark encoder configuration.
        fusion: Feature fusion configuration.
        temporal: Temporal model configuration.
        num_classes: Number of output classes.
        classifier_dropout: Dropout before the final classification head.
    """

    hand_encoder: HandEncoderConfig = field(default_factory=HandEncoderConfig)
    landmark_encoder: LandmarkEncoderConfig = field(
        default_factory=LandmarkEncoderConfig
    )
    fusion: FusionConfig = field(default_factory=FusionConfig)
    temporal: TemporalConfig = field(default_factory=TemporalConfig)
    num_classes: int = 100
    classifier_dropout: float = 0.5


# ============================================================================
# Dataset & augmentation configs
# ============================================================================


@dataclass
class DatasetConfig:
    """Configuration for dataset loading.

    Attributes:
        name: Registered dataset name (e.g., 'wlasl').
        annotation_file: Path to annotation file.
        video_dir: Directory containing video files.
        num_classes: Number of classes to use (e.g., 100, 300, 1000).
        num_frames: Number of frames to sample per video.
        class_list_file: Path to class label list file.
        train_split: Split name for training data.
        val_split: Split name for validation data.
        test_split: Split name for test data.
    """

    name: str = "wlasl"
    annotation_file: str = "nslt_100.json"
    video_dir: str = "videos/"
    num_classes: int = 100
    num_frames: int = 16
    class_list_file: str = "wlasl_class_list.txt"
    train_split: str = "train"
    val_split: str = "val"
    test_split: str = "test"


@dataclass
class AugmentationConfig:
    """Configuration for data augmentation.

    Attributes:
        enabled: Master switch for augmentations.
        random_crop: Enable random spatial cropping.
        random_horizontal_flip: Enable random horizontal flip.
        random_rotation: Maximum rotation angle in degrees.
        random_brightness: Brightness jitter factor.
        random_noise_std: Standard deviation for Gaussian noise.
        frame_drop_prob: Probability of dropping a frame.
        random_speed_range: Range for temporal speed augmentation [min, max].
        hand_random_blur: Enable random blur on hand crops.
        hand_random_translation: Max translation in pixels for hand crops.
    """

    enabled: bool = True
    random_crop: bool = True
    random_horizontal_flip: bool = True
    random_rotation: float = 15.0
    random_brightness: float = 0.2
    random_noise_std: float = 0.01
    frame_drop_prob: float = 0.1
    random_speed_range: List[float] = field(default_factory=lambda: [0.8, 1.2])
    hand_random_blur: bool = True
    hand_random_translation: int = 10


# ============================================================================
# Training config
# ============================================================================


@dataclass
class TrainingConfig:
    """Configuration for the training loop.

    Attributes:
        epochs: Maximum number of training epochs.
        batch_size: Training batch size.
        learning_rate: Initial learning rate.
        weight_decay: L2 regularization weight.
        optimizer: Optimizer name ('adam', 'adamw', 'sgd').
        scheduler: LR scheduler name ('cosine', 'step', 'plateau', 'none').
        scheduler_params: Additional scheduler parameters.
        mixed_precision: Enable AMP mixed precision training.
        gradient_clip: Max gradient norm (0 to disable).
        num_workers: DataLoader worker processes.
        pin_memory: Pin memory for faster GPU transfer.
        early_stopping_patience: Epochs to wait before stopping (0 to disable).
        checkpoint_dir: Directory for saving checkpoints.
        resume_from: Path to checkpoint to resume from.
        log_every_n_steps: Logging frequency in training steps.
        val_every_n_epochs: Validation frequency in epochs.
    """

    epochs: int = 100
    batch_size: int = 8
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    optimizer: str = "adamw"
    scheduler: str = "cosine"
    scheduler_params: Dict[str, Any] = field(default_factory=dict)
    mixed_precision: bool = True
    gradient_clip: float = 1.0
    num_workers: int = 4
    pin_memory: bool = True
    early_stopping_patience: int = 15
    checkpoint_dir: str = "checkpoints/"
    resume_from: Optional[str] = None
    log_every_n_steps: int = 10
    val_every_n_epochs: int = 1


# ============================================================================
# Top-level experiment config
# ============================================================================


@dataclass
class ExperimentConfig:
    """Top-level experiment configuration.

    Composes all sub-configs into a single experiment definition.

    Attributes:
        name: Experiment name (used for logging and checkpointing).
        seed: Random seed for reproducibility.
        device: Device string ('cuda', 'cpu', 'mps', 'auto').
        dataset: Dataset configuration.
        model: Model architecture configuration.
        training: Training loop configuration.
        augmentation: Data augmentation configuration.
        output_dir: Base output directory for all experiment artifacts.
    """

    name: str = "default_experiment"
    seed: int = 42
    device: str = "auto"
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)
    output_dir: str = "outputs/"


# ============================================================================
# Config loading utilities
# ============================================================================


def _merge_dicts(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge override dict into base dict.

    Args:
        base: Base configuration dictionary.
        override: Override values to merge in.

    Returns:
        Merged dictionary (base is not modified).
    """
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _merge_dicts(result[key], value)
        else:
            result[key] = value
    return result


def _dict_to_config(data: Dict[str, Any]) -> ExperimentConfig:
    """Convert a nested dictionary to an ExperimentConfig.

    Args:
        data: Dictionary with config values.

    Returns:
        Fully populated ExperimentConfig.
    """
    dataset_cfg = DatasetConfig(**data.get("dataset", {}))
    augmentation_cfg = AugmentationConfig(**data.get("augmentation", {}))

    training_data = data.get("training", {})
    training_cfg = TrainingConfig(**training_data)

    model_data = data.get("model", {})
    hand_enc = HandEncoderConfig(**model_data.get("hand_encoder", {}))
    landmark_enc = LandmarkEncoderConfig(**model_data.get("landmark_encoder", {}))
    fusion_cfg = FusionConfig(**model_data.get("fusion", {}))
    temporal_cfg = TemporalConfig(**model_data.get("temporal", {}))

    model_cfg = ModelConfig(
        hand_encoder=hand_enc,
        landmark_encoder=landmark_enc,
        fusion=fusion_cfg,
        temporal=temporal_cfg,
        num_classes=model_data.get("num_classes", 100),
        classifier_dropout=model_data.get("classifier_dropout", 0.5),
    )

    return ExperimentConfig(
        name=data.get("name", "default_experiment"),
        seed=data.get("seed", 42),
        device=data.get("device", "auto"),
        dataset=dataset_cfg,
        model=model_cfg,
        training=training_cfg,
        augmentation=augmentation_cfg,
        output_dir=data.get("output_dir", "outputs/"),
    )


def load_config(
    path: str | Path,
    overrides: Optional[Dict[str, Any]] = None,
) -> ExperimentConfig:
    """Load an ExperimentConfig from a YAML file with optional overrides.

    Args:
        path: Path to the YAML configuration file.
        overrides: Optional dictionary of values to override.

    Returns:
        Fully populated ExperimentConfig.

    Raises:
        FileNotFoundError: If the config file does not exist.
        yaml.YAMLError: If the YAML is malformed.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    if overrides:
        data = _merge_dicts(data, overrides)

    return _dict_to_config(data)
