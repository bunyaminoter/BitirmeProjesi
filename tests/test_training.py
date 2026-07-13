"""Tests for training utilities."""

from __future__ import annotations

import pytest

from src.core.config import ExperimentConfig, TrainingConfig


class TestConfig:
    """Test configuration loading and defaults."""

    def test_default_experiment_config(self):
        """Test default ExperimentConfig values."""
        config = ExperimentConfig()
        assert config.seed == 42
        assert config.model.num_classes == 100
        assert config.training.batch_size == 8
        assert config.training.mixed_precision is True

    def test_training_config_defaults(self):
        """Test TrainingConfig defaults."""
        config = TrainingConfig()
        assert config.optimizer == "adamw"
        assert config.scheduler == "cosine"
        assert config.gradient_clip == 1.0


class TestOptimizers:
    """Test optimizer and scheduler factory."""

    def test_build_adamw(self):
        """Test building AdamW optimizer."""
        import torch.nn as nn
        from src.training.optimizers import build_optimizer

        model = nn.Linear(10, 5)
        optimizer = build_optimizer("adamw", model.parameters(), lr=1e-4)
        assert optimizer is not None

    def test_build_unknown_raises(self):
        """Test that unknown optimizer raises ValueError."""
        import torch.nn as nn
        from src.training.optimizers import build_optimizer

        model = nn.Linear(10, 5)
        with pytest.raises(ValueError, match="Unknown optimizer"):
            build_optimizer("unknown", model.parameters(), lr=1e-4)

    def test_build_cosine_scheduler(self):
        """Test building cosine scheduler."""
        import torch.nn as nn
        from src.training.optimizers import build_optimizer, build_scheduler

        model = nn.Linear(10, 5)
        optimizer = build_optimizer("adam", model.parameters(), lr=1e-4)
        scheduler = build_scheduler("cosine", optimizer, epochs=100)
        assert scheduler is not None
