"""
Shared test fixtures and configuration for pytest.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Ensure project root is importable
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))


@pytest.fixture
def sample_config():
    """Return a minimal ExperimentConfig for testing."""
    from src.core.config import ExperimentConfig
    return ExperimentConfig(name="test_experiment", seed=42)


@pytest.fixture
def sample_model_config():
    """Return a minimal ModelConfig for testing."""
    from src.core.config import ModelConfig
    return ModelConfig(num_classes=10)
