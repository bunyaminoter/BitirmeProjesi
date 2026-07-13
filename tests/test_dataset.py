"""Tests for dataset interfaces and data types."""

from __future__ import annotations

import pytest

from src.core.types import LandmarkResult, HandCrop, SampleMetadata


class TestLandmarkResult:
    """Test LandmarkResult dataclass."""

    def test_empty_result(self):
        """Test default empty landmark result."""
        result = LandmarkResult()
        assert not result.has_pose
        assert not result.has_face
        assert not result.has_left_hand
        assert not result.has_right_hand

    def test_with_pose(self):
        """Test landmark result with pose data."""
        import numpy as np
        pose = np.zeros((33, 3), dtype=np.float32)
        result = LandmarkResult(pose=pose)
        assert result.has_pose
        assert not result.has_face


class TestHandCrop:
    """Test HandCrop dataclass."""

    def test_invalid_crop(self):
        """Test that empty crop reports invalid."""
        crop = HandCrop()
        assert not crop.is_valid

    def test_valid_crop(self):
        """Test that crop with image reports valid."""
        import numpy as np
        image = np.zeros((224, 224, 3), dtype=np.uint8)
        crop = HandCrop(image=image, handedness="right")
        assert crop.is_valid
        assert crop.handedness == "right"


class TestSampleMetadata:
    """Test SampleMetadata dataclass."""

    def test_defaults(self):
        """Test default metadata values."""
        meta = SampleMetadata()
        assert meta.video_id == ""
        assert meta.label == -1
        assert meta.split == ""

    def test_custom_values(self):
        """Test metadata with custom values."""
        meta = SampleMetadata(
            video_id="12345",
            label=42,
            gloss="hello",
            split="train",
        )
        assert meta.video_id == "12345"
        assert meta.label == 42
        assert meta.gloss == "hello"
