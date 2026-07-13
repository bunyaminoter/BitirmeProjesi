"""Tests for model components."""

from __future__ import annotations

import pytest
import torch

from src.core.config import (
    ModelConfig,
    HandEncoderConfig,
    LandmarkEncoderConfig,
    FusionConfig,
    TemporalConfig,
)


class TestLandmarkEncoder:
    """Test the landmark MLP encoder."""

    def test_forward_shape(self):
        """Test that output shape is correct."""
        from src.models.encoders.landmark_encoder import LandmarkEncoder

        config = LandmarkEncoderConfig(
            input_dim=99,
            hidden_dims=[128],
            output_dim=64,
        )
        encoder = LandmarkEncoder(config)

        x = torch.randn(4, 99)  # batch=4, input=99
        output = encoder(x)
        assert output.shape == (4, 64)

    def test_output_dim_property(self):
        """Test output_dim property."""
        from src.models.encoders.landmark_encoder import LandmarkEncoder

        config = LandmarkEncoderConfig(output_dim=128)
        encoder = LandmarkEncoder(config)
        assert encoder.output_dim == 128


class TestConcatFusion:
    """Test concatenation fusion module."""

    def test_forward_shape(self):
        """Test fused output shape."""
        from src.models.fusion.concat_fusion import ConcatFusion

        config = FusionConfig(output_dim=256)
        fusion = ConcatFusion(config, input_dims=[128, 64, 64])

        features = [
            torch.randn(4, 128),
            torch.randn(4, 64),
            torch.randn(4, 64),
        ]
        output = fusion(features)
        assert output.shape == (4, 256)


class TestGatedFusion:
    """Test gated fusion module."""

    def test_forward_shape(self):
        """Test gated fusion output shape."""
        from src.models.fusion.gated_fusion import GatedFusion

        config = FusionConfig(output_dim=128)
        fusion = GatedFusion(config, input_dims=[64, 64, 64])

        features = [torch.randn(4, 64) for _ in range(3)]
        output = fusion(features)
        assert output.shape == (4, 128)


class TestBiLSTMTemporal:
    """Test BiLSTM temporal model."""

    def test_forward_shape(self):
        """Test temporal output shape."""
        from src.models.temporal.bilstm_temporal import BiLSTMTemporal

        config = TemporalConfig(hidden_dim=128, num_layers=2)
        model = BiLSTMTemporal(config, input_dim=256)

        x = torch.randn(4, 16, 256)  # batch=4, seq_len=16, dim=256
        output = model(x)
        assert output.shape == (4, 128)


class TestTransformerTemporal:
    """Test Transformer temporal model."""

    def test_forward_shape(self):
        """Test transformer output shape."""
        from src.models.temporal.transformer_temporal import TransformerTemporal

        config = TemporalConfig(hidden_dim=128, num_layers=2, num_heads=4)
        model = TransformerTemporal(config, input_dim=256)

        x = torch.randn(4, 16, 256)
        output = model(x)
        assert output.shape == (4, 128)
