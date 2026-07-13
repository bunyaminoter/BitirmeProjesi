"""
Abstract base class for temporal sequence models.

Temporal models process sequences of per-frame feature vectors and
produce a single video-level representation. This captures the
temporal dynamics essential for sign language recognition.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class BaseTemporal(nn.Module, ABC):
    """Abstract base class for temporal sequence models.

    Takes a sequence of per-frame fused features and produces a
    single representation for the entire video clip.

    Input: (B, T, D) — batch of T time steps, each with D features.
    Output: (B, output_dim) — single video-level feature vector.

    Subclasses implement different temporal architectures:
        - BiLSTM
        - GRU
        - Temporal Convolution Network (TCN)
        - Transformer Encoder
    """

    @property
    @abstractmethod
    def output_dim(self) -> int:
        """Return the dimensionality of the output representation.

        Returns:
            Integer output dimension.
        """
        ...

    @abstractmethod
    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Process a sequence of features and produce a video representation.

        Args:
            x: (B, T, D) sequence of per-frame features.
            mask: (B, T) optional boolean mask (True = valid frame).

        Returns:
            (B, output_dim) video-level feature tensor.
        """
        ...
