"""
Spatial-Temporal Graph Convolutional Network (ST-GCN) encoder for pose landmarks.

Models the skeleton as a graph to learn the spatial relationships between joints.
"""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn
import numpy as np

from src.core.config import LandmarkEncoderConfig
from src.core.registry import ENCODER_REGISTRY
from src.models.encoders.base_encoder import BaseEncoder

# Pre-defined graph structure for MediaPipe Pose (33 landmarks)
# Edges define the connections between joints
_POSE_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 7),  # Left eye and ear
    (0, 4), (4, 5), (5, 6), (6, 8),  # Right eye and ear
    (9, 10),                         # Mouth
    (11, 12),                        # Shoulders
    (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19), # Left arm and hand
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20), # Right arm and hand
    (11, 23), (12, 24), (23, 24),    # Torso
    (23, 25), (25, 27), (27, 29), (27, 31), (29, 31), # Left leg and foot
    (24, 26), (26, 28), (28, 30), (28, 32), (30, 32)  # Right leg and foot
]

class Graph:
    """The Graph to model the skeleton topology."""

    def __init__(self, num_node: int = 33, edges: List[tuple[int, int]] = _POSE_CONNECTIONS):
        self.num_node = num_node
        self.edges = edges
        self.A = self._get_adjacency()

    def _get_adjacency(self) -> np.ndarray:
        # A simple adjacency matrix (with self loops)
        A = np.zeros((self.num_node, self.num_node))
        for i, j in self.edges:
            if i < self.num_node and j < self.num_node:
                A[i, j] = 1
                A[j, i] = 1
        for i in range(self.num_node):
            A[i, i] = 1
        
        # Normalize
        d = np.sum(A, axis=1)
        d_inv = 1.0 / (d + 1e-6)
        A = A * d_inv[:, None]
        return A

class SpatialGraphConv(nn.Module):
    """Spatial Graph Convolution Layer."""
    
    def __init__(self, in_channels: int, out_channels: int, A: torch.Tensor, dropout: float = 0.0):
        super().__init__()
        self.register_buffer("A", A)
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)
        
        # Residual connection if channels don't match
        self.down = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm1d(out_channels)
        ) if in_channels != out_channels else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B*T, C, V)
        B_T, C, V = x.shape
        
        # Graph convolution (A: V x V)
        # We want to do: (B*T, C, V) @ (V, V) -> (B*T, C, V)
        x_a = torch.einsum("bcv,vw->bcw", x, self.A)
        
        # Feature transformation
        out = self.conv(x_a)
        out = self.bn(out)
        
        # Residual
        res = self.down(x)
        out = out + res
        
        out = self.relu(out)
        out = self.dropout(out)
        
        return out


@ENCODER_REGISTRY.register("landmark_stgcn")
class LandmarkSTGCNEncoder(BaseEncoder):
    """Spatial-Temporal Graph Convolutional Network (ST-GCN) for pose landmarks.

    Models the skeleton as a graph and uses graph convolution layers to
    learn the spatial relationships between joints.
    """

    def __init__(self, config: LandmarkEncoderConfig) -> None:
        super().__init__()
        self._config = config
        self._output_dim = config.output_dim
        
        # Get graph adjacency matrix
        graph = Graph(num_node=config.num_pose_landmarks, edges=_POSE_CONNECTIONS)
        A = torch.tensor(graph.A, dtype=torch.float32)
        
        # Build GCN layers
        layers = []
        in_c = 3  # (x, y, z) per node
        for out_c in config.gcn_channels:
            layers.append(SpatialGraphConv(in_c, out_c, A, dropout=config.dropout))
            in_c = out_c
            
        self.gcn = nn.Sequential(*layers)
        
        # Final projection after global average pooling
        self.fc = nn.Sequential(
            nn.Linear(in_c, config.output_dim),
            nn.ReLU(inplace=True),
        )

    @property
    def output_dim(self) -> int:
        return self._output_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode landmark coordinates to a feature vector using GCN.

        Args:
            x: (B*T, num_landmarks, 3) float tensor, or flattened (B*T, num_landmarks * 3).

        Returns:
            (B*T, output_dim) feature tensor.
        """
        # Ensure correct shape for GCN: (B*T, C, V)
        # The input x is either (B*T, 99) flattened, or already (B*T, 33, 3).
        if x.dim() == 2:
            # Re-shape from flattened (B*T, 99) -> (B*T, 33, 3)
            B_T, D = x.shape
            V = self._config.num_pose_landmarks
            C = D // V
            x = x.view(B_T, V, C)
            
        # x is now (B*T, V, C)
        # We need (B*T, C, V) for 1D convolution over nodes
        x = x.permute(0, 2, 1)
        
        # GCN forward
        out = self.gcn(x)  # (B*T, in_c, V)
        
        # Global average pooling over nodes (V)
        out = out.mean(dim=-1)  # (B*T, in_c)
        
        # Final projection
        out = self.fc(out)  # (B*T, output_dim)
        
        return out
