"""
Convolutional Block Attention Module (CBAM)

Attention mechanism that computes attention maps sequentially along two separate
dimensions: channel and spatial, then multiplies the attention maps to the
input feature map for adaptive feature refinement.
"""

import torch
import torch.nn as nn

class ChannelAttention(nn.Module):
    def __init__(self, in_planes: int, reduction_ratio: int = 16):
        super().__init__()
        # Clamp reduction ratio to not be too small (at least 1)
        reduced_planes = max(1, in_planes // reduction_ratio)
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        # Shared MLP
        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, reduced_planes, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced_planes, in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_concat = torch.cat([avg_out, max_out], dim=1)
        out = self.conv1(x_concat)
        return self.sigmoid(out)


class CBAMBlock(nn.Module):
    """CBAM: Convolutional Block Attention Module.
    
    Sequentially applies Channel Attention and Spatial Attention.
    """
    def __init__(self, in_planes: int, reduction_ratio: int = 16, spatial_kernel_size: int = 7):
        super().__init__()
        self.channel_attention = ChannelAttention(in_planes, reduction_ratio)
        self.spatial_attention = SpatialAttention(spatial_kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Apply Channel Attention
        out = x * self.channel_attention(x)
        # Apply Spatial Attention
        out = out * self.spatial_attention(out)
        return out
