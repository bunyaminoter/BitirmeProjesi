"""
Explainable AI (XAI) utilities for the Hybrid ASL Model.

Provides Grad-CAM visualizations for the Hand CNN encoders and
attention map visualizations for the spatial-temporal components.
"""

from __future__ import annotations

import copy
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class GradCAM:
    """Computes Grad-CAM for a specific CNN layer."""

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output.detach()

    def save_gradient(self, module, grad_input, grad_output):
        # grad_output[0] is the gradient with respect to the output of the layer
        self.gradients = grad_output[0].detach()

    def __call__(self, x_batch: Dict[str, torch.Tensor], target_class: int) -> np.ndarray:
        """Compute Grad-CAM for a batch of inputs.
        
        Args:
            x_batch: Input batch dict.
            target_class: The class index to compute gradients for.
            
        Returns:
            cam: A numpy array of shape (B, T, H, W) containing the CAMs.
        """
        # Forward pass
        self.model.eval()
        # Enable gradients for the input if needed, though we usually just need it for the model parameters/activations.
        self.model.zero_grad()
        
        logits = self.model(x_batch) # (B, num_classes)
        
        # Target for backprop
        if logits.dim() == 2:
            score = logits[:, target_class].sum()
        else:
            score = logits[target_class]
            
        # Backward pass
        score.backward()
        
        # Get activations and gradients
        # activations: (B*T, C, H, W)
        # gradients: (B*T, C, H, W)
        if self.activations is None or self.gradients is None:
            raise RuntimeError("Failed to capture activations/gradients. Check target_layer.")
            
        # Global average pooling on gradients to get weights
        # weights: (B*T, C, 1, 1)
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        
        # Weighted sum of activations
        # cam: (B*T, 1, H, W)
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)
        cam = F.relu(cam) # ReLU on CAM
        
        # Normalize between 0 and 1
        cam_min, _ = torch.min(cam.view(cam.size(0), -1), dim=1, keepdim=True)
        cam_max, _ = torch.max(cam.view(cam.size(0), -1), dim=1, keepdim=True)
        cam_min = cam_min.view(-1, 1, 1, 1)
        cam_max = cam_max.view(-1, 1, 1, 1)
        cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)
        
        # Interpolate to original image size
        # Hand images are usually 224x224
        cam = F.interpolate(cam, size=(224, 224), mode='bilinear', align_corners=False)
        
        # Reshape back to (B, T, H, W)
        B = x_batch["left_hand_images"].shape[0]
        T = x_batch["left_hand_images"].shape[1]
        
        cam = cam.squeeze(1).view(B, T, 224, 224).cpu().numpy()
        
        return cam

def overlay_cam_on_image(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Overlays the CAM mask onto the image.
    
    Args:
        img: (H, W, 3) RGB image in [0, 1] or [0, 255].
        mask: (H, W) CAM mask in [0, 1].
        
    Returns:
        (H, W, 3) heatmap overlaid on image.
    """
    if img.max() <= 1.0:
        img = (img * 255).astype(np.uint8)
        
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
    overlay = cv2.addWeighted(img, 0.5, heatmap, 0.5, 0)
    return overlay
