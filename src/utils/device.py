"""
Device management utilities.

Auto-detects the best available compute device (CUDA, MPS, CPU)
and provides device-related helper functions.
"""

from __future__ import annotations

import torch


def get_device(preference: str = "auto") -> torch.device:
    """Get the best available compute device.

    Args:
        preference: Device preference ('auto', 'cuda', 'mps', 'cpu').

    Returns:
        torch.device for the selected device.
    """
    if preference == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    return torch.device(preference)


def get_device_info(device: torch.device) -> dict[str, str]:
    """Get device information for logging.

    Args:
        device: The compute device.

    Returns:
        Dictionary with device information.
    """
    info = {"device": str(device)}

    if device.type == "cuda":
        info["gpu_name"] = torch.cuda.get_device_name(device)
        info["gpu_memory"] = f"{torch.cuda.get_device_properties(device).total_memory / 1e9:.1f} GB"
        info["cuda_version"] = torch.version.cuda or "unknown"

    info["pytorch_version"] = torch.__version__

    return info
