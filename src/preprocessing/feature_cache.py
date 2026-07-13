"""
Disk-based feature caching for preprocessed landmarks and hand crops.

Avoids re-running MediaPipe during training by storing extracted features
on disk. Supports both NumPy (.npz) and HDF5 formats.

This is the single biggest training speedup: MediaPipe extraction at
~30ms/frame dominates preprocessing time. Caching eliminates this
from the training loop entirely.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np


class FeatureCache:
    """Manages disk-based caching of preprocessed features.

    Features are stored per video_id in a configurable directory.
    Each cached entry contains:
        - Pose landmarks for all frames
        - Hand landmarks for all frames
        - Hand crop images (optional, can be large)
        - Metadata (frame indices, video info)

    Attributes:
        cache_dir: Root directory for cached features.
        cache_format: Storage format ('npz' or 'hdf5').
        store_hand_crops: Whether to cache hand crop images.
    """

    def __init__(
        self,
        cache_dir: str | Path,
        cache_format: str = "npz",
        store_hand_crops: bool = False,
    ) -> None:
        """Initialize the feature cache.

        Args:
            cache_dir: Directory for storing cached features.
            cache_format: Storage format ('npz' or 'hdf5').
            store_hand_crops: Whether to store hand crop images
                             (significantly increases disk usage).
        """
        self.cache_dir = Path(cache_dir)
        self.cache_format = cache_format
        self.store_hand_crops = store_hand_crops
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_cache_path(self, video_id: str) -> Path:
        """Get the cache file path for a given video ID.

        Args:
            video_id: Unique video identifier.

        Returns:
            Path to the cache file.
        """
        ext = ".npz" if self.cache_format == "npz" else ".h5"
        return self.cache_dir / f"{video_id}{ext}"

    def exists(self, video_id: str) -> bool:
        """Check if cached features exist for a video.

        Args:
            video_id: Unique video identifier.

        Returns:
            True if cached features exist.
        """
        return self._get_cache_path(video_id).exists()

    def save(self, video_id: str, features: Dict[str, Any]) -> None:
        """Save extracted features to disk.

        Args:
            video_id: Unique video identifier.
            features: Dictionary of feature arrays to cache.
        """
        cache_path = self._get_cache_path(video_id)

        if self.cache_format == "npz":
            np.savez_compressed(str(cache_path), **features)
        else:
            # TODO: Implement HDF5 storage for very large datasets
            raise NotImplementedError("HDF5 caching not yet implemented")

    def load(self, video_id: str) -> Optional[Dict[str, np.ndarray]]:
        """Load cached features from disk.

        Args:
            video_id: Unique video identifier.

        Returns:
            Dictionary of feature arrays, or None if not cached.
        """
        cache_path = self._get_cache_path(video_id)
        if not cache_path.exists():
            return None

        if self.cache_format == "npz":
            data = np.load(str(cache_path), allow_pickle=False)
            return dict(data)
        else:
            raise NotImplementedError("HDF5 loading not yet implemented")

    def clear(self) -> None:
        """Remove all cached features.

        Deletes all files in the cache directory.
        """
        for path in self.cache_dir.iterdir():
            if path.is_file():
                path.unlink()

    @property
    def num_cached(self) -> int:
        """Return the number of cached video features."""
        ext = ".npz" if self.cache_format == "npz" else ".h5"
        return len(list(self.cache_dir.glob(f"*{ext}")))
