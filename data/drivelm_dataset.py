"""
drivelm_dataset.py
==================
Skeleton for the DriveLM dataset loader.

DriveLM is built on top of nuScenes and provides:
  - Multi-camera keyframes (6 surround cameras)
  - Structured QA pairs with Chain-of-Thought reasoning annotations
  - Ground-truth human trajectories for open-loop evaluation

This module will expose a PyTorch Dataset that:
  1. Reads the DriveLM JSON annotation files.
  2. Loads the corresponding nuScenes sensor frames.
  3. Returns (image_tensors, question, answer, trajectory) tuples
     ready for the Gemma 4 VLM backbone.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# NOTE: Imports below will be uncommented as implementation progresses.
# import numpy as np
# import torch
# from PIL import Image
# from torch.utils.data import Dataset
# from nuscenes.nuscenes import NuScenes


class DriveLMDataset:
    """PyTorch-compatible dataset for DriveLM / nuScenes driving logs.

    Parameters
    ----------
    drivelm_root : str | Path
        Root directory of the downloaded DriveLM dataset.
    nuscenes_root : str | Path
        Root directory of the nuScenes dataset (v1.0-trainval or mini).
    split : {"train", "val", "test"}
        Which data split to load.
    cameras : list[str] | None
        Camera names to include. Defaults to all 6 surround cameras.
    max_samples : int | None
        Optional cap on the number of samples (useful for debugging).
    transform : callable | None
        Image transform applied to each camera frame before batching.
    """

    # Default nuScenes surround-view camera channels
    DEFAULT_CAMERAS: List[str] = [
        "CAM_FRONT",
        "CAM_FRONT_LEFT",
        "CAM_FRONT_RIGHT",
        "CAM_BACK",
        "CAM_BACK_LEFT",
        "CAM_BACK_RIGHT",
    ]

    def __init__(
        self,
        drivelm_root: str | Path,
        nuscenes_root: str | Path,
        split: str = "train",
        cameras: Optional[List[str]] = None,
        max_samples: Optional[int] = None,
        transform=None,
    ) -> None:
        self.drivelm_root = Path(drivelm_root)
        self.nuscenes_root = Path(nuscenes_root)
        self.split = split
        self.cameras = cameras or self.DEFAULT_CAMERAS
        self.max_samples = max_samples
        self.transform = transform

        # Populated by _load_annotations()
        self._samples: List[Dict[str, Any]] = []

        self._load_annotations()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        """Return the number of samples in this split."""
        return len(self._samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Return a single sample dict.

        Keys (to be finalised during implementation):
            images      : Tensor[num_cameras, C, H, W]  — camera frames
            question    : str                            — DriveLM question text
            answer      : str                            — ground-truth CoT answer
            trajectory  : Tensor[T, 2]                  — (x, y) waypoints
            scene_token : str                            — nuScenes scene identifier
        """
        raise NotImplementedError("__getitem__ will be implemented in Phase 1.")

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_annotations(self) -> None:
        """Parse the DriveLM annotation JSON for the requested split.

        Expected file layout (subject to change after inspecting the DriveLM
        repository):
            <drivelm_root>/
                data/
                    QA_dataset_nus_v1_1/
                        train.json
                        val.json
                        test.json
        """
        # TODO: Locate and parse the annotation file for self.split.
        # TODO: Populate self._samples with one entry per QA pair.
        # TODO: Respect self.max_samples if set.
        pass

    def _load_images(self, scene_token: str, sample_token: str) -> Any:
        """Load and stack camera frames for a given nuScenes sample.

        Parameters
        ----------
        scene_token : str
            nuScenes scene identifier.
        sample_token : str
            nuScenes sample (keyframe) identifier.

        Returns
        -------
        Tensor[num_cameras, C, H, W]
            Stacked image tensors (after applying self.transform if set).
        """
        # TODO: Use the nuScenes devkit to retrieve image paths.
        # TODO: Open each image with PIL, apply self.transform, and stack.
        raise NotImplementedError

    def _parse_trajectory(self, annotation: Dict[str, Any]) -> Any:
        """Extract the ground-truth (x, y) waypoint trajectory.

        Parameters
        ----------
        annotation : dict
            Raw annotation dict from the DriveLM JSON file.

        Returns
        -------
        Tensor[T, 2]
            Trajectory waypoints in ego-vehicle coordinates (metres).
        """
        # TODO: Parse trajectory fields from the DriveLM annotation format.
        raise NotImplementedError


def build_dataloader(
    drivelm_root: str | Path,
    nuscenes_root: str | Path,
    split: str = "train",
    batch_size: int = 4,
    num_workers: int = 4,
    **dataset_kwargs,
):
    """Convenience factory that returns a DataLoader for DriveLMDataset.

    Parameters
    ----------
    drivelm_root, nuscenes_root, split :
        Forwarded to :class:`DriveLMDataset`.
    batch_size : int
        Mini-batch size.
    num_workers : int
        Number of parallel data-loading workers.
    **dataset_kwargs :
        Any additional keyword arguments forwarded to :class:`DriveLMDataset`.

    Returns
    -------
    torch.utils.data.DataLoader
    """
    # TODO: Instantiate DriveLMDataset.
    # TODO: Wrap in a DataLoader with a custom collate_fn that handles
    #       variable-length QA strings and trajectory tensors.
    raise NotImplementedError
