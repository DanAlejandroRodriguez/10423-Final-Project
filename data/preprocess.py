"""
preprocess.py
=============
Skeleton for DriveLM feature extraction and tokenisation helpers.

This module will provide:
  - Image pre-processing transforms compatible with Gemma 4's vision encoder.
  - Text tokenisation helpers that wrap the Gemma 4 tokeniser.
  - Scene-difficulty estimation used by the adaptive-compute router.
  - Trajectory normalisation / de-normalisation utilities.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# NOTE: Imports below will be uncommented as implementation progresses.
# import numpy as np
# import torch
# import torchvision.transforms as T
# from PIL import Image
# from transformers import AutoProcessor


# ---------------------------------------------------------------------------
# Image transforms
# ---------------------------------------------------------------------------

def build_image_transform(
    image_size: Tuple[int, int] = (448, 448),
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
):
    """Build the torchvision transform pipeline for Gemma 4's vision encoder.

    Parameters
    ----------
    image_size : (H, W)
        Target spatial resolution expected by the vision encoder.
    mean, std :
        Per-channel normalisation statistics (ImageNet defaults).

    Returns
    -------
    torchvision.transforms.Compose
    """
    # TODO: Build and return a Compose transform:
    #   Resize → CenterCrop → ToTensor → Normalize
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Text tokenisation
# ---------------------------------------------------------------------------

def build_tokenizer(model_name_or_path: str = "google/gemma-3-4b-it"):
    """Load the Gemma 4 tokeniser / processor.

    Parameters
    ----------
    model_name_or_path : str
        HuggingFace model ID or local path.

    Returns
    -------
    transformers.AutoProcessor
        A processor that handles both image and text inputs for Gemma 4.
    """
    # TODO: Load via AutoProcessor.from_pretrained(model_name_or_path).
    raise NotImplementedError


def tokenize_qa_pair(
    question: str,
    answer: Optional[str],
    processor,
    images=None,
    max_length: int = 512,
    add_eos: bool = True,
) -> Dict[str, Any]:
    """Tokenise a (question, answer) pair for Gemma 4.

    Parameters
    ----------
    question : str
        The DriveLM question / instruction prompt.
    answer : str | None
        The ground-truth CoT answer. Pass ``None`` during inference.
    processor :
        The Gemma 4 processor returned by :func:`build_tokenizer`.
    images : list[PIL.Image] | Tensor | None
        Camera frames to include in the multimodal prompt.
    max_length : int
        Maximum total token length (question + answer).
    add_eos : bool
        Whether to append an EOS token to the answer.

    Returns
    -------
    dict
        ``input_ids``, ``attention_mask``, and ``labels`` tensors
        ready to be passed to the model.
    """
    # TODO: Format the prompt using Gemma 4's chat template.
    # TODO: Tokenise question and (optionally) answer.
    # TODO: Construct labels tensor (mask question tokens with -100).
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Trajectory utilities
# ---------------------------------------------------------------------------

def normalize_trajectory(
    waypoints,
    origin=None,
    scale: float = 1.0,
):
    """Normalise ego-vehicle waypoints to a canonical coordinate frame.

    Parameters
    ----------
    waypoints : Tensor[T, 2] | np.ndarray[T, 2]
        Raw (x, y) trajectory in metres (nuScenes ego frame).
    origin : array-like | None
        Reference point to subtract. Defaults to the first waypoint.
    scale : float
        Divisor applied after shifting (e.g., max expected displacement).

    Returns
    -------
    Tensor[T, 2]
        Normalised waypoints in the range approximately [-1, 1].
    """
    # TODO: Shift by origin, divide by scale.
    raise NotImplementedError


def denormalize_trajectory(
    waypoints_norm,
    origin=None,
    scale: float = 1.0,
):
    """Inverse of :func:`normalize_trajectory`.

    Parameters
    ----------
    waypoints_norm : Tensor[T, 2] | np.ndarray[T, 2]
        Normalised waypoints.
    origin : array-like | None
        The same origin used during normalisation.
    scale : float
        The same scale used during normalisation.

    Returns
    -------
    Tensor[T, 2]
        Waypoints in metres (nuScenes ego frame).
    """
    # TODO: Multiply by scale, add origin.
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Scene-difficulty estimation
# ---------------------------------------------------------------------------

def estimate_scene_difficulty(annotation: Dict[str, Any]) -> int:
    """Estimate scene complexity from a DriveLM annotation.

    Difficulty is approximated by counting the number of *critical objects*
    (vehicles, pedestrians, cyclists, and other dynamic obstacles) present
    in the scene description.  This count is used by the adaptive-compute
    router to decide between fast parallel decoding and MCTS search.

    Parameters
    ----------
    annotation : dict
        Raw annotation dict from the DriveLM JSON file, which includes
        structured object-level fields from FastDriveCoT's enumeration stage.

    Returns
    -------
    int
        Estimated number of critical objects in the scene.
    """
    # TODO: Parse the annotation's object list.
    # TODO: Filter for critical object categories.
    # TODO: Return the count.
    raise NotImplementedError


def difficulty_to_routing_label(
    difficulty: int,
    simple_threshold: int = 3,
    complex_threshold: int = 7,
) -> str:
    """Map an integer difficulty score to a compute-routing label.

    Parameters
    ----------
    difficulty : int
        Output of :func:`estimate_scene_difficulty`.
    simple_threshold : int
        Scenes with ``difficulty <= simple_threshold`` are routed to fast
        parallel decoding.
    complex_threshold : int
        Scenes with ``difficulty >= complex_threshold`` are routed to MCTS.
        Scenes in between receive an intermediate compute budget.

    Returns
    -------
    str
        One of ``"simple"``, ``"medium"``, or ``"complex"``.
    """
    if difficulty <= simple_threshold:
        return "simple"
    if difficulty >= complex_threshold:
        return "complex"
    return "medium"
