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
