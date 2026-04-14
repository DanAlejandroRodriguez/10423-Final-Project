"""
drivelm_dataset.py
==================
Skeleton for the DriveLM dataset loader.

DriveLM is built on top of nuScenes and provides:
  - Multi-camera keyframes (6 surround cameras)
  - Structured QA pairs with Chain-of-Thought reasoning annotations
  - Ground-truth human trajectories for open-loop evaluation
"""

import os
import json
import re
import torch
from torch.utils.data import Dataset
from PIL import Image
from typing import List, Tuple, Dict, Any, Optional
import numpy as np

class DriveLMDataset(Dataset):
    """
    PyTorch Dataset for DriveLM.
    
    Reads DriveLM json annotations, loads the multi-camera images, 
    and returns (images, question, answer, trajectory) tuples.
    """
    
    def __init__(self, data_root: str, ann_file: str, transform=None):
        """
        Args:
            data_root (str): Root directory of the nuScenes/DriveLM images.
            ann_file (str): Path to the DriveLM JSON annotation file.
            transform (callable, optional): Optional transform to be applied
                on the loaded images format.
        """
        super().__init__()
        self.data_root = data_root
        self.transform = transform
        
        if not os.path.exists(ann_file):
            print(f"Warning: Annotation file {ann_file} not found. Initialising empty dataset.")
            self.data = []
        else:
            with open(ann_file, 'r') as f:
                raw_data = json.load(f)
                # Parse depending on whether the json is a list of entries or has a 'data' key
                self.data = raw_data.get("data", raw_data) if isinstance(raw_data, dict) else raw_data
                
                # If the dataset is a dictionary mapping IDs to samples, flatten the values into a list for integer indexing
                if isinstance(self.data, dict):
                    self.data = list(self.data.values())
            
    def __len__(self) -> int:
        return len(self.data)
        
    def __getitem__(self, idx: int) -> Tuple[List[torch.Tensor], str, str, torch.Tensor]:
        """
        Retrieves the item at the specified index.
        
        Returns:
            images (List[torch.Tensor]): List of 6 surround camera image tensors.
            question (str): The DriveLM QA question string.
            answer (str): The DriveLM QA answer string (which may have cot, action).
            trajectory (torch.Tensor): Ground-truth human trajectories of shape (T, 2).
        """
        item = self.data[idx]
        
        # 1. Load multi-camera images (6 surround cameras from nuScenes)
        cam_names = [
            'CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT',
            'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT'
        ]
        
        images = []
        # DriveLM typically references camera images within the data item
        cam_paths = item.get('camera_paths', {})
        
        for cam in cam_names:
            rel_path = cam_paths.get(cam, "")
            full_path = os.path.join(self.data_root, rel_path)
            
            # Load the image if the file exists, otherwise generate a dummy tensor (useful for testing)
            if rel_path and os.path.exists(full_path):
                img = Image.open(full_path).convert('RGB')
            else:
                # Black dummy image placeholder
                img = Image.new('RGB', (224, 224))
                
            # Apply image transforms compatible with Gemma 4's vision encoder
            if self.transform is not None:
                img = self.transform(img)
            else:
                # Default to basic FloatTensor conversion
                img = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
                
            images.append(img)
            
        # 2. Extract Question and Answer strings
        qa_pairs = item.get("QA_pairs", {})
        
        # Collect ALL questions from perception, prediction, planning, behavior
        all_qas = []
        for category, qas in qa_pairs.items():
            if isinstance(qas, list):
                for qa in qas:
                    if qa.get("Q", "").strip():
                        all_qas.append(f"Q: {qa['Q']}")
                        
        # Join them to form a unified question context
        question = "\n".join(all_qas)
        if not question:
            question = "Describe the autonomous driving scene."
        
        # 3. Extract the target action Answer and Trajectory
        planning_qas = qa_pairs.get("planning", [])
        answer = ""
        traj_list = []
        
        for qa in planning_qas:
            text_ans = qa.get("A", "")
            if "actions" in qa.get("Q", "").lower():
                answer = text_ans
                
            # Extract trajectory coordinates if present
            matches = re.findall(r'\(\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*\)', text_ans)
            if matches:
                traj_list = [[float(x), float(y)] for x, y in matches]
                
        # Simple fallback for empty trajectories (allows evaluators to handle dynamic lengths safely)
        if not traj_list:
            traj_list = [[0.0, 0.0]]
            
        trajectory = torch.tensor(traj_list, dtype=torch.float32)
        
        return images, question, answer, trajectory
