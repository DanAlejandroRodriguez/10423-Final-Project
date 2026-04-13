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

import torch
from torch.utils.data import Dataset, Dataloader
from PIL import Image
from datasets import load_dataset
import torchvision.transforms as T

class DriveLMDataset(Dataset):
    def __init__(self, dataset_name="OpenDriveLab/DriveLM", split="train", transform=None):
        """
        Initialization for loading DriveLM data.
        """
        # Load the dataset from HuggingFace
        self.dataset = load_dataset(dataset_name, split=split)
        self.transform = transform if transform else T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
        ])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]

        # 1. Process Image
        # DriveLM often has multiple views; here we assume a primary camera
        image = item['image_paths'] 
        if not isinstance(image, Image.Image):
            image = Image.open(image).convert("RGB")
        
        image_tensor = self.transform(image)

        # 2. Process QA Pairs
        # DriveLM structure usually involves a list of QA objects
        question = item['Q']
        answer = item['A']

        # 3. Process Trajectory
        # Typically represented as a list of [x, y] coordinates
        # We convert this to a float tensor
        trajectory = torch.tensor(item['trajectory'], dtype=torch.float32)

        return {
            "image": image_tensor,
            "question": question,
            "answer": answer,
            "trajectory": trajectory
        }

def collate_fn(batch):
    """
    Custom collate to handle varying string lengths and tensors.
    """
    images = torch.stack([item['image'] for item in batch])
    questions = [item['question'] for item in batch]
    answers = [item['answer'] for item in batch]
    trajectories = torch.stack([item['trajectory'] for item in batch])

    return images, questions, answers, trajectories