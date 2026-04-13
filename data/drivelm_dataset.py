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

import os
import json
import re
from torch.utils.data import Dataset
from PIL import Image
from huggingface_hub import hf_hub_download

class DriveLMDataset(Dataset):
    CAMERA_VIEWS = [
        'CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT',
        'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT'
    ]

    def __init__(self, dataset_name="OpenDriveLab/DriveLM", split="train", nuscenes_img_dir="data/raw/nuscenes_mini/samples"):
        """
        Initialization for loading DriveLM data.
        """
        self.img_dir = nuscenes_img_dir

        filename = f"v1_1_{split}_nus.json" if split == "train" else f"v1_1_val_nus_q_only.json"
        print(f"Downloading/Loading {filename} from Hugging Face...")
        json_path = hf_hub_download(repo_id=dataset_name, filename=filename, repo_type="dataset")

        with open(json_path, 'r') as f:
            raw_data = json.load(f)

        self.scenes = {}
        for scene_id, scene_info in raw_data.items():
            key_frames = scene_info.get("key_frames", {})
            for sample_token, frame_info in key_frames.items():
                
                qas = []
                qa_dict = frame_info.get("QA", {})
                if isinstance(qa_dict, dict):
                    for _, qa_list in qa_dict.items():
                        for qa_item in qa_list:
                            qas.append({
                                "question": qa_item.get("Q", ""),
                                "answer": qa_item.get("A", "")
                            })
                
                self.scenes[sample_token] = {
                    "token": sample_token,
                    "scene_id": scene_id,
                    "qas": qas,
                    "trajectory": self._extract_trajectory(qa_dict),
                    "graph": frame_info
                }
            
        self.scene_list = list(self.scenes.values())
        print(f"Loaded {len(self.scene_list)} unique scenes/keyframes from {split} split.")
        has_traj = sum(1 for s in self.scene_list if s["trajectory"])
        print(f"  → {has_traj}/{len(self.scene_list)} frames have parseable trajectories.")

    def _extract_trajectory(self, qa_dict):
        planning_qas = qa_dict.get("planning", [])
        for qa in planning_qas:
            answer = qa.get("A", "")
            matches = re.findall(r'\(\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*\)', answer)
            if matches:
                return [[float(x), float(y)] for x, y in matches]
        return []

    def _load_images_for_token(self, sample_token):
        """Helper to load all 6 camera images for a given scene token."""
        images = []
        for cam in self.CAMERA_VIEWS:
            cam_dir = os.path.join(self.img_dir, cam)
            matched_file = None
            if os.path.exists(cam_dir):
                for fname in os.listdir(cam_dir):
                    if sample_token in fname and fname.endswith(('.jpg', '.png')):
                        matched_file = os.path.join(cam_dir, fname)
                        break
            
            if matched_file:
                img = Image.open(matched_file).convert("RGB")
            else:
                img = Image.new("RGB", (1600, 900))
            
            images.append(img)
        return images

    def __len__(self):
        return len(self.scene_list)

    def __getitem__(self, idx):
        scene_data = self.scene_list[idx]

        images = self._load_images_for_token(scene_data["token"])

        question = "\n".join([
            f"Q: {qa['question']}" 
            for qa in scene_data["qas"]
            if qa['question'].strip()
        ])

        return {
            "images": images,
            "question": question,
            "answer": scene_data["graph"],
            "gt_trajectory": scene_data["trajectory"],
            "token": scene_data["token"]
        }

def collate_fn(batch):
    images = [item['images'] for item in batch]
    questions = [item['question'] for item in batch]
    answers = [item['answer'] for item in batch]
    trajectories = [item['gt_trajectory'] for item in batch]

    return images, questions, answers, trajectories