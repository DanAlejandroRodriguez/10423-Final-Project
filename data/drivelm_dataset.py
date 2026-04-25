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
     ready for the Qwen2.5-VL backbone.
"""

import os
import json
import numpy as np
import tarfile
from pyquaternion import Quaternion
from torch.utils.data import Dataset
from PIL import Image
from huggingface_hub import hf_hub_download
from nuscenes.nuscenes import NuScenes
from dotenv import load_dotenv

load_dotenv()

_DRIVELM_IMG_PREFIX = "../nuscenes/"

class DriveLMDataset(Dataset):
    CAMERA_VIEWS = [
        'CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT',
        'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT'
    ]

    def __init__(self, dataset_name="OpenDriveLab/DriveLM", split="train", nuscenes_img_dir="data/raw/nuscenes"):
        """
        Initialization for loading DriveLM data.
        """
        self.img_dir = nuscenes_img_dir
        # self.nusc_version = "v1.0-trainval"
        
        self.nusc_version = "v1.0-mini"
        self.nuscenes_root = nuscenes_img_dir  
        dataroot = nuscenes_img_dir
        version_dir = os.path.join(dataroot, self.nusc_version)
        
        if not os.path.exists(os.path.join(version_dir, 'category.json')):
            meta_tar = os.path.join(dataroot, f"{self.nusc_version}_meta.tgz")
            if os.path.exists(meta_tar):
                print(f"Extracting metadata from {meta_tar} to {dataroot}...")
                with tarfile.open(meta_tar, "r:gz") as tar:
                    tar.extractall(path=dataroot)
                print("Metadata extraction complete!")
            else:
                print(f"Error: Neither the extracted {self.nusc_version} folder nor {self.nusc_version}_meta.tgz found in {dataroot}.")

        try:
            self.nusc = NuScenes(version=self.nusc_version, dataroot=dataroot, verbose=False)
        except Exception as e:
            print(f"Warning: Could not load nuScenes devkit: {e}")
            self.nusc = None

        filename = f"v1_1_{split}_nus.json" if split == "train" else f"v1_1_val_nus_q_only.json"
        print(f"Downloading/Loading {filename} from Hugging Face...")
        json_path = hf_hub_download(repo_id=dataset_name, filename=filename,
                                    repo_type="dataset", token=os.environ.get("HF_TOKEN"))

        with open(json_path, 'r') as f:
            raw_data = json.load(f)

        self.scenes = {}
        for scene_id, scene_info in raw_data.items():
            key_frames = scene_info.get("key_frames", {})
            for sample_token, frame_info in key_frames.items():
                
                qas = []
                qa_dict = frame_info.get("QA", {})
                if isinstance(qa_dict, dict):
                    for task, qa_list in qa_dict.items():
                        for qa_item in qa_list:
                            qas.append({
                                "task": task,
                                "question": qa_item.get("Q", ""),
                                "answer": qa_item.get("A", "")
                            })
                
                self.scenes[sample_token] = {
                    "token": sample_token,
                    "scene_id": scene_id,
                    "qas": qas,
                    "image_paths": self._resolve_image_paths(frame_info),
                    "trajectory": self._extract_trajectory(sample_token),
                    "graph": frame_info
                }
            
        self.scene_list = list(self.scenes.values())
        print(f"Loaded {len(self.scene_list)} unique scenes/keyframes from {split} split.")
        
        self.scene_list = [s for s in self.scene_list if len(s.get("trajectory", [])) > 0]
        print(f"  → Filtered down to {len(self.scene_list)} frames that have native NuScenes trajectories.")

    def _resolve_image_paths(self, frame_info):
        """Convert DriveLM v1.1 relative image_paths to absolute paths."""
        resolved = {}
        for cam, rel_path in frame_info.get("image_paths", {}).items():
            if rel_path.startswith(_DRIVELM_IMG_PREFIX):
                rel_path = rel_path[len(_DRIVELM_IMG_PREFIX):]
            resolved[cam] = os.path.join(self.nuscenes_root, rel_path)
        return resolved

    def _extract_trajectory(self, sample_token, future_steps=6):
        """Extract future trajectories using nuScenes devkit."""
        if self.nusc is None:
            return []
        
        try:
            sample = self.nusc.get('sample', sample_token)
            sample_data = self.nusc.get('sample_data', sample['data']['CAM_FRONT'])
            ego_pose = self.nusc.get('ego_pose', sample_data['ego_pose_token'])
            
            ref_translation = np.array(ego_pose['translation'])
            ref_rotation = Quaternion(ego_pose['rotation']).rotation_matrix
            
            trajectory = []
            curr_sample = sample
            for _ in range(future_steps):
                if curr_sample['next'] == '':
                    break
                curr_sample = self.nusc.get('sample', curr_sample['next'])
                curr_sd = self.nusc.get('sample_data', curr_sample['data']['CAM_FRONT'])
                next_pose = self.nusc.get('ego_pose', curr_sd['ego_pose_token'])
                
                glob_pos = np.array(next_pose['translation'])
                
                local_pos = glob_pos - ref_translation
                local_pos = ref_rotation.T @ local_pos
                
                trajectory.append([float(local_pos[0]), float(local_pos[1])])
                
            return trajectory
        except Exception:
            return []

    def _load_images(self, image_paths):
        """Helper to load all 6 camera images using pre-resolved paths from the JSON."""
        images = []
        for cam in self.CAMERA_VIEWS:
            path = image_paths.get(cam, "")
            if path and os.path.exists(path):
                img = Image.open(path).convert("RGB")
            else:
                img = Image.new("RGB", (1600, 900))
            images.append(img)
        return images

    def __len__(self):
        return len(self.scene_list)

    def __getitem__(self, idx):
        scene_data = self.scene_list[idx]

        images = self._load_images(scene_data["image_paths"])

        question = "\n".join([
            f"[{qa['task']}] Q: {qa['question']}" 
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
