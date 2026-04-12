# Data Setup

This document explains how to download and prepare the data required for this project.

---

## 1. nuScenes

DriveLM is built on top of the **nuScenes** autonomous-driving dataset.

1. Register for a free account at [nuScenes.org](https://www.nuscenes.org/sign-up).
2. Download **nuScenes v1.0-trainval** (or the smaller **v1.0-mini** for development):
   ```
   # Example directory layout
   /data/nuscenes/
       v1.0-trainval/
           samples/
           sweeps/
           maps/
           v1.0-trainval/
   ```
3. Set the environment variable `NUSCENES_ROOT` to point to this directory:
   ```bash
   export NUSCENES_ROOT=/data/nuscenes
   ```

---

## 2. DriveLM

1. Clone or download the DriveLM dataset from the official repository:
   ```bash
   git clone https://github.com/OpenDriveLab/DriveLM.git
   ```
2. Follow the DriveLM data-preparation instructions to generate the QA annotation JSON files.
3. Set the environment variable `DRIVELM_ROOT` to point to the resulting directory:
   ```bash
   export DRIVELM_ROOT=/data/drivelm
   ```

Expected layout after setup:
```
/data/drivelm/
    data/
        QA_dataset_nus_v1_1/
            train.json
            val.json
            test.json
```

---

## 3. Verify setup

After completing the steps above, verify the paths are accessible:

```python
from data.drivelm_dataset import DriveLMDataset

dataset = DriveLMDataset(
    drivelm_root="/data/drivelm",
    nuscenes_root="/data/nuscenes",
    split="val",
    max_samples=10,
)
print(f"Loaded {len(dataset)} samples.")
```

> **Note:** The dataset loader is not yet implemented. This verification step will be uncommented once `drivelm_dataset.py` is complete.

---

## Module Overview

| File | Description |
|---|---|
| `drivelm_dataset.py` | PyTorch Dataset for DriveLM — loads images, QA pairs, and trajectories |
| `preprocess.py` | Image transforms, Gemma 4 tokenisation, trajectory normalisation, scene-difficulty estimation |
