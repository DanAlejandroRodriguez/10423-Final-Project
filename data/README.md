# DriveLM Dataset Setup

Two parts: DriveLM annotations (automatic) and nuScenes images (manual download).

## 1. DriveLM Annotations
Handled automatically. `DriveLMDataset` downloads the JSON from `OpenDriveLab/DriveLM` on Hugging Face at runtime.

## 2. nuScenes v1.0-trainval

1. Go to [nuscenes.org/download](https://www.nuscenes.org/download), create a free account, and accept the terms.
2. Download **v1.0-trainval** — you need:
   - `v1.0-trainval_meta.tgz` (metadata)
   - Any blob files you need for images (`v1.0-trainval_blob01.tgz` through `v1.0-trainval_blob10.tgz`)
3. Place everything under `data/raw/nuscenes/` and extract. The metadata tarball will unpack as `v1.0-trainval/`. The blobs unpack into `samples/`.

Expected structure:
```
data/raw/nuscenes/
├── v1.0-trainval/
│   ├── category.json
│   ├── ego_pose.json
│   ├── sample.json
│   ├── sample_data.json
│   └── ... (all metadata files)
└── samples/
    ├── CAM_FRONT/
    ├── CAM_FRONT_LEFT/
    ├── CAM_FRONT_RIGHT/
    ├── CAM_BACK/
    ├── CAM_BACK_LEFT/
    └── CAM_BACK_RIGHT/
```

## 3. Usage

```python
from data.drivelm_dataset import DriveLMDataset

dataset = DriveLMDataset(split="train", nuscenes_img_dir="data/raw/nuscenes")
```

The metadata tarball is auto-extracted by `DriveLMDataset` if the `v1.0-trainval/` folder is missing but the `.tgz` is present.
