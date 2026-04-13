# DriveLM Dataset Setup

Please keep the raw dataset files locally in the `data/raw/` folder. 

## Setup Instructions

1. **Create the data directories** in the root of the project:
   ```bash
   mkdir -p data/raw/DriveLM
   mkdir -p data/raw/nuScenes
   ```

2. **Download nuScenes & DriveLM** into the newly created folders.
   - You need the corresponding nuScenes v1.0-mini or trainval dataset.
   - Place the structural JSONs from DriveLM right beside it.

   *Your folder structure inside `data/` should look like this after downloading:*
   ```text
   data/
   ├── __init__.py
   ├── drivelm_dataset.py
   ├── preprocess.py
   ├── README.md              ← You are here
   └── raw/
       ├── nuScenes/
       │   ├── maps/
       │   ├── samples/       ← The actual images are here
       │   ├── sweeps/
       │   └── v1.0-mini/     ← nuScenes metadata
       └── DriveLM/
           ├── drivelm_train.json
           └── drivelm_val.json
   ```

3. **Configure the Dataset Python Script**
   When implementing `data/drivelm_dataset.py`, point `dataroot="data/raw/nuScenes"` and `drivelm_path="data/raw/DriveLM"` as default kwargs.