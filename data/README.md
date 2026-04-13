# DriveLM Dataset Setup

The dataset consists of two parts: DriveLM text/annotations and nuScenes images. The text and annotations are handled automatically via the Hugging Face `datasets` library. We only need to manually download the images from nuScenes.

Here is the breakdown of the setup.

### 1. DriveLM (The Text / Q&A)
The Hugging Face `datasets` library handles this automatically. When the code calls `load_dataset("OpenDriveLab/DriveLM")` in `drivelm_dataset.py`, it downloads the JSON graphs and text questions directly into memory. 

### 2. nuScenes (The Images)
DriveLM uses the original nuScenes dataset images, which must be downloaded directly from the creators of nuScenes.

**How to download nuScenes:**
1. Go to **[nuscenes.org/download](https://www.nuscenes.org/download)**.
2. Create a free account and agree to their academic terms of use.
3. Scroll down to the **"Full dataset (v1.0)"** section.
4. Download the ZIP file for the **v1.0-mini** dataset. (Recommend starting with the "Mini" split, which is about 4 GB, before scaling to the full dataset).

### 3. Folder Setup
Once you have the `v1.0-mini.zip` file:

1. Unzip the file. Inside, you will see several folders, including the `samples/` folder which contains the 6 surround-camera images (e.g., `CAM_FRONT`, `CAM_BACK`).
2. Navigate to your shared Google Drive or local workspace.
3. Create a new folder at `data/raw/nuscenes_mini`.
4. Run or manually move the unzipped folders (specifically the `samples/` folder) into the newly created folder.

**The structure should look like this:**
```text
data/
└── raw/
    └── nuscenes_mini/
        ├── samples/
        │   ├── CAM_FRONT/
        │   │   ├── n015-2018-07-24-11-22-45+0800__CAM_FRONT__1532402927612460.jpg
        │   │   └── ...
        │   ├── CAM_FRONT_RIGHT/
        │   └── ...
```

### 4. Connecting it to the Code
When executing the pipeline, mount your Google Drive (if in Colab) or configure the local relative path:

```python
NUSCENES_IMAGE_DIR = "data/raw/nuscenes_mini"
```