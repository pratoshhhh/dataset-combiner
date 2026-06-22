# UAV Dataset Fusion — AVSCA Perception Pipeline

Fuses five aerial/UAV datasets into a single **Ultralytics-compatible YOLO dataset** with 6 master classes, designed for the AVSCA (Autonomous Visual Surveillance and Classification for Airborne imagery) system running on an NVIDIA Jetson Orin Nano.

---

## Master Classes

| ID | Name | Primary Source |
|----|------|----------------|
| 0 | human | VisDrone, Heridal |
| 1 | vehicle | VisDrone |
| 2 | building | Drone Buildings |
| 3 | wire | TTPLA |
| 4 | utility-tower | TTPLA |
| 5 | tree | yolov8tree |

---

## Source Datasets

| Dataset | Source | Annotation Format | Perspective | Classes Mapped |
|---------|--------|-------------------|-------------|----------------|
| [VisDrone-DET](https://github.com/VisDrone/VisDrone-Dataset) | GitHub | Custom CSV, absolute pixels, 1-indexed | Overhead | pedestrian/people→0, car/van/truck/bus→1 |
| [Heridal](https://universe.roboflow.com/licenta-ynwvo/heridal-lrbkc) | Roboflow | YOLO format | Overhead | person→0 |
| [TTPLA](https://github.com/R3ab/ttpla_dataset) | GitHub | COCO JSON, polygon segmentation | Overhead | cable→3, tower_lattice/wooden/monopole/tucohy→4 |
| [Drone Buildings](https://universe.roboflow.com/buildingyolo/drone-buildings) | Roboflow | YOLO format | Overhead/oblique | building/building2→2 |
| [yolov8tree](https://universe.roboflow.com/trees-sam/yolov8tree/dataset/2) | Roboflow | YOLO format | Overhead/oblique | tree→5 |

---

## Project Structure

```
dataset-combiner/
├── src/
│   └── data_fusion.py       # Main CLI fusion script
├── requirements.txt
├── uav_colab_run.ipynb      # Google Colab execution notebook
└── README.md
```

---

## Quick Start

### Local Usage

```bash
pip install -r requirements.txt

python src/data_fusion.py \
  --visdrone_dir VisDrone2019-DET-train.zip \
                 VisDrone2019-DET-val.zip \
                 VisDrone2019-DET-test-dev.zip \
  --heridal_dir  HERIDAL.yolov8.zip \
  --ttpla_dir    data_original_size_v1.zip \
  --building_dir "Drone Buildings.v1i.yolov8.zip" \
  --tree_dir     yolov8tree.v2i.yolov8.zip \
  --output_dir   ./master_uav_dataset
```

Each argument accepts either a directory path or a `.zip` archive — zips are extracted automatically to a temp directory and cleaned up after.

### Google Colab

1. **Google Drive setup** — create these two folders in your Drive:
   - `My Drive/UAV_Data/Raw/` — place all dataset zips here
   - `My Drive/UAV_Data/Ready/` — leave empty; the fused dataset zip lands here

2. Place these files in `Raw/` (exact filenames matter):
   ```
   VisDrone2019-DET-train.zip
   VisDrone2019-DET-val.zip
   VisDrone2019-DET-test-dev.zip
   HERIDAL.yolov8.zip
   data_original_size_v1.zip
   Drone Buildings.v1i.yolov8.zip
   yolov8tree.v2i.yolov8.zip
   ```

3. Open `uav_colab_run.ipynb` in Google Colab, connect to a GPU runtime, and run the three cells sequentially.

---

## Output Structure

```
master_uav_dataset/
├── images/
│   ├── train/
│   └── val/
├── labels/
│   ├── train/
│   └── val/
└── data.yaml
```

The generated `data.yaml` is directly consumable by Ultralytics YOLO:

```yaml
path: /content/master_uav_dataset
train: images/train
val:   images/val
nc: 6
names: [human, vehicle, building, wire, utility-tower, tree]
```

---

## CLI Reference

```
usage: data_fusion.py [-h] [--visdrone_dir [PATH ...]]
                      [--heridal_dir HERIDAL_DIR]
                      [--ttpla_dir TTPLA_DIR]
                      [--building_dir BUILDING_DIR]
                      [--tree_dir TREE_DIR]
                      [--output_dir OUTPUT_DIR]

optional arguments:
  --visdrone_dir      One or more VisDrone-DET zip/folder paths (train, val, test-dev)
  --heridal_dir       Path to Heridal dataset folder or .zip archive
  --ttpla_dir         Path to TTPLA dataset folder or .zip archive (COCO JSON)
  --building_dir      Path to Drone Buildings dataset folder or .zip archive
  --tree_dir          Path to yolov8tree dataset folder or .zip archive
  --output_dir        Output directory (default: /content/master_uav_dataset)
```

---

## Notes

- **VisDrone test-dev**: GT annotations are publicly available — the script automatically folds `testset-dev` into `train` for additional data.
- **TTPLA**: Uses COCO JSON polygon segmentation; bounding boxes are derived from the `bbox` field in each annotation. Category mapping is done by name (not integer ID) for robustness.
- **Edge cases**: Empty label files, missing images, malformed lines, and unrecognized class IDs are logged as warnings and skipped without crashing.
- **Name collisions**: Output filenames are prefixed with the dataset name (e.g. `visdrone_image001.jpg`) to prevent overwrites across datasets.
