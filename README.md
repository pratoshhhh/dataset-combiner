# UAV Dataset Fusion — AVSCA Perception Pipeline

Fuses four aerial computer vision datasets into a single **Ultralytics-compatible YOLO dataset** with 6 master classes, designed for the AVSCA (Autonomous Visual Surveillance and Classification for Airborne imagery) system running on an NVIDIA Jetson Orin Nano.

## Master Classes

| ID | Name |
|----|------|
| 0 | human |
| 1 | vehicle |
| 2 | building |
| 3 | wire |
| 4 | utility-tower |
| 5 | tree |

## Source Datasets

| Dataset | Source | Annotation Format | Classes Mapped |
|---------|--------|-------------------|----------------|
| [VisDrone-DET](https://github.com/VisDrone/VisDrone-Dataset) | GitHub | Custom CSV, absolute pixels, 1-indexed | pedestrian/people→0, car/van/truck/bus→1, bicycle/tricycle/motor discarded |
| [Heridal](https://universe.roboflow.com/licenta-ynwvo/heridal-lrbkc) | Roboflow | YOLO format | person→0 |
| [TTPLA](https://github.com/R3ab/ttpla_dataset) | GitHub | COCO JSON, polygon segmentation | cable→3, tower_lattice/wooden/monopole/tucohy→4 |
| [Drone Buildings](https://universe.roboflow.com/buildingyolo/drone-buildings) | Roboflow | YOLO format | building/building2→2, statue discarded |
| [Tree-Top-View](https://universe.roboflow.com/pine-tree/tree-top-view-k71bz) | Roboflow / Zenodo | YOLO format | tree-top→5 |

## Project Structure

```
dataset-combiner/
├── src/
│   └── data_fusion.py       # Main CLI fusion script
├── requirements.txt
├── uav_colab_run.ipynb      # Google Colab execution notebook
└── README.md
```

## Quick Start

### Local Usage

```bash
pip install -r requirements.txt

python src/data_fusion.py \
  --visdrone_dir /path/to/VisDrone2019-DET-train.zip \
                 /path/to/VisDrone2019-DET-val.zip \
                 /path/to/VisDrone2019-DET-test-dev.zip \
  --heridal_dir  /path/to/HERIDAL.yolov8.zip \
  --ttpla_dir    /path/to/data_original_size_v1.zip \
  --building_dir /path/to/drone-buildings.yolov8.zip \
  --output_dir   ./master_uav_dataset
```

Each argument accepts either a directory path or a `.zip` archive — zips are extracted automatically to a temp directory and cleaned up after.

### Google Colab (GPU Training)

1. **Google Drive setup** — create these folders in your Drive:
   - `My Drive/UAV_Data/Raw/` — place the dataset zips here (see filenames below)
   - `My Drive/UAV_Data/Ready/` — leave empty; the fused dataset zip lands here

   Expected filenames in `Raw/`:
   ```
   VisDrone2019-DET-train.zip
   VisDrone2019-DET-val.zip
   VisDrone2019-DET-test-dev.zip
   HERIDAL.yolov8.zip
   data_original_size_v1.zip
   Drone Buildings.v1i.yolov8.zip
   tree-top-view.yolov8.zip
   ```

2. **Push this repo to GitHub** (one-time):
   ```bash
   git remote add origin https://github.com/pratoshhhh/dataset-combiner.git
   git push -u origin main
   ```

3. **Open `uav_colab_run.ipynb`** in Google Colab, connect to a GPU runtime, and run the three cells sequentially.

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

## CLI Reference

```
usage: data_fusion.py [-h] [--visdrone_dir [PATH ...]]
                      [--heridal_dir HERIDAL_DIR]
                      [--ttpla_dir TTPLA_DIR]
                      [--building_dir BUILDING_DIR]
                      [--tree_dir TREE_DIR]
                      [--output_dir OUTPUT_DIR]

optional arguments:
  --visdrone_dir    One or more VisDrone-DET zip/folder paths (train, val, test-dev)
  --heridal_dir     Path to Heridal dataset folder or .zip archive
  --ttpla_dir       Path to TTPLA dataset folder or .zip archive
  --building_dir    Path to Drone Buildings dataset folder or .zip archive
  --tree_dir        Path to Tree-Top-View dataset folder or .zip archive
  --output_dir      Output directory (default: /content/master_uav_dataset)
```

## Notes

- **VisDrone test-dev**: GT annotations are publicly available — the script folds `testset-dev` into `train` for extra data.
- **TTPLA**: Uses COCO JSON polygon segmentation internally; bounding boxes are derived automatically from the `bbox` field in each annotation.
- **Edge cases**: Empty label files, missing images, malformed lines, and unrecognized class IDs are logged as warnings and skipped — they do not crash the pipeline.
- **Name collisions**: Output filenames are prefixed with the dataset name (e.g. `visdrone_image001.jpg`) to prevent overwrites across datasets.
