# UAV Dataset Fusion — AVSCA Perception Pipeline

Fuses four aerial computer vision datasets into a single **Ultralytics-compatible YOLO dataset** with 6 master classes, designed for the AVSCA (Autonomous Visual Surveillance and Classification for Airborne imagery) system running on an NVIDIA Jetson Orin Nano.

## Master Classes

| ID | Name |
|----|------|
| 0 | human |
| 1 | vehicle |
| 2 | building |
| 3 | wire |
| 4 | two-wheeler |
| 5 | utility-tower |

## Source Datasets

| Dataset | Source | Classes Mapped |
|---------|--------|----------------|
| [VisDrone](https://github.com/VisDrone/VisDrone-Dataset) | GitHub | pedestrian/people→0, car/van/truck/bus→1, bicycle/tricycle/motor→4 |
| [Heridal](https://universe.roboflow.com/licenta-ynwvo/heridal-lrbkc) | Roboflow | person→0 |
| [TTPLA](https://github.com/R3ab/ttpla_dataset) | GitHub | cable→3, tower_*→5 |
| [WiSARD](https://sites.google.com/uw.edu/wisard/) | UW | person→0, vehicle→1 |

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
  --visdrone_dir /path/to/VisDrone.zip \
  --heridal_dir  /path/to/Heridal.zip \
  --ttpla_dir    /path/to/TTPLA.zip \
  --wisard_dir   /path/to/WiSARD.zip \
  --output_dir   ./master_uav_dataset
```

Each `--*_dir` argument accepts either a directory path or a `.zip` archive — the script extracts zips automatically.

### Google Colab (GPU Training)

1. **Google Drive setup** — create these folders in your Drive:
   - `My Drive/UAV_Data/Raw/` — place `VisDrone.zip`, `Heridal.zip`, `TTPLA.zip`, `WiSARD.zip` here
   - `My Drive/UAV_Data/Ready/` — leave empty; the fused dataset zip lands here

2. **Push this repo to GitHub** (one-time):
   ```bash
   git remote add origin https://github.com/pratoshhhh/uav-dataset-fusion.git
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
names: [human, vehicle, building, wire, two-wheeler, utility-tower]
```

## CLI Reference

```
usage: data_fusion.py [-h] [--visdrone_dir VISDRONE_DIR]
                      [--heridal_dir HERIDAL_DIR]
                      [--ttpla_dir TTPLA_DIR]
                      [--wisard_dir WISARD_DIR]
                      [--output_dir OUTPUT_DIR]

optional arguments:
  --visdrone_dir    Path to VisDrone dataset folder or .zip archive
  --heridal_dir     Path to Heridal dataset folder or .zip archive
  --ttpla_dir       Path to TTPLA dataset folder or .zip archive
  --wisard_dir      Path to WiSARD dataset folder or .zip archive
  --output_dir      Output directory (default: /content/master_uav_dataset)
```

## Notes

- **WiSARD**: Distributed with a custom annotation format. Inspect your local files and confirm class integer IDs match the remap table in `src/data_fusion.py` before running. The script logs any unrecognized class IDs.
- **Edge cases**: Empty label files, missing images, and malformed lines are logged as warnings and skipped — they do not crash the pipeline.
- **Name collisions**: Output filenames are prefixed with the dataset name (e.g. `visdrone_image001.jpg`) to prevent overwrites when multiple datasets share identical filenames.
