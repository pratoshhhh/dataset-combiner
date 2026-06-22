# AVSCA Training Pipeline — Cursor Agent Prompt

> Paste everything below this line into a new Cursor Composer session (Ctrl+I) in an empty directory.

---

You are an expert ML engineer and embedded systems developer specializing in
real-time computer vision on NVIDIA edge hardware. I am building the training
pipeline for AVSCA (Autonomous Visual Surveillance and Classification for
Airborne imagery), a UAV perception system that runs on an NVIDIA Jetson Orin
Nano for low-altitude obstacle avoidance and aerial Search and Rescue (SAR).

I have already built a dataset fusion pipeline that produces a unified
Ultralytics-compatible YOLO dataset. I now need you to build the complete
YOLOv8 training, evaluation, and export pipeline as a clean GitHub-ready
project.

---

## HARDWARE TARGET

- Inference device: NVIDIA Jetson Orin Nano (8 GB)
- CUDA compute: Ampere architecture
- Runtime: TensorRT 8.x via Ultralytics export
- Target inference speed: ≥30 FPS at 640×640 on Jetson
- Deployment format: TensorRT engine (.engine) via ONNX intermediate

---

## DATASET SPECIFICATION

The fused dataset produced by the fusion pipeline has this exact structure:

```
master_uav_dataset/
├── images/
│   ├── train/      ← JPEG/PNG images
│   └── val/
├── labels/
│   ├── train/      ← YOLO .txt files (class cx cy w h, normalised)
│   └── val/
└── data.yaml
```

`data.yaml` content:

```yaml
path: /content/master_uav_dataset
train: images/train
val:   images/val
nc: 6
names: [human, vehicle, building, wire, two-wheeler, utility-tower]
```

Class notes:

- **Class 0 (human):** SAR primary target — prioritise recall over precision
- **Class 3 (wire):** Thin elongated objects, extremely safety-critical for obstacle avoidance — requires high sensitivity
- **Class 5 (utility-tower):** Large structures, easier to detect
- Dataset is heavily imbalanced: human and vehicle dominate; wire and building are minority classes

---

## REQUIRED FILE STRUCTURE

Generate the following files directly in the workspace:

### File 1: `train.py`

A production-grade training script using Ultralytics YOLOv8.

- Use `ultralytics` Python API (not CLI)
- Default model: `yolov8s.pt` (small — balanced speed/accuracy for Jetson)
- Accept `--data`, `--model`, `--epochs`, `--imgsz`, `--batch`, `--project`, `--name` via argparse
- Default values:
  - `--data`: `/content/master_uav_dataset/data.yaml`
  - `--model`: `yolov8s.pt`
  - `--epochs`: `100`
  - `--imgsz`: `640`
  - `--batch`: `16`
  - `--project`: `runs/train`
  - `--name`: `avsca_v1`
- Training hyperparameters to set explicitly:
  - `optimizer`: AdamW
  - `lr0`: 0.001
  - `lrf`: 0.01
  - `warmup_epochs`: 5
  - `mosaic`: 1.0
  - `mixup`: 0.1
  - `copy_paste`: 0.1 — helps minority classes (wire, building) by synthetically pasting them into more scenes
  - `degrees`: 15.0 — UAV rotates; rotational augmentation is critical for aerial imagery
  - `flipud`: 0.5 — aerial top-down view makes vertical flip geometrically valid
  - `hsv_h`: 0.015
  - `hsv_s`: 0.7
  - `hsv_v`: 0.4
  - `patience`: 20 — early stopping
  - `save_period`: 10
  - `cos_lr`: True
- After training completes, automatically run validation on `best.pt` and print per-class AP50 and AP50-95
- Log training to a `training_log.json` in the output directory containing: model name, dataset path, epochs run, final mAP50, final mAP50-95, per-class AP50, training duration in minutes

### File 2: `export.py`

A Jetson-optimised export script.

- Accept `--weights` (path to best.pt), `--imgsz`, `--format` via argparse
- Default format: `engine` (TensorRT)
- Export pipeline:
  1. Export to ONNX first (opset 12, dynamic=False, simplify=True)
  2. Then export to TensorRT engine (FP16 precision for Jetson Orin Nano)
- Print the output path and file size of both exports
- Include a fallback: if TensorRT export fails (not on Jetson), export to ONNX only and warn the user
- Also support `--format onnx` to export ONNX only (for testing on non-Jetson machines)

### File 3: `validate.py`

A standalone validation script that can be run independently of training.

- Accept `--weights`, `--data`, `--imgsz`, `--conf`, `--iou` via argparse
- Run Ultralytics validation
- Print a formatted per-class results table showing: precision, recall, mAP50, mAP50-95 for each of the 6 classes
- Flag classes with mAP50 < 0.40 with a visible `WARNING` in the output (underperforming classes that need more data or tuning)
- Save results to `validation_results.json`

### File 4: `requirements.txt`

```
ultralytics>=8.0.0
torch>=2.0.0
torchvision>=0.15.0
opencv-python>=4.7.0
pyyaml>=6.0
tqdm>=4.64.0
onnx>=1.14.0
onnxsim>=0.4.33
```

### File 5: `colab_train.ipynb`

A complete Google Colab notebook with GPU runtime, containing these cells:

**Cell 1 — Mount Drive and set paths (Python):**
```python
from google.colab import drive
drive.mount('/content/drive')

DATASET_ZIP = '/content/drive/MyDrive/UAV_Data/Ready/master_uav_dataset.zip'
DATASET_DIR = '/content/master_uav_dataset'
REPO_DIR    = '/content/avsca-training'
OUTPUT_DIR  = '/content/drive/MyDrive/UAV_Data/Models'
```

**Cell 2 — Clone repo, install dependencies, extract dataset (bash):**
```bash
git clone https://github.com/pratoshhhh/avsca-training $REPO_DIR
pip install -r $REPO_DIR/requirements.txt
unzip -q $DATASET_ZIP -d /content/
```

**Cell 3 — Run training (bash):**
```bash
python $REPO_DIR/train.py \
  --data  $DATASET_DIR/data.yaml \
  --model yolov8s.pt \
  --epochs 100 \
  --imgsz 640 \
  --batch 16
```

**Cell 4 — Run validation on best weights (bash):**
```bash
python $REPO_DIR/validate.py \
  --weights $REPO_DIR/runs/train/avsca_v1/weights/best.pt \
  --data    $DATASET_DIR/data.yaml
```

**Cell 5 — Export to ONNX (Colab cannot export TensorRT; do that on Jetson) (bash):**
```bash
python $REPO_DIR/export.py \
  --weights $REPO_DIR/runs/train/avsca_v1/weights/best.pt \
  --format onnx
```

**Cell 6 — Copy model outputs back to Drive (bash):**
```bash
cp -r $REPO_DIR/runs/train/avsca_v1/ $OUTPUT_DIR/avsca_v1/
echo "Model saved to Google Drive."
```

### File 6: `README.md`

Full documentation covering:

- Project overview and hardware target
- Dataset requirements (link to dataset-combiner repo: https://github.com/pratoshhhh/dataset-combiner)
- Installation
- Training instructions (local and Colab)
- Validation instructions
- Export workflow: Colab (ONNX) → transfer to Jetson → TensorRT engine
- Jetson deployment notes: how to run TensorRT engine with Ultralytics on Jetson Orin Nano

---

## IMPORTANT CONSTRAINTS

- Do NOT hardcode absolute paths anywhere except as argparse defaults that the user can override
- Do NOT add any inference/deployment code to this repo — that belongs in a separate repo
- The `.gitignore` must block: `runs/`, `*.pt`, `*.onnx`, `*.engine`, `*.pth`, `__pycache__/`, `.venv/`, `master_uav_dataset/`
- Wire (class 3) is the most safety-critical class — add a comment wherever class-specific logic could affect it
- All scripts must handle a missing GPU gracefully (fall back to CPU with a warning, do not crash)

Go ahead and construct these files directly in the workspace.
