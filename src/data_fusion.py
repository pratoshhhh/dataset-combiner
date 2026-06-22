"""
AVSCA UAV Dataset Fusion Pipeline
Merges VisDrone, Heridal, TTPLA, and WiSARD into a single
Ultralytics-compatible YOLO dataset with 6 master classes.

VisDrone annotation format (CSV, 1-indexed classes, absolute pixel coords)
is handled by a dedicated parser. All other datasets are expected to already
be in YOLO format (normalized cx cy w h).
"""

import argparse
import logging
import shutil
import zipfile
import tempfile
from pathlib import Path

import yaml
from PIL import Image
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Master class definitions
# ---------------------------------------------------------------------------
MASTER_CLASSES = {
    0: "human",
    1: "vehicle",
    2: "building",
    3: "wire",
    4: "two-wheeler",
    5: "utility-tower",
}

# ---------------------------------------------------------------------------
# VisDrone native class IDs (1-indexed in the annotation files)
#
# 0  = ignored region  (score field == 0 also flags ignored)
# 1  = pedestrian
# 2  = people
# 3  = bicycle
# 4  = car
# 5  = van
# 6  = truck
# 7  = tricycle
# 8  = awning-tricycle
# 9  = bus
# 10 = motor
# 11 = others
#
# We discard: 0 (ignored), 11 (others)
# ---------------------------------------------------------------------------
VISDRONE_REMAP: dict[int, int] = {
    1:  0,   # pedestrian      → human
    2:  0,   # people          → human
    4:  1,   # car             → vehicle
    5:  1,   # van             → vehicle
    6:  1,   # truck           → vehicle
    9:  1,   # bus             → vehicle
    3:  4,   # bicycle         → two-wheeler
    7:  4,   # tricycle        → two-wheeler
    8:  4,   # awning-tricycle → two-wheeler
    10: 4,   # motor           → two-wheeler
}

# ---------------------------------------------------------------------------
# YOLO-format dataset remapping tables (0-indexed native class IDs)
# ---------------------------------------------------------------------------
YOLO_REMAP: dict[str, dict[int, int]] = {
    "heridal": {
        0: 0,   # human/person → human
    },
    "ttpla": {
        0: 3,   # cable          → wire
        1: 5,   # tower_lattice  → utility-tower
        2: 5,   # tower_wooden   → utility-tower
        3: 5,   # tower_monopole → utility-tower
    },
    "wisard": {
        0: 0,   # human/person signature → human
        1: 1,   # vehicle signature      → vehicle
    },
}

IMAGE_EXTENSIONS = [".jpg", ".JPG", ".jpeg", ".JPEG", ".png", ".PNG"]


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def resolve_dataset_path(raw_path: str, scratch_root: Path, name: str) -> Path | None:
    """Return a resolved directory Path for the dataset.

    If raw_path ends in .zip it is extracted into scratch_root/name/ first.
    Returns None when the path is absent or does not exist.
    """
    if raw_path is None:
        return None

    p = Path(raw_path)
    if not p.exists():
        log.warning("Path does not exist, skipping dataset '%s': %s", name, p)
        return None

    if p.suffix.lower() == ".zip":
        dest = scratch_root / name
        dest.mkdir(parents=True, exist_ok=True)
        log.info("Extracting %s → %s", p.name, dest)
        with zipfile.ZipFile(p, "r") as zf:
            zf.extractall(dest)
        return dest

    return p


def find_image(label_path: Path) -> Path | None:
    """Locate the matching image for a label file.

    Checks the same directory first, then a sibling 'images/' folder
    (covers the VisDrone annotations/ → images/ layout and similar).
    """
    for ext in IMAGE_EXTENSIONS:
        candidate = label_path.with_suffix(ext)
        if candidate.exists():
            return candidate

    for ext in IMAGE_EXTENSIONS:
        candidate = label_path.parent.parent / "images" / (label_path.stem + ext)
        if candidate.exists():
            return candidate

    return None


def infer_split(path: Path) -> str:
    """Infer train/val split from the path hierarchy."""
    parts = [p.lower() for p in path.parts]
    if "val" in parts or "valid" in parts or "validation" in parts:
        return "val"
    if "test" in parts:
        # testset-dev has GT so we fold it into train
        return "train"
    return "train"


def write_sample(
    image_path: Path,
    remapped_lines: list[str],
    dataset_name: str,
    split: str,
    output_dir: Path,
    stats: dict,
) -> None:
    """Copy image and write remapped label file into the output directory."""
    unique_stem = f"{dataset_name}_{image_path.stem}"
    out_img_dir = output_dir / "images" / split
    out_lbl_dir = output_dir / "labels" / split
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_lbl_dir.mkdir(parents=True, exist_ok=True)

    out_img_path = out_img_dir / (unique_stem + image_path.suffix)
    out_lbl_path = out_lbl_dir / (unique_stem + ".txt")

    try:
        shutil.copy2(image_path, out_img_path)
        out_lbl_path.write_text("\n".join(remapped_lines) + "\n", encoding="utf-8")
        stats["copied"] += 1
    except OSError as exc:
        log.warning("Failed to write sample for %s: %s", image_path, exc)
        stats["errors"] += 1


# ---------------------------------------------------------------------------
# VisDrone-specific parser
# ---------------------------------------------------------------------------

def parse_visdrone_annotation(
    ann_path: Path,
    img_w: int,
    img_h: int,
) -> list[str] | None:
    """Convert a VisDrone CSV annotation file to YOLO-format lines.

    VisDrone format per line:
        bbox_left, bbox_top, bbox_width, bbox_height, score, object_category,
        truncation, occlusion

    score == 0 means the region is marked 'ignored' and must be skipped.
    object_category == 0 also means ignored.
    Coordinates are absolute pixels; we convert to normalised cx cy w h.
    """
    remapped: list[str] = []

    try:
        raw_text = ann_path.read_text(encoding="utf-8", errors="replace").strip()
    except OSError as exc:
        log.warning("Cannot read annotation %s: %s", ann_path, exc)
        return None

    if not raw_text:
        return None

    for line_no, line in enumerate(raw_text.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        fields = line.split(",")
        if len(fields) < 6:
            log.debug("Malformed VisDrone line %d in %s, skipping", line_no, ann_path)
            continue
        try:
            x1    = int(fields[0])
            y1    = int(fields[1])
            w_box = int(fields[2])
            h_box = int(fields[3])
            score = int(fields[4])
            cat   = int(fields[5])
        except ValueError:
            log.debug("Non-integer field at line %d in %s, skipping", line_no, ann_path)
            continue

        if score == 0 or cat == 0:
            continue  # ignored region

        master_cls = VISDRONE_REMAP.get(cat)
        if master_cls is None:
            continue  # unmapped class (e.g. 'others' = 11)

        # Clamp to image bounds
        x1    = max(0, x1)
        y1    = max(0, y1)
        w_box = min(w_box, img_w - x1)
        h_box = min(h_box, img_h - y1)

        if w_box <= 0 or h_box <= 0:
            continue

        cx = (x1 + w_box / 2) / img_w
        cy = (y1 + h_box / 2) / img_h
        nw = w_box / img_w
        nh = h_box / img_h

        remapped.append(f"{master_cls} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

    return remapped if remapped else None


def process_visdrone(dataset_dir: Path, output_dir: Path, stats: dict) -> None:
    """Process a VisDrone-DET split (train / val / testset-dev)."""
    ann_files = sorted(dataset_dir.rglob("*.txt"))

    if not ann_files:
        log.warning("No .txt annotation files found in VisDrone dataset at %s", dataset_dir)
        return

    log.info("Processing dataset 'visdrone' — %d annotation files found", len(ann_files))

    for ann_path in tqdm(ann_files, desc="visdrone", unit="file"):
        image_path = find_image(ann_path)
        if image_path is None:
            log.debug("No matching image for annotation %s, skipping", ann_path)
            stats["skipped_no_image"] += 1
            continue

        try:
            with Image.open(image_path) as img:
                img_w, img_h = img.size
        except Exception as exc:
            log.warning("Cannot read image dimensions for %s: %s", image_path, exc)
            stats["errors"] += 1
            continue

        remapped_lines = parse_visdrone_annotation(ann_path, img_w, img_h)
        if remapped_lines is None:
            stats["skipped_no_valid_class"] += 1
            continue

        split = infer_split(ann_path)
        write_sample(image_path, remapped_lines, "visdrone", split, output_dir, stats)


# ---------------------------------------------------------------------------
# Generic YOLO-format dataset processor
# ---------------------------------------------------------------------------

def remap_yolo_label(
    label_path: Path,
    remap_table: dict[int, int],
) -> list[str] | None:
    """Parse a YOLO .txt label file and return remapped lines.

    Returns None if the file produces no valid output lines.
    """
    remapped: list[str] = []

    try:
        raw_text = label_path.read_text(encoding="utf-8", errors="replace").strip()
    except OSError as exc:
        log.warning("Cannot read label file %s: %s", label_path, exc)
        return None

    if not raw_text:
        return None

    for line_no, line in enumerate(raw_text.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 5:
            log.debug("Malformed line %d in %s (too few fields), skipping", line_no, label_path)
            continue
        try:
            native_cls = int(parts[0])
        except ValueError:
            log.debug("Non-integer class at line %d in %s, skipping", line_no, label_path)
            continue

        master_cls = remap_table.get(native_cls)
        if master_cls is None:
            continue  # discard unmapped class

        coords = " ".join(parts[1:])
        remapped.append(f"{master_cls} {coords}")

    return remapped if remapped else None


def process_yolo_dataset(
    dataset_dir: Path,
    dataset_name: str,
    output_dir: Path,
    stats: dict,
) -> None:
    """Walk all YOLO label .txt files in dataset_dir, remap, and copy to output_dir."""
    remap_table = YOLO_REMAP[dataset_name]
    label_files = sorted(dataset_dir.rglob("*.txt"))

    if not label_files:
        log.warning("No .txt label files found in dataset '%s' at %s", dataset_name, dataset_dir)
        return

    log.info("Processing dataset '%s' — %d label files found", dataset_name, len(label_files))

    for label_path in tqdm(label_files, desc=dataset_name, unit="file"):
        remapped_lines = remap_yolo_label(label_path, remap_table)
        if remapped_lines is None:
            stats["skipped_no_valid_class"] += 1
            continue

        image_path = find_image(label_path)
        if image_path is None:
            log.debug("No matching image for label %s, skipping", label_path)
            stats["skipped_no_image"] += 1
            continue

        split = infer_split(label_path)
        write_sample(image_path, remapped_lines, dataset_name, split, output_dir, stats)


# ---------------------------------------------------------------------------
# data.yaml generation
# ---------------------------------------------------------------------------

def generate_data_yaml(output_dir: Path) -> None:
    """Write an Ultralytics-compatible data.yaml into output_dir."""
    data = {
        "path": str(output_dir),
        "train": "images/train",
        "val": "images/val",
        "nc": len(MASTER_CLASSES),
        "names": [MASTER_CLASSES[i] for i in sorted(MASTER_CLASSES)],
    }
    yaml_path = output_dir / "data.yaml"
    with yaml_path.open("w", encoding="utf-8") as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True)
    log.info("data.yaml written → %s", yaml_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="AVSCA UAV Dataset Fusion — merges VisDrone, Heridal, TTPLA, WiSARD "
                    "into a single Ultralytics YOLO dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--visdrone_dir", type=str, default=None,
                        help="Path to VisDrone-DET dataset folder or .zip archive. "
                             "Accepts a single zip or a folder containing multiple "
                             "split zips (trainset, valset, testset-dev).")
    parser.add_argument("--heridal_dir", type=str, default=None,
                        help="Path to Heridal dataset folder or .zip archive")
    parser.add_argument("--ttpla_dir", type=str, default=None,
                        help="Path to TTPLA dataset folder or .zip archive")
    parser.add_argument("--wisard_dir", type=str, default=None,
                        help="Path to WiSARD dataset folder or .zip archive")
    parser.add_argument("--output_dir", type=str, default="/content/master_uav_dataset",
                        help="Output directory for the fused dataset")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log.info("Output directory: %s", output_dir)

    yolo_dataset_inputs = {
        "heridal": args.heridal_dir,
        "ttpla":   args.ttpla_dir,
        "wisard":  args.wisard_dir,
    }

    if args.visdrone_dir is None and all(v is None for v in yolo_dataset_inputs.values()):
        log.error("No dataset paths provided. Pass at least one of "
                  "--visdrone_dir, --heridal_dir, --ttpla_dir, --wisard_dir.")
        raise SystemExit(1)

    stats = {
        "copied": 0,
        "skipped_no_valid_class": 0,
        "skipped_no_image": 0,
        "errors": 0,
    }

    with tempfile.TemporaryDirectory(prefix="avsca_scratch_") as scratch_str:
        scratch = Path(scratch_str)

        # --- VisDrone (custom CSV parser) ---
        if args.visdrone_dir is not None:
            visdrone_path = resolve_dataset_path(args.visdrone_dir, scratch, "visdrone")
            if visdrone_path is not None:
                process_visdrone(visdrone_path, output_dir, stats)

        # --- YOLO-format datasets ---
        for name, raw_path in yolo_dataset_inputs.items():
            if raw_path is None:
                log.info("Skipping dataset '%s' (no path provided)", name)
                continue

            dataset_dir = resolve_dataset_path(raw_path, scratch, name)
            if dataset_dir is None:
                continue

            process_yolo_dataset(dataset_dir, name, output_dir, stats)

    generate_data_yaml(output_dir)

    log.info(
        "Done. Copied: %d | Skipped (no valid class): %d | "
        "Skipped (no image): %d | Errors: %d",
        stats["copied"],
        stats["skipped_no_valid_class"],
        stats["skipped_no_image"],
        stats["errors"],
    )


if __name__ == "__main__":
    main()
