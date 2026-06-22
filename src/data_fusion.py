"""
AVSCA UAV Dataset Fusion Pipeline
Merges VisDrone, Heridal, TTPLA, and WiSARD into a single
Ultralytics-compatible YOLO dataset with 6 master classes.
"""

import argparse
import logging
import shutil
import zipfile
import tempfile
from pathlib import Path

import yaml
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
# Per-dataset native-class → master-class remapping tables
# Keys are native integer class IDs; values are master class IDs.
# Any native class NOT listed here is silently discarded.
# ---------------------------------------------------------------------------
REMAP: dict[str, dict[int, int]] = {
    "visdrone": {
        0: 0,   # pedestrian  → human
        1: 0,   # people      → human
        3: 1,   # car         → vehicle
        4: 1,   # van         → vehicle
        5: 1,   # truck       → vehicle
        8: 1,   # bus         → vehicle
        2: 4,   # bicycle     → two-wheeler
        6: 4,   # tricycle    → two-wheeler
        7: 4,   # awning-tricycle → two-wheeler
        9: 4,   # motor       → two-wheeler
    },
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
# Helpers
# ---------------------------------------------------------------------------

def resolve_dataset_path(raw_path: str, scratch_root: Path, name: str) -> Path | None:
    """Return a directory Path for the dataset.

    If raw_path points to a .zip file it is extracted into scratch_root/name/.
    Returns None if the path is None or does not exist.
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
    """Locate the matching image for a given label file."""
    for ext in IMAGE_EXTENSIONS:
        candidate = label_path.with_suffix(ext)
        if candidate.exists():
            return candidate

    # Some datasets store images in a sibling 'images' folder
    for ext in IMAGE_EXTENSIONS:
        candidate = label_path.parent.parent / "images" / (label_path.stem + ext)
        if candidate.exists():
            return candidate

    return None


def infer_split(label_path: Path) -> str:
    """Infer train/val split from the path hierarchy."""
    parts = [p.lower() for p in label_path.parts]
    if "val" in parts or "valid" in parts or "validation" in parts:
        return "val"
    return "train"


def remap_label_file(
    label_path: Path,
    remap_table: dict[int, int],
) -> list[str] | None:
    """Parse a YOLO .txt label file and return remapped lines.

    Returns None if the file produces no valid output lines.
    Returns an empty-list sentinel (treated as skip) on parse errors.
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


def process_dataset(
    dataset_dir: Path,
    dataset_name: str,
    output_dir: Path,
    stats: dict,
) -> None:
    """Walk all label .txt files in dataset_dir, remap, and copy to output_dir."""
    remap_table = REMAP[dataset_name]
    label_files = sorted(dataset_dir.rglob("*.txt"))

    if not label_files:
        log.warning("No .txt label files found in dataset '%s' at %s", dataset_name, dataset_dir)
        return

    log.info("Processing dataset '%s' — %d label files found", dataset_name, len(label_files))

    for label_path in tqdm(label_files, desc=dataset_name, unit="file"):
        remapped_lines = remap_label_file(label_path, remap_table)
        if remapped_lines is None:
            stats["skipped_no_valid_class"] += 1
            continue

        image_path = find_image(label_path)
        if image_path is None:
            log.debug("No matching image for label %s, skipping", label_path)
            stats["skipped_no_image"] += 1
            continue

        split = infer_split(label_path)

        # Build a unique output stem to avoid name collisions across datasets
        unique_stem = f"{dataset_name}_{label_path.stem}"
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
            log.warning("Failed to copy %s: %s", image_path, exc)
            stats["errors"] += 1


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
                        help="Path to VisDrone dataset folder or .zip archive")
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

    dataset_inputs = {
        "visdrone": args.visdrone_dir,
        "heridal":  args.heridal_dir,
        "ttpla":    args.ttpla_dir,
        "wisard":   args.wisard_dir,
    }

    if all(v is None for v in dataset_inputs.values()):
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

        for name, raw_path in dataset_inputs.items():
            if raw_path is None:
                log.info("Skipping dataset '%s' (no path provided)", name)
                continue

            dataset_dir = resolve_dataset_path(raw_path, scratch, name)
            if dataset_dir is None:
                continue

            process_dataset(dataset_dir, name, output_dir, stats)

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
