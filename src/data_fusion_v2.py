"""
AVSCA UAV Dataset Fusion Pipeline — v2
=======================================
Builds a balanced 4-class YOLO dataset from the datav2/ folder.

Master classes
--------------
  0 → human     (VisDrone pedestrian + people)
  1 → vehicle   (VisDrone car + van + truck + bus)
  2 → building  (Roboflow "Building detection" zip — class-name matched)
  3 → tree      (Roboflow "archive" zip — class-name matched)

Design
------
  Two-pass approach:
    1. Collect ALL candidate samples in memory (no disk writes).
    2. Compute cap = min(2999, smallest_class_count).
    3. Proportionally sample `cap` images per class (random.seed(seed)).
    4. Write the union of all sampled images to disk.

  An image that contains multiple master classes is added to each matching
  candidate pool and, when selected, is written once with ALL its annotations.
"""

from __future__ import annotations

import argparse
import logging
import random
import sys
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path

import yaml
from PIL import Image
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Re-use shared helpers from data_fusion.py (same src/ package)
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent))
from data_fusion import (  # noqa: E402
    IMAGE_EXTENSIONS,
    find_image,
    infer_split,
    parse_visdrone_annotation,
    write_sample,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MASTER_CLASSES: dict[int, str] = {
    0: "human",
    1: "vehicle",
    2: "building",
    3: "tree",
}

# Filenames that must be present inside --datav2_dir
REQUIRED_ZIPS: dict[str, str] = {
    "visdrone_train": "VisDrone2019-DET-train.zip",
    "visdrone_val":   "VisDrone2019-DET-val.zip",
    "visdrone_test":  "VisDrone2019-DET-test-dev.zip",
    "building":       "Building detection.v1i.yolov8.zip",
    "tree":           "archive.zip",
}

# Maximum images per class (hard ceiling)
MAX_PER_CLASS = 2999

# .txt files that are NOT YOLO label files in Roboflow exports
_NON_LABEL_NAMES = frozenset(
    {
        "readme.txt",
        "notes.txt",
        "classes.txt",
        "predefined_classes.txt",
        "_darknet.labels",
    }
)


# ---------------------------------------------------------------------------
# Extended image finder (superset of data_fusion.find_image)
# ---------------------------------------------------------------------------
def _find_image(label_path: Path) -> Path | None:
    """Locate the image matching a label file.

    Search order (extends the original helper with an extra location):
    1. Same directory as the label file.
    2. Sibling ``images/`` folder  (VisDrone annotations/ layout).
    3. Direct parent of the label's directory — handles the ``archive.zip``
       flat layout: ``final tree/<image>.jpg`` + ``final tree/labels/<image>.txt``.
    """
    for ext in IMAGE_EXTENSIONS:
        candidate = label_path.with_suffix(ext)
        if candidate.exists():
            return candidate

    for ext in IMAGE_EXTENSIONS:
        candidate = label_path.parent.parent / "images" / (label_path.stem + ext)
        if candidate.exists():
            return candidate

    for ext in IMAGE_EXTENSIONS:
        candidate = label_path.parent.parent / (label_path.stem + ext)
        if candidate.exists():
            return candidate

    return None


# ---------------------------------------------------------------------------
# Sample container
# ---------------------------------------------------------------------------
@dataclass
class Sample:
    """A single fully-remapped training sample, ready to write."""

    image_path: Path        # absolute path in temp extraction dir
    label_lines: list[str]  # remapped YOLO lines  (class cx cy w h)
    split: str              # "train" | "val"
    dataset_name: str       # used as filename prefix to avoid collisions


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------
def _extract_zip(zip_path: Path, dest: Path) -> None:
    log.info("Extracting %s → %s", zip_path.name, dest)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dest)


def _classes_in_lines(label_lines: list[str]) -> set[int]:
    """Return the set of master class IDs present in remapped YOLO lines."""
    classes: set[int] = set()
    for line in label_lines:
        parts = line.split()
        if parts:
            try:
                classes.add(int(parts[0]))
            except ValueError:
                pass
    return classes


def _find_label_files(root: Path) -> list[Path]:
    """
    Find YOLO .txt label files in a Roboflow export tree.

    Prefers files inside any directory named 'labels'.
    Falls back to all .txt files while excluding known non-label names.
    """
    in_labels_dir = [
        p for p in sorted(root.rglob("*.txt"))
        if "labels" in {part.lower() for part in p.parts}
    ]
    if in_labels_dir:
        return in_labels_dir

    return [
        p for p in sorted(root.rglob("*.txt"))
        if p.name.lower() not in _NON_LABEL_NAMES
    ]


# ---------------------------------------------------------------------------
# Roboflow data.yaml remap builder
# ---------------------------------------------------------------------------
def build_roboflow_remap(
    extracted_dir: Path,
    keyword: str,
    master_cls: int,
) -> dict[int, int]:
    """
    Read ``data.yaml`` from a Roboflow YOLOv8 export and return a remap dict
    ``{native_class_idx: master_cls}`` for every class whose name contains
    *keyword* (case-insensitive substring match).

    If no ``data.yaml`` is found the archive is assumed to be a single-class
    dataset whose sole class (index 0) matches *keyword*, and the fallback
    remap ``{0: master_cls}`` is returned with a warning.
    """
    yaml_files = list(extracted_dir.rglob("data.yaml"))
    if not yaml_files:
        log.warning(
            "No data.yaml found in %s — assuming single class 0 → master %d (%s).",
            extracted_dir, master_cls, MASTER_CLASSES[master_cls],
        )
        return {0: master_cls}

    yaml_path = yaml_files[0]
    log.info("Reading class list from %s", yaml_path)

    with yaml_path.open(encoding="utf-8") as f:
        meta = yaml.safe_load(f)

    names = meta.get("names", [])
    remap: dict[int, int] = {}
    for i, name in enumerate(names):
        if keyword.lower() in str(name).lower():
            remap[i] = master_cls
            log.info(
                "  Native class %d '%s' → master %d (%s)",
                i, name, master_cls, MASTER_CLASSES[master_cls],
            )

    if not remap:
        log.warning(
            "No classes matching keyword '%s' found in %s  (classes: %s)",
            keyword, yaml_path, names,
        )

    return remap


# ---------------------------------------------------------------------------
# Remap a single YOLO .txt label file
# ---------------------------------------------------------------------------
def _remap_yolo_file(
    label_path: Path,
    remap: dict[int, int],
) -> list[str] | None:
    """
    Parse a YOLO .txt label file and return remapped lines.

    Returns None if the file is empty or yields no valid mapped annotations.
    """
    try:
        raw = label_path.read_text(encoding="utf-8", errors="replace").strip()
    except OSError as exc:
        log.warning("Cannot read label file %s: %s", label_path, exc)
        return None

    if not raw:
        return None

    remapped: list[str] = []
    for lineno, line in enumerate(raw.splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 5:
            log.debug("Malformed line %d in %s (too few fields)", lineno, label_path)
            continue
        try:
            native_cls = int(parts[0])
        except ValueError:
            log.debug("Non-integer class at line %d in %s", lineno, label_path)
            continue
        master_cls = remap.get(native_cls)
        if master_cls is None:
            continue
        remapped.append(f"{master_cls} {' '.join(parts[1:])}")

    return remapped if remapped else None


# ---------------------------------------------------------------------------
# First-pass collection functions (in-memory, no output I/O)
# ---------------------------------------------------------------------------
def collect_visdrone(
    extracted_dir: Path,
    forced_split: str,
    candidates: dict[int, list[Sample]],
) -> int:
    """
    Scan a VisDrone-DET extraction and populate *candidates*.

    The split is forced from the caller (train zip → "train",
    val zip → "val", test-dev zip → "train").
    """
    ann_files = sorted(extracted_dir.rglob("*.txt"))
    if not ann_files:
        log.warning("No .txt annotation files found at %s", extracted_dir)
        return 0

    log.info(
        "Scanning VisDrone (%s split) — %d annotation files",
        forced_split, len(ann_files),
    )

    collected = 0
    for ann_path in tqdm(ann_files, desc=f"visdrone-{forced_split}", unit="file"):
        image_path = _find_image(ann_path)
        if image_path is None:
            continue

        try:
            with Image.open(image_path) as img:
                img_w, img_h = img.size
        except Exception as exc:
            log.debug("Cannot open image %s: %s", image_path, exc)
            continue

        label_lines = parse_visdrone_annotation(ann_path, img_w, img_h)
        if not label_lines:
            continue

        sample = Sample(
            image_path=image_path,
            label_lines=label_lines,
            split=forced_split,
            dataset_name="visdrone",
        )
        for cls in _classes_in_lines(label_lines):
            if cls in candidates:
                candidates[cls].append(sample)
        collected += 1

    log.info("  → %d valid samples collected (forced split=%s)", collected, forced_split)
    return collected


def collect_roboflow_yolo(
    extracted_dir: Path,
    dataset_name: str,
    remap: dict[int, int],
    candidates: dict[int, list[Sample]],
) -> int:
    """
    Scan a Roboflow YOLOv8 export and populate *candidates*.

    Split is inferred from the path ("val"/"valid" in any path component → val).
    """
    label_files = _find_label_files(extracted_dir)
    log.info(
        "Scanning Roboflow '%s' — %d label files",
        dataset_name, len(label_files),
    )

    collected = 0
    for label_path in tqdm(label_files, desc=dataset_name, unit="file"):
        label_lines = _remap_yolo_file(label_path, remap)
        if not label_lines:
            continue

        image_path = _find_image(label_path)
        if image_path is None:
            continue

        split = infer_split(label_path)
        sample = Sample(
            image_path=image_path,
            label_lines=label_lines,
            split=split,
            dataset_name=dataset_name,
        )
        for cls in _classes_in_lines(label_lines):
            if cls in candidates:
                candidates[cls].append(sample)
        collected += 1

    log.info("  → %d valid samples collected", collected)
    return collected


# ---------------------------------------------------------------------------
# Balanced sampling
# ---------------------------------------------------------------------------
def sample_balanced(
    candidates: dict[int, list[Sample]],
    seed: int,
) -> tuple[dict[int, list[Sample]], int]:
    """
    Sample exactly *cap* images per master class.

    cap = min(MAX_PER_CLASS, smallest_class_candidate_count)

    Within each class the train/val proportions of the candidate pool are
    preserved (proportional allocation, train gets the rounding remainder).

    Returns (sampled_dict, cap).
    """
    raw_counts = {cls: len(v) for cls, v in candidates.items()}
    log.info(
        "Candidate counts — %s",
        "  ".join(
            f"{MASTER_CLASSES[c]}:{n}" for c, n in raw_counts.items()
        ),
    )

    cap = min(MAX_PER_CLASS, min(raw_counts.values()))
    log.info(
        "cap = min(%d, %d) = %d",
        MAX_PER_CLASS, min(raw_counts.values()), cap,
    )

    rng = random.Random(seed)
    sampled: dict[int, list[Sample]] = {}

    for cls in sorted(MASTER_CLASSES):
        pool = candidates[cls]
        train_pool = [s for s in pool if s.split == "train"]
        val_pool   = [s for s in pool if s.split == "val"]
        total      = len(train_pool) + len(val_pool)

        if total == 0:
            sampled[cls] = []
            continue

        # Proportional allocation — train gets the rounding surplus
        train_n = round(cap * len(train_pool) / total)
        val_n   = cap - train_n

        # Clamp to available
        train_n = min(train_n, len(train_pool))
        val_n   = min(val_n,   len(val_pool))

        # If one pool was short, try to compensate from the other
        shortfall = cap - train_n - val_n
        if shortfall > 0:
            extra = min(shortfall, len(train_pool) - train_n)
            train_n += extra
            shortfall -= extra
        if shortfall > 0:
            extra = min(shortfall, len(val_pool) - val_n)
            val_n += extra

        s_train = rng.sample(train_pool, train_n)
        s_val   = rng.sample(val_pool,   val_n)
        sampled[cls] = s_train + s_val

        log.info(
            "  Class %d (%s): %d train + %d val = %d",
            cls, MASTER_CLASSES[cls], train_n, val_n, train_n + val_n,
        )

    return sampled, cap


# ---------------------------------------------------------------------------
# Second-pass write
# ---------------------------------------------------------------------------
def write_dataset(
    sampled: dict[int, list[Sample]],
    output_dir: Path,
) -> dict[int, dict[str, int]]:
    """
    Write all uniquely sampled images to disk.

    Deduplication: if the same image was sampled for multiple classes it is
    written exactly once (with ALL its annotations preserved).

    Returns per-class train/val counts (from the sampled lists, not file counts).
    """
    write_queue: dict[Path, Sample] = {}
    for cls_samples in sampled.values():
        for sample in cls_samples:
            write_queue.setdefault(sample.image_path, sample)

    stats = {"copied": 0, "errors": 0}
    log.info("Writing %d unique images to %s …", len(write_queue), output_dir)

    for sample in tqdm(write_queue.values(), desc="writing", unit="img"):
        write_sample(
            sample.image_path,
            sample.label_lines,
            sample.dataset_name,
            sample.split,
            output_dir,
            stats,
        )

    log.info(
        "Disk write complete — copied: %d  errors: %d",
        stats["copied"], stats["errors"],
    )

    # Per-class split counts (reflects the sampling, not the deduplicated writes)
    counts: dict[int, dict[str, int]] = {
        cls: {
            "train": sum(1 for s in sampled[cls] if s.split == "train"),
            "val":   sum(1 for s in sampled[cls] if s.split == "val"),
        }
        for cls in range(len(MASTER_CLASSES))
    }
    return counts


# ---------------------------------------------------------------------------
# data.yaml
# ---------------------------------------------------------------------------
def generate_data_yaml(output_dir: Path) -> None:
    """Write an Ultralytics-compatible data.yaml to *output_dir*."""
    data = {
        "path":  f"./{output_dir.name}",
        "train": "images/train",
        "val":   "images/val",
        "nc":    len(MASTER_CLASSES),
        "names": [MASTER_CLASSES[i] for i in sorted(MASTER_CLASSES)],
    }
    yaml_path = output_dir / "data.yaml"
    with yaml_path.open("w", encoding="utf-8") as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
    log.info("data.yaml written → %s", yaml_path)


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------
def print_summary(counts: dict[int, dict[str, int]], cap: int) -> None:
    col_w = 12
    sep = "-" * (col_w + 2 + 7 + 2 + 7 + 2 + 7)
    print(f"\n{'Class':<{col_w}}  {'Train':>7}  {'Val':>7}  {'Total':>7}")
    print(sep)

    total_train = total_val = 0
    for cls in sorted(MASTER_CLASSES):
        t = counts[cls]["train"]
        v = counts[cls]["val"]
        total_train += t
        total_val   += v
        print(f"{MASTER_CLASSES[cls]:<{col_w}}  {t:>7}  {v:>7}  {t + v:>7}")

    print(sep)
    grand = total_train + total_val
    print(f"{'TOTAL':<{col_w}}  {total_train:>7}  {total_val:>7}  {grand:>7}\n")
    log.info("Cap applied: %d  |  Grand total images in output: %d", cap, grand)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "AVSCA UAV Dataset Fusion v2 — balanced 4-class YOLO dataset "
            "(human, vehicle, building, tree) from datav2/."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--datav2_dir",
        type=str,
        default="datav2",
        help="Path to the datav2/ folder containing the source zip files",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="master_uav_dataset2",
        help="Destination directory for the fused YOLO dataset",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible class-balanced sampling",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    datav2_dir = Path(args.datav2_dir)
    output_dir = Path(args.output_dir)

    # ------------------------------------------------------------------
    # Validate that all required zips exist before touching any output
    # ------------------------------------------------------------------
    zip_paths: dict[str, Path] = {}
    missing: list[str] = []
    for key, filename in REQUIRED_ZIPS.items():
        p = datav2_dir / filename
        if p.exists():
            zip_paths[key] = p
        else:
            missing.append(str(p))

    if missing:
        log.error(
            "Missing required input zip(s):\n  %s\n"
            "Ensure all files are inside --datav2_dir (%s).",
            "\n  ".join(missing),
            datav2_dir,
        )
        raise SystemExit(1)

    output_dir.mkdir(parents=True, exist_ok=True)
    log.info("Output directory: %s", output_dir.resolve())

    # ------------------------------------------------------------------
    # Per-class candidate pools  (populated during first pass)
    # ------------------------------------------------------------------
    candidates: dict[int, list[Sample]] = {cls: [] for cls in range(len(MASTER_CLASSES))}

    with tempfile.TemporaryDirectory(prefix="avsca_v2_") as scratch_str:
        scratch = Path(scratch_str)

        # ── VisDrone (custom CSV format) ────────────────────────────────
        # train zip → split="train"
        # val   zip → split="val"
        # test  zip → split="train"  (fold in for more data)
        for key, forced_split in (
            ("visdrone_train", "train"),
            ("visdrone_val",   "val"),
            ("visdrone_test",  "train"),
        ):
            dest = scratch / key
            dest.mkdir()
            _extract_zip(zip_paths[key], dest)
            collect_visdrone(dest, forced_split, candidates)

        # ── Building detection (Roboflow YOLOv8) ───────────────────────
        building_dir = scratch / "building"
        building_dir.mkdir()
        _extract_zip(zip_paths["building"], building_dir)
        building_remap = build_roboflow_remap(building_dir, "building", master_cls=2)
        collect_roboflow_yolo(building_dir, "building", building_remap, candidates)

        # ── Tree archive (Roboflow YOLOv8) ─────────────────────────────
        tree_dir = scratch / "tree"
        tree_dir.mkdir()
        _extract_zip(zip_paths["tree"], tree_dir)
        tree_remap = build_roboflow_remap(tree_dir, "tree", master_cls=3)
        collect_roboflow_yolo(tree_dir, "tree", tree_remap, candidates)

        # ── Balanced sampling ───────────────────────────────────────────
        sampled, cap = sample_balanced(candidates, seed=args.seed)

        # ── Write to output directory ───────────────────────────────────
        counts = write_dataset(sampled, output_dir)

    # Temp dir is cleaned up here; all output already on disk
    generate_data_yaml(output_dir)
    print_summary(counts, cap)


if __name__ == "__main__":
    main()
