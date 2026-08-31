"""
SPIRO ML — DatasetManager
Handles the full data lifecycle:
  1. Validation of raw annotations
  2. Preprocessing (resize, normalize, format conversion)
  3. Stratified train/val/test splitting
  4. Dataset statistics analysis
  5. YOLO-format directory creation
"""
from __future__ import annotations

import json
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

from lib.ml.core.config import ConfigManager
from lib.ml.core.logger import get_logger

log = get_logger(__name__)


class DatasetManager:
    """
    Manages SPIRO dataset from raw files to YOLO-ready splits.

    Parameters
    ----------
    cfg : ConfigManager
        Loaded project config.

    Example
    -------
    >>> dm = DatasetManager(cfg)
    >>> dm.validate_raw()
    >>> dm.create_splits()
    >>> dm.analyze()
    >>> dm.export_stats("reports/dataset_stats.json")
    """

    YOLO_EXTENSIONS = {".txt"}
    IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

    def __init__(self, cfg: ConfigManager) -> None:
        self.cfg = cfg
        self.raw_dir = Path(cfg.paths.raw_data)
        self.processed_dir = Path(cfg.paths.processed_data)
        self.splits_dir = Path(cfg.paths.splits_dir)
        self.num_classes = cfg.dataset.num_classes
        self.class_names = list(cfg.dataset.class_names)
        self.split_ratios = cfg.dataset.split_ratios
        self.image_size = tuple(cfg.dataset.image_size)
        self._stats: Dict = {}

    # ------------------------------------------------------------------
    # 1. Validation
    # ------------------------------------------------------------------

    def validate_raw(self) -> bool:
        """
        Validate raw data directory for:
        - Paired image + label files
        - Valid YOLO label format (class_id cx cy w h)
        - Class ID range
        - Image readability

        Returns True if valid, raises ValueError on failure.
        """
        log.info(f"Validating raw dataset at: {self.raw_dir}")
        images_dir = self.raw_dir / "images"
        labels_dir = self.raw_dir / "labels"

        if not images_dir.exists():
            raise FileNotFoundError(f"images/ directory not found at {images_dir}")
        if not labels_dir.exists():
            raise FileNotFoundError(f"labels/ directory not found at {labels_dir}")

        images = sorted(
            p for p in images_dir.iterdir() if p.suffix.lower() in self.IMAGE_EXTENSIONS
        )
        labels = sorted(
            p for p in labels_dir.iterdir() if p.suffix in self.YOLO_EXTENSIONS
        )

        if not images:
            raise ValueError("No images found in raw/images/")

        img_stems = {p.stem for p in images}
        lbl_stems = {p.stem for p in labels}
        missing_labels = img_stems - lbl_stems
        orphan_labels = lbl_stems - img_stems

        if missing_labels:
            log.warning(f"{len(missing_labels)} images have no label file")
        if orphan_labels:
            log.warning(f"{len(orphan_labels)} label files have no matching image")

        errors: List[str] = []
        for lbl_path in tqdm(labels, desc="Validating labels"):
            errors.extend(self._validate_label_file(lbl_path))

        unreadable = []
        for img_path in tqdm(images, desc="Validating images"):
            img = cv2.imread(str(img_path))
            if img is None:
                unreadable.append(img_path.name)

        if unreadable:
            errors.extend([f"Unreadable image: {n}" for n in unreadable])

        if errors:
            for e in errors[:20]:
                log.error(e)
            if len(errors) > 20:
                log.error(f"... and {len(errors) - 20} more errors")
            raise ValueError(f"Dataset validation failed with {len(errors)} errors.")

        log.info(
            f"Validation passed — {len(images)} images, {len(labels)} labels"
        )
        return True

    def _validate_label_file(self, path: Path) -> List[str]:
        errors = []
        with open(path, "r") as f:
            for i, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) != 5:
                    errors.append(
                        f"{path.name}:{i} — expected 5 values, got {len(parts)}"
                    )
                    continue
                try:
                    cls_id = int(parts[0])
                    cx, cy, w, h = map(float, parts[1:])
                except ValueError:
                    errors.append(f"{path.name}:{i} — non-numeric values")
                    continue
                if cls_id < 0 or cls_id >= self.num_classes:
                    errors.append(
                        f"{path.name}:{i} — class_id {cls_id} out of range [0, {self.num_classes-1}]"
                    )
                for val, name in zip([cx, cy, w, h], ["cx", "cy", "w", "h"]):
                    if not (0.0 <= val <= 1.0):
                        errors.append(
                            f"{path.name}:{i} — {name}={val:.4f} not in [0,1]"
                        )
        return errors

    # ------------------------------------------------------------------
    # 2. Preprocessing
    # ------------------------------------------------------------------

    def preprocess(self, target_size: Optional[Tuple[int, int]] = None) -> None:
        """
        Copy images from raw/ to processed/, optionally resizing.
        Labels are copied as-is (YOLO format is size-invariant).
        """
        target_size = target_size or self.image_size
        log.info(f"Preprocessing images → {target_size}")

        raw_images = self.raw_dir / "images"
        raw_labels = self.raw_dir / "labels"
        proc_images = self.processed_dir / "images"
        proc_labels = self.processed_dir / "labels"
        proc_images.mkdir(parents=True, exist_ok=True)
        proc_labels.mkdir(parents=True, exist_ok=True)

        image_paths = sorted(
            p for p in raw_images.iterdir() if p.suffix.lower() in self.IMAGE_EXTENSIONS
        )

        for img_path in tqdm(image_paths, desc="Preprocessing"):
            img = cv2.imread(str(img_path))
            if img is None:
                log.warning(f"Skipping unreadable: {img_path.name}")
                continue

            # Resize with letterboxing to preserve aspect ratio
            img_resized = self._letterbox(img, target_size)
            out_path = proc_images / (img_path.stem + ".jpg")
            cv2.imwrite(str(out_path), img_resized, [cv2.IMWRITE_JPEG_QUALITY, 95])

            # Copy label
            lbl_src = raw_labels / (img_path.stem + ".txt")
            lbl_dst = proc_labels / (img_path.stem + ".txt")
            if lbl_src.exists():
                shutil.copy2(lbl_src, lbl_dst)
            else:
                # Create empty label file (background image)
                lbl_dst.touch()

        log.info(f"Preprocessing complete — {len(image_paths)} images")

    @staticmethod
    def _letterbox(
        img: np.ndarray,
        target: Tuple[int, int],
        color: Tuple[int, int, int] = (114, 114, 114),
    ) -> np.ndarray:
        """Resize with padding to maintain aspect ratio."""
        h, w = img.shape[:2]
        tw, th = target
        scale = min(tw / w, th / h)
        nw, nh = int(w * scale), int(h * scale)
        img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
        canvas = np.full((th, tw, 3), color, dtype=np.uint8)
        x_off = (tw - nw) // 2
        y_off = (th - nh) // 2
        canvas[y_off : y_off + nh, x_off : x_off + nw] = img
        return canvas

    # ------------------------------------------------------------------
    # 3. Splitting
    # ------------------------------------------------------------------

    def create_splits(self) -> Dict[str, List[str]]:
        """
        Create stratified train/val/test splits.
        Writes split index files and builds YOLO directory structure.
        Returns dict with stem lists per split.
        """
        log.info("Creating dataset splits")
        proc_images = self.processed_dir / "images"
        proc_labels = self.processed_dir / "labels"

        stems = [
            p.stem
            for p in sorted(proc_images.iterdir())
            if p.suffix.lower() in self.IMAGE_EXTENSIONS
        ]

        # Build per-class lists for stratification
        class_to_stems: Dict[int, List[str]] = defaultdict(list)
        no_label: List[str] = []
        for stem in stems:
            lbl = proc_labels / f"{stem}.txt"
            if not lbl.exists() or lbl.stat().st_size == 0:
                no_label.append(stem)
                continue
            with open(lbl) as f:
                classes = [int(l.split()[0]) for l in f if l.strip()]
            primary = Counter(classes).most_common(1)[0][0] if classes else -1
            class_to_stems[primary].append(stem)

        splits: Dict[str, List[str]] = {"train": [], "val": [], "test": []}
        ratios = self.split_ratios

        for cls_id, cls_stems in class_to_stems.items():
            random.shuffle(cls_stems)
            n = len(cls_stems)
            n_val = max(1, int(n * ratios["val"]))
            n_test = max(1, int(n * ratios["test"]))
            splits["val"].extend(cls_stems[:n_val])
            splits["test"].extend(cls_stems[n_val : n_val + n_test])
            splits["train"].extend(cls_stems[n_val + n_test :])

        # Background images — put in train only
        splits["train"].extend(no_label)

        for split_name, split_stems in splits.items():
            random.shuffle(split_stems)
            self._build_yolo_split(split_name, split_stems, proc_images, proc_labels)

        log.info(
            f"Splits created — train:{len(splits['train'])} "
            f"val:{len(splits['val'])} test:{len(splits['test'])}"
        )
        return splits

    def _build_yolo_split(
        self,
        split: str,
        stems: List[str],
        src_images: Path,
        src_labels: Path,
    ) -> None:
        dst_img = self.processed_dir / split / "images"
        dst_lbl = self.processed_dir / split / "labels"
        dst_img.mkdir(parents=True, exist_ok=True)
        dst_lbl.mkdir(parents=True, exist_ok=True)

        for stem in stems:
            # Image
            for ext in [".jpg", ".jpeg", ".png"]:
                src = src_images / f"{stem}{ext}"
                if src.exists():
                    shutil.copy2(src, dst_img / src.name)
                    break
            # Label
            src_l = src_labels / f"{stem}.txt"
            if src_l.exists():
                shutil.copy2(src_l, dst_lbl / f"{stem}.txt")
            else:
                (dst_lbl / f"{stem}.txt").touch()

        # Write index file
        idx_path = self.splits_dir / f"{split}.txt"
        idx_path.parent.mkdir(parents=True, exist_ok=True)
        with open(idx_path, "w") as f:
            f.writelines(f"{s}\n" for s in stems)

    # ------------------------------------------------------------------
    # 4. Analysis
    # ------------------------------------------------------------------

    def analyze(self) -> Dict:
        """Compute and cache dataset statistics."""
        log.info("Analysing dataset statistics")
        stats: Dict = {}

        for split in ["train", "val", "test"]:
            split_dir = self.processed_dir / split
            img_dir = split_dir / "images"
            lbl_dir = split_dir / "labels"
            if not img_dir.exists():
                continue

            images = list(img_dir.iterdir())
            class_counts: Counter = Counter()
            box_counts: List[int] = []

            for lbl_path in lbl_dir.iterdir():
                with open(lbl_path) as f:
                    lines = [l.strip() for l in f if l.strip()]
                box_counts.append(len(lines))
                for line in lines:
                    cls_id = int(line.split()[0])
                    class_counts[cls_id] += 1

            stats[split] = {
                "num_images": len(images),
                "num_boxes": sum(box_counts),
                "avg_boxes_per_image": float(np.mean(box_counts)) if box_counts else 0,
                "class_distribution": {
                    self.class_names[k]: v for k, v in sorted(class_counts.items())
                },
            }

        self._stats = stats
        log.info("Analysis complete")
        return stats

    def export_stats(self, path: str | Path) -> None:
        """Write statistics to JSON."""
        if not self._stats:
            self.analyze()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self._stats, f, indent=2)
        log.info(f"Dataset stats written to {path}")

    def update_dataset_yaml(self, yaml_path: str | Path = "configs/dataset.yaml") -> None:
        """Patch the dataset.yaml with current split sizes."""
        if not self._stats:
            self.analyze()
        yaml_path = Path(yaml_path)
        with open(yaml_path) as f:
            data = yaml.safe_load(f)
        data["stats"]["total_images"] = sum(
            v.get("num_images", 0) for v in self._stats.values()
        )
        for split in ["train", "val", "test"]:
            key = f"{split}_images"
            data["stats"][key] = self._stats.get(split, {}).get("num_images", 0)
        with open(yaml_path, "w") as f:
            yaml.dump(data, f, default_flow_style=False)
        log.info(f"Updated {yaml_path} with dataset statistics")
