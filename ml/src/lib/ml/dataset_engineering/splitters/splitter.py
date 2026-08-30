"""
SPIRO ML — DatasetSplitter
Stratified train/val/test splitting that:
  - Maintains class balance across splits
  - Handles multi-label images (assigned by dominant class)
  - Uses reproducible random seeds
  - Writes YOLO-structured output directories
  - Generates split index files
"""
from __future__ import annotations

import json
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm

from lib.ml.core.logger import get_logger
from lib.ml.dataset_engineering.taxonomy import TaxonomyLoader

log = get_logger(__name__)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}


class DatasetSplitter:
    """
    Stratified train/val/test splitter for YOLO-format datasets.

    Parameters
    ----------
    merged_dir : Path
        Root of merged dataset (images/ + labels/).
    output_dir : Path
        Where to write train/ val/ test/ splits.
    train_ratio : float
    val_ratio : float
    test_ratio : float
    seed : int
    min_class_samples : int
        Classes with fewer than this many samples are flagged as rare.

    Example
    -------
    >>> splitter = DatasetSplitter(Path("datasets/merged"), Path("datasets/processed"))
    >>> report = splitter.split()
    """

    def __init__(
        self,
        merged_dir: Path,
        output_dir: Path,
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        seed: int = 42,
        min_class_samples: int = 20,
    ) -> None:
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"
        self.merged_dir = Path(merged_dir)
        self.output_dir = Path(output_dir)
        self.ratios = {"train": train_ratio, "val": val_ratio, "test": test_ratio}
        self.seed = seed
        self.min_class_samples = min_class_samples
        self.taxonomy = TaxonomyLoader()

        for split in ["train", "val", "test"]:
            (self.output_dir / split / "images").mkdir(parents=True, exist_ok=True)
            (self.output_dir / split / "labels").mkdir(parents=True, exist_ok=True)

        self._report: Dict[str, Any] = {
            "seed": seed,
            "ratios": self.ratios,
            "splits": {"train": 0, "val": 0, "test": 0},
            "class_distribution": {
                "train": {}, "val": {}, "test": {}
            },
            "rare_classes": [],
        }

    def split(self) -> Dict[str, Any]:
        """
        Execute the stratified split.

        Returns
        -------
        dict — split report
        """
        random.seed(self.seed)
        log.info(f"Splitting dataset from {self.merged_dir}")

        images_dir = self.merged_dir / "images"
        labels_dir = self.merged_dir / "labels"

        image_paths = sorted(
            p for p in images_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS
        )

        # Assign each image a primary class (dominant by count)
        class_to_stems: Dict[int, List[str]] = defaultdict(list)
        stem_to_class: Dict[str, int] = {}
        background: List[str] = []

        for img_path in image_paths:
            stem = img_path.stem
            lbl = labels_dir / f"{stem}.txt"
            cls_counts: Counter = Counter()
            if lbl.exists():
                with open(lbl) as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) == 5:
                            cls_counts[int(parts[0])] += 1
            if cls_counts:
                primary = cls_counts.most_common(1)[0][0]
                class_to_stems[primary].append(stem)
                stem_to_class[stem] = primary
            else:
                background.append(stem)

        # Identify rare classes
        rare = [
            self.taxonomy.id_to_name(cls_id)
            for cls_id, stems in class_to_stems.items()
            if len(stems) < self.min_class_samples
        ]
        self._report["rare_classes"] = rare
        if rare:
            log.warning(f"Rare classes (< {self.min_class_samples} samples): {rare}")

        # Stratified split per class
        split_assignment: Dict[str, str] = {}
        for cls_id, stems in class_to_stems.items():
            random.shuffle(stems)
            n = len(stems)
            n_val = max(1, int(n * self.ratios["val"]))
            n_test = max(1, int(n * self.ratios["test"]))
            for s in stems[:n_val]:
                split_assignment[s] = "val"
            for s in stems[n_val: n_val + n_test]:
                split_assignment[s] = "test"
            for s in stems[n_val + n_test:]:
                split_assignment[s] = "train"

        # Background images → train only
        for s in background:
            split_assignment[s] = "train"

        # Copy files
        for img_path in tqdm(image_paths, desc="Writing splits"):
            stem = img_path.stem
            split = split_assignment.get(stem, "train")

            dst_img = self.output_dir / split / "images" / img_path.name
            dst_lbl = self.output_dir / split / "labels" / f"{stem}.txt"

            shutil.copy2(img_path, dst_img)
            lbl_src = labels_dir / f"{stem}.txt"
            if lbl_src.exists():
                shutil.copy2(lbl_src, dst_lbl)
                # Record class distribution per split
                with open(lbl_src) as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) == 5:
                            name = self.taxonomy.id_to_name(int(parts[0]))
                            self._report["class_distribution"][split][name] = (
                                self._report["class_distribution"][split].get(name, 0) + 1
                            )
            else:
                dst_lbl.touch()

            self._report["splits"][split] += 1

        # Write index files
        for split in ["train", "val", "test"]:
            stems = [p.stem for p in (self.output_dir / split / "images").iterdir()]
            idx_path = self.output_dir / f"{split}.txt"
            with open(idx_path, "w") as f:
                f.writelines(s + "\n" for s in stems)

        log.info(
            f"Split complete — train:{self._report['splits']['train']} "
            f"val:{self._report['splits']['val']} "
            f"test:{self._report['splits']['test']}"
        )
        return self._report

    def save_report(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self._report, f, indent=2)
        log.info(f"Split report saved to {path}")
