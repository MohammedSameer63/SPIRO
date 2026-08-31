"""
SPIRO ML — DataCleaner
Full cleaning pipeline for mapped YOLO datasets:

1. Corrupt image detection (OpenCV read check + file size)
2. Exact duplicate detection (MD5 hash)
3. Near-duplicate detection (perceptual hash / dHash)
4. Missing annotation detection
5. Invalid / out-of-range bounding box detection
6. Image resolution validation
7. Class ID validation
8. Orphan label removal (label with no matching image)
9. Cleaning report generation
"""
from __future__ import annotations

import hashlib
import json
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import cv2
import numpy as np
from tqdm import tqdm

from lib.ml.core.logger import get_logger
from lib.ml.dataset_engineering.taxonomy import TaxonomyLoader

log = get_logger(__name__)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}


class DataCleaner:
    """
    Cleans a YOLO-format dataset directory in-place (or to a separate output dir).

    Parameters
    ----------
    dataset_dir : Path
        Root containing images/ and labels/ subdirectories.
    output_dir : Path, optional
        If provided, cleaned data is written here. Otherwise edits in-place.
    min_resolution : tuple
        Minimum (width, height) to keep an image.
    max_resolution : tuple
        Maximum (width, height) — larger images are resized or flagged.
    min_bbox_area : float
        Minimum bbox area as fraction of image area (0.0–1.0) to keep annotation.
    phash_threshold : int
        Hamming distance threshold for near-duplicate detection (0 = exact, 10 = loose).
    num_classes : int
        Expected number of SPIRO classes.

    Example
    -------
    >>> cleaner = DataCleaner(Path("datasets/mapped/TACO"))
    >>> report = cleaner.clean()
    >>> cleaner.save_report("reports/taco_cleaning_report.json")
    """

    def __init__(
        self,
        dataset_dir: Path,
        output_dir: Optional[Path] = None,
        min_resolution: Tuple[int, int] = (64, 64),
        max_resolution: Tuple[int, int] = (8192, 8192),
        min_bbox_area: float = 0.0004,  # ~20x20 px at 1280 width
        phash_threshold: int = 8,
        num_classes: int = 109,
    ) -> None:
        self.dataset_dir = Path(dataset_dir)
        self.output_dir = Path(output_dir) if output_dir else None
        self.images_dir = self.dataset_dir / "images"
        self.labels_dir = self.dataset_dir / "labels"
        self.min_res = min_resolution
        self.max_res = max_resolution
        self.min_bbox_area = min_bbox_area
        self.phash_threshold = phash_threshold
        self.num_classes = num_classes
        self.taxonomy = TaxonomyLoader()

        if self.output_dir:
            (self.output_dir / "images").mkdir(parents=True, exist_ok=True)
            (self.output_dir / "labels").mkdir(parents=True, exist_ok=True)

        self._report: Dict[str, Any] = {
            "dataset_dir": str(self.dataset_dir),
            "total_images": 0,
            "corrupt_images": [],
            "exact_duplicates": [],
            "near_duplicates": [],
            "low_resolution": [],
            "missing_labels": [],
            "orphan_labels": [],
            "invalid_bbox_images": [],
            "invalid_class_images": [],
            "clean_images": 0,
            "removed_annotations": 0,
        }

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def clean(self, remove: bool = True) -> Dict[str, Any]:
        """
        Run full cleaning pipeline.

        Parameters
        ----------
        remove : bool
            If True, remove/skip corrupt, duplicate, and invalid items.
            If False, only report (dry run).

        Returns
        -------
        dict — cleaning report
        """
        log.info(f"Starting cleaning pipeline on {self.dataset_dir}")

        images = sorted(
            p for p in self.images_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS
        )
        self._report["total_images"] = len(images)

        # 1. Detect corrupt images
        corrupt = self._detect_corrupt(images)
        self._report["corrupt_images"] = [str(p) for p in corrupt]

        # 2. Exact duplicate detection
        duplicates = self._detect_exact_duplicates(images, exclude=corrupt)
        self._report["exact_duplicates"] = [
            [str(p) for p in group] for group in duplicates
        ]

        # 3. Near-duplicate detection
        near_dups = self._detect_near_duplicates(
            images, exclude=corrupt | {p for g in duplicates for p in g[1:]}
        )
        self._report["near_duplicates"] = [
            [str(p) for p in group] for group in near_dups
        ]

        # 4. Resolution check
        low_res = self._detect_low_resolution(images, exclude=corrupt)
        self._report["low_resolution"] = [str(p) for p in low_res]

        # 5. Missing labels
        missing_lbl = self._detect_missing_labels(images)
        self._report["missing_labels"] = [str(p) for p in missing_lbl]

        # 6. Orphan labels
        orphans = self._detect_orphan_labels(images)
        self._report["orphan_labels"] = [str(p) for p in orphans]

        # 7. Annotation quality
        invalid_bbox, invalid_cls, removed_anns = self._validate_annotations(
            images, exclude=corrupt, remove=remove
        )
        self._report["invalid_bbox_images"] = [str(p) for p in invalid_bbox]
        self._report["invalid_class_images"] = [str(p) for p in invalid_cls]
        self._report["removed_annotations"] = removed_anns

        # Compute to-remove set
        to_remove: Set[Path] = set()
        if remove:
            to_remove |= corrupt
            to_remove |= {p for g in duplicates for p in g[1:]}  # keep first
            to_remove |= {p for g in near_dups for p in g[1:]}
            to_remove |= low_res

        # Write or copy clean images
        kept = 0
        for img_path in images:
            if img_path in to_remove:
                if self.output_dir is None:
                    # In-place: remove
                    img_path.unlink(missing_ok=True)
                    lbl = self.labels_dir / f"{img_path.stem}.txt"
                    lbl.unlink(missing_ok=True)
                continue
            if self.output_dir:
                dst = self.output_dir / "images" / img_path.name
                shutil.copy2(img_path, dst)
                lbl_src = self.labels_dir / f"{img_path.stem}.txt"
                lbl_dst = self.output_dir / "labels" / f"{img_path.stem}.txt"
                if lbl_src.exists():
                    shutil.copy2(lbl_src, lbl_dst)
                else:
                    lbl_dst.touch()
            kept += 1

        # Remove orphan labels in-place
        if remove and self.output_dir is None:
            for lbl_path in orphans:
                lbl_path.unlink(missing_ok=True)

        self._report["clean_images"] = kept
        log.info(
            f"Cleaning complete — kept {kept}/{len(images)} images | "
            f"corrupt={len(corrupt)} | exact_dups={sum(len(g)-1 for g in duplicates)} | "
            f"near_dups={sum(len(g)-1 for g in near_dups)} | low_res={len(low_res)}"
        )
        return self._report

    # ------------------------------------------------------------------
    # Detection methods
    # ------------------------------------------------------------------

    def _detect_corrupt(self, images: List[Path]) -> Set[Path]:
        corrupt: Set[Path] = set()
        for p in tqdm(images, desc="Corrupt check"):
            # Size check
            if p.stat().st_size < 1024:
                corrupt.add(p)
                continue
            # OpenCV read check
            img = cv2.imread(str(p))
            if img is None or img.size == 0:
                corrupt.add(p)
        log.info(f"Corrupt images: {len(corrupt)}")
        return corrupt

    def _detect_exact_duplicates(
        self, images: List[Path], exclude: Set[Path]
    ) -> List[List[Path]]:
        """Group images with identical MD5."""
        hash_to_paths: Dict[str, List[Path]] = defaultdict(list)
        for p in tqdm(images, desc="MD5 dedup"):
            if p in exclude:
                continue
            h = self._md5(p)
            hash_to_paths[h].append(p)
        groups = [v for v in hash_to_paths.values() if len(v) > 1]
        n = sum(len(g) - 1 for g in groups)
        log.info(f"Exact duplicates: {n} images in {len(groups)} groups")
        return groups

    def _detect_near_duplicates(
        self, images: List[Path], exclude: Set[Path]
    ) -> List[List[Path]]:
        """Detect near-duplicates using difference hash (dHash)."""
        hashes: List[Tuple[Path, int]] = []
        for p in tqdm(images, desc="pHash dedup"):
            if p in exclude:
                continue
            dhash = self._dhash(p)
            if dhash is not None:
                hashes.append((p, dhash))

        groups: List[List[Path]] = []
        used: Set[int] = set()
        for i, (pi, hi) in enumerate(hashes):
            if i in used:
                continue
            group = [pi]
            for j, (pj, hj) in enumerate(hashes[i + 1:], start=i + 1):
                if j in used:
                    continue
                dist = bin(hi ^ hj).count("1")
                if dist <= self.phash_threshold:
                    group.append(pj)
                    used.add(j)
            if len(group) > 1:
                groups.append(group)
                used.add(i)

        n = sum(len(g) - 1 for g in groups)
        log.info(f"Near-duplicates: {n} images in {len(groups)} groups")
        return groups

    def _detect_low_resolution(
        self, images: List[Path], exclude: Set[Path]
    ) -> Set[Path]:
        low: Set[Path] = set()
        for p in images:
            if p in exclude:
                continue
            img = cv2.imread(str(p))
            if img is None:
                continue
            h, w = img.shape[:2]
            if w < self.min_res[0] or h < self.min_res[1]:
                low.add(p)
        log.info(f"Low-resolution images (<{self.min_res}): {len(low)}")
        return low

    def _detect_missing_labels(self, images: List[Path]) -> List[Path]:
        missing = [
            p for p in images
            if not (self.labels_dir / f"{p.stem}.txt").exists()
        ]
        log.info(f"Missing label files: {len(missing)}")
        return missing

    def _detect_orphan_labels(self, images: List[Path]) -> List[Path]:
        img_stems = {p.stem for p in images}
        orphans = [
            p for p in self.labels_dir.iterdir()
            if p.suffix == ".txt" and p.stem not in img_stems
        ]
        log.info(f"Orphan label files: {len(orphans)}")
        return orphans

    def _validate_annotations(
        self, images: List[Path], exclude: Set[Path], remove: bool
    ) -> Tuple[List[Path], List[Path], int]:
        """
        Validate bounding boxes and class IDs.
        Removes invalid individual annotations (rewrites label files).
        Returns (invalid_bbox_images, invalid_class_images, removed_count).
        """
        invalid_bbox_imgs: List[Path] = []
        invalid_cls_imgs: List[Path] = []
        removed = 0

        for img_path in tqdm(images, desc="Annotation validation"):
            if img_path in exclude:
                continue
            lbl_path = self.labels_dir / f"{img_path.stem}.txt"
            if not lbl_path.exists():
                continue

            with open(lbl_path) as f:
                lines = [l.strip() for l in f if l.strip()]

            img = cv2.imread(str(img_path))
            if img is None:
                continue
            ih, iw = img.shape[:2]
            img_area = iw * ih

            valid_lines: List[str] = []
            has_bad_bbox = False
            has_bad_cls = False

            for line in lines:
                parts = line.split()
                if len(parts) != 5:
                    removed += 1
                    continue
                try:
                    cls_id = int(parts[0])
                    cx, cy, w, h = map(float, parts[1:])
                except ValueError:
                    removed += 1
                    continue

                # Class range
                if cls_id < 0 or cls_id >= self.num_classes:
                    has_bad_cls = True
                    removed += 1
                    continue

                # Coordinate range
                if not all(0.0 <= v <= 1.0 for v in [cx, cy, w, h]):
                    # Try to clip
                    cx = max(0.0, min(1.0, cx))
                    cy = max(0.0, min(1.0, cy))
                    w = max(0.001, min(1.0, w))
                    h = max(0.001, min(1.0, h))
                    has_bad_bbox = True

                # Minimum area
                abs_area = w * h * img_area
                if abs_area < (self.min_bbox_area * img_area * 1000000 / (iw * ih)):
                    # bbox area as fraction of image
                    bbox_frac = w * h
                    if bbox_frac < self.min_bbox_area:
                        removed += 1
                        has_bad_bbox = True
                        continue

                valid_lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

            if has_bad_bbox:
                invalid_bbox_imgs.append(img_path)
            if has_bad_cls:
                invalid_cls_imgs.append(img_path)

            if remove:
                with open(lbl_path, "w") as f:
                    f.writelines(line + "\n" for line in valid_lines)

        log.info(
            f"Annotation validation: {len(invalid_bbox_imgs)} bbox issues, "
            f"{len(invalid_cls_imgs)} class issues, {removed} annotations removed"
        )
        return invalid_bbox_imgs, invalid_cls_imgs, removed

    # ------------------------------------------------------------------
    # Hashing utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _md5(path: Path) -> str:
        h = hashlib.md5()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()

    @staticmethod
    def _dhash(path: Path, hash_size: int = 8) -> Optional[int]:
        """
        Compute difference hash (dHash) for perceptual deduplication.
        Returns an integer representing the hash bit pattern.
        """
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            return None
        # Resize to (hash_size+1) x hash_size
        img = cv2.resize(img, (hash_size + 1, hash_size), interpolation=cv2.INTER_AREA)
        # Horizontal difference
        diff = img[:, 1:] > img[:, :-1]
        # Pack bits into integer
        return sum(int(b) << i for i, b in enumerate(diff.flatten()))

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------

    def save_report(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self._report, f, indent=2)
        log.info(f"Cleaning report saved to {path}")

    def report(self) -> Dict[str, Any]:
        return self._report
