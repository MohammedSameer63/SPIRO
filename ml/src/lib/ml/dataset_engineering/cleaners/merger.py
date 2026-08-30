"""
SPIRO ML — DatasetMerger
Merges all individually-mapped and cleaned datasets into a single
unified SPIRO dataset with:
  - Normalised folder structure
  - Conflict-free image naming (dataset prefix)
  - YOLO annotation format
  - Per-source provenance metadata
  - Deduplication across datasets
"""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from tqdm import tqdm

from lib.ml.core.logger import get_logger
from lib.ml.dataset_engineering.taxonomy import TaxonomyLoader

log = get_logger(__name__)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}

_MERGED_DIR = Path("datasets/merged")


class DatasetMerger:
    """
    Merges multiple YOLO-format dataset directories into one.

    Parameters
    ----------
    source_dirs : list of Paths
        Each is a cleaned mapped dataset root with images/ and labels/.
    output_dir : Path
        Merged output location.
    dedup_across_sources : bool
        If True, run MD5 dedup across all source datasets before merging.

    Example
    -------
    >>> merger = DatasetMerger(
    ...     source_dirs=[Path("datasets/mapped/TACO"),
    ...                  Path("datasets/mapped/TrashNet")],
    ...     output_dir=Path("datasets/merged"),
    ... )
    >>> report = merger.merge()
    """

    def __init__(
        self,
        source_dirs: List[Path],
        output_dir: Path = _MERGED_DIR,
        dedup_across_sources: bool = True,
    ) -> None:
        self.source_dirs = [Path(d) for d in source_dirs]
        self.output_dir = Path(output_dir)
        self.dedup = dedup_across_sources
        self.taxonomy = TaxonomyLoader()

        (self.output_dir / "images").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "labels").mkdir(parents=True, exist_ok=True)

        self._report: Dict[str, Any] = {
            "sources": [str(d) for d in source_dirs],
            "output_dir": str(self.output_dir),
            "per_source": {},
            "total_images": 0,
            "total_annotations": 0,
            "cross_source_duplicates_removed": 0,
            "class_distribution": {name: 0 for name in self.taxonomy.all_names()},
        }

    def merge(self) -> Dict[str, Any]:
        """
        Merge all source datasets.

        Returns
        -------
        dict — merge report
        """
        log.info(f"Merging {len(self.source_dirs)} datasets → {self.output_dir}")

        seen_hashes: Set[str] = set()
        total_imgs = 0
        total_anns = 0

        for src_dir in self.source_dirs:
            dataset_name = src_dir.name
            images_dir = src_dir / "images"
            labels_dir = src_dir / "labels"

            if not images_dir.exists():
                log.warning(f"No images/ in {src_dir}, skipping")
                continue

            src_imgs = sorted(
                p for p in images_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS
            )

            src_count = 0
            src_anns = 0
            src_dups = 0

            for img_path in tqdm(src_imgs, desc=f"Merging {dataset_name}"):
                # Cross-source dedup
                if self.dedup:
                    h = self._md5(img_path)
                    if h in seen_hashes:
                        src_dups += 1
                        continue
                    seen_hashes.add(h)

                # Build unique output stem
                out_stem = f"{dataset_name}__{img_path.stem}"
                out_ext = img_path.suffix.lower()
                dst_img = self.output_dir / "images" / f"{out_stem}{out_ext}"
                dst_lbl = self.output_dir / "labels" / f"{out_stem}.txt"

                shutil.copy2(img_path, dst_img)

                lbl_src = labels_dir / f"{img_path.stem}.txt"
                anns_this = 0
                if lbl_src.exists():
                    shutil.copy2(lbl_src, dst_lbl)
                    # Count & record class distribution
                    with open(lbl_src) as f:
                        for line in f:
                            line = line.strip()
                            if not line:
                                continue
                            parts = line.split()
                            if len(parts) == 5:
                                cls_id = int(parts[0])
                                name = self.taxonomy.id_to_name(cls_id)
                                self._report["class_distribution"][name] += 1
                                anns_this += 1
                else:
                    dst_lbl.touch()

                src_count += 1
                src_anns += anns_this

            self._report["per_source"][dataset_name] = {
                "images": src_count,
                "annotations": src_anns,
                "duplicates_skipped": src_dups,
            }
            self._report["cross_source_duplicates_removed"] += src_dups
            total_imgs += src_count
            total_anns += src_anns
            log.info(
                f"  {dataset_name}: {src_count} images, {src_anns} annotations "
                f"({src_dups} cross-source dups removed)"
            )

        self._report["total_images"] = total_imgs
        self._report["total_annotations"] = total_anns

        # Write provenance metadata
        meta_path = self.output_dir / "merge_meta.json"
        with open(meta_path, "w") as f:
            json.dump(self._report, f, indent=2)

        log.info(
            f"Merge complete — {total_imgs} total images, "
            f"{total_anns} total annotations"
        )
        return self._report

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _md5(path: Path) -> str:
        h = hashlib.md5()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()

    def report(self) -> Dict[str, Any]:
        return self._report

    def save_report(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self._report, f, indent=2)
        log.info(f"Merge report saved to {path}")
