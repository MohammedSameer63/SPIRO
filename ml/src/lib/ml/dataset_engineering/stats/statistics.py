"""
SPIRO ML — DatasetStatistics
Computes comprehensive per-dataset and merged statistics:
  - Class distribution & imbalance
  - Images per class
  - Annotations per class
  - Bounding box size distribution
  - Image resolution distribution
  - Duplicate percentages
  - Rare class identification
  - Dataset coverage per SPIRO group
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from tqdm import tqdm

from lib.ml.core.logger import get_logger
from lib.ml.dataset_engineering.taxonomy import TaxonomyLoader

log = get_logger(__name__)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}


class DatasetStatistics:
    """
    Analyse a YOLO-format dataset directory and produce a full statistics report.

    Parameters
    ----------
    dataset_dir : Path
        Root with images/ and labels/ subdirectories.
    name : str
        Human-readable dataset name.

    Example
    -------
    >>> stats = DatasetStatistics(Path("datasets/merged"), name="SPIRO-Merged")
    >>> report = stats.compute()
    >>> stats.save_json("reports/dataset_statistics.json")
    """

    def __init__(self, dataset_dir: Path, name: str = "dataset") -> None:
        self.dataset_dir = Path(dataset_dir)
        self.images_dir = self.dataset_dir / "images"
        self.labels_dir = self.dataset_dir / "labels"
        self.name = name
        self.taxonomy = TaxonomyLoader()
        self._report: Dict[str, Any] = {}

    def compute(self) -> Dict[str, Any]:
        """Compute all statistics. Returns the full report dict."""
        log.info(f"Computing statistics for {self.name}")

        images = sorted(
            p for p in self.images_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS
        )
        labels = list(self.labels_dir.iterdir()) if self.labels_dir.exists() else []

        # ── Basic counts ───────────────────────────────────────────────────
        n_images = len(images)
        n_labels = len([l for l in labels if l.suffix == ".txt"])

        # ── Per-image annotation gathering ────────────────────────────────
        class_counter: Counter = Counter()
        bbox_widths: List[float] = []
        bbox_heights: List[float] = []
        bbox_areas: List[float] = []
        img_widths: List[int] = []
        img_heights: List[int] = []
        imgs_per_class: Dict[int, set] = defaultdict(set)
        anns_per_image: List[int] = []
        empty_images = 0

        for img_path in tqdm(images, desc="Computing stats"):
            lbl_path = self.labels_dir / f"{img_path.stem}.txt"

            # Image resolution
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            ih, iw = img.shape[:2]
            img_widths.append(iw)
            img_heights.append(ih)

            if not lbl_path.exists() or lbl_path.stat().st_size == 0:
                empty_images += 1
                anns_per_image.append(0)
                continue

            ann_count = 0
            with open(lbl_path) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) != 5:
                        continue
                    cls_id = int(parts[0])
                    cx, cy, w, h = map(float, parts[1:])
                    class_counter[cls_id] += 1
                    bbox_widths.append(w)
                    bbox_heights.append(h)
                    bbox_areas.append(w * h)
                    imgs_per_class[cls_id].add(img_path.stem)
                    ann_count += 1
            anns_per_image.append(ann_count)

        # ── Imbalance metrics ──────────────────────────────────────────────
        total_anns = sum(class_counter.values())
        class_dist = {
            self.taxonomy.id_to_name(k): v
            for k, v in sorted(class_counter.items())
        }
        images_per_class = {
            self.taxonomy.id_to_name(k): len(v)
            for k, v in sorted(imgs_per_class.items())
        }

        all_counts = list(class_counter.values())
        imbalance_ratio = (max(all_counts) / max(min(all_counts), 1)) if all_counts else 0

        # ── Rare classes ──────────────────────────────────────────────────
        rare_threshold = max(10, int(np.median(all_counts) * 0.1)) if all_counts else 10
        rare_classes = [
            self.taxonomy.id_to_name(cls_id)
            for cls_id, cnt in class_counter.items()
            if cnt < rare_threshold
        ]

        # ── Coverage per group ────────────────────────────────────────────
        group_coverage: Dict[str, Dict] = {}
        for group in self.taxonomy.group_names():
            group_ids = self.taxonomy.group_ids(group)
            represented = [i for i in group_ids if i in class_counter]
            group_coverage[group] = {
                "total_classes": len(group_ids),
                "represented": len(represented),
                "coverage_pct": round(100 * len(represented) / max(len(group_ids), 1), 1),
                "annotations": sum(class_counter.get(i, 0) for i in group_ids),
            }

        # ── Missing classes (in taxonomy but not in dataset) ──────────────
        seen_ids = set(class_counter.keys())
        missing_classes = [
            self.taxonomy.id_to_name(i)
            for i in range(self.taxonomy.num_classes)
            if i not in seen_ids
        ]

        # ── BBox distribution ─────────────────────────────────────────────
        def _array_stats(arr: List[float]) -> Dict:
            if not arr:
                return {}
            a = np.array(arr)
            return {
                "mean": float(a.mean()),
                "std": float(a.std()),
                "min": float(a.min()),
                "p25": float(np.percentile(a, 25)),
                "p50": float(np.percentile(a, 50)),
                "p75": float(np.percentile(a, 75)),
                "max": float(a.max()),
            }

        self._report = {
            "name": self.name,
            "total_images": n_images,
            "total_labels": n_labels,
            "total_annotations": total_anns,
            "empty_images": empty_images,
            "avg_annotations_per_image": round(
                float(np.mean(anns_per_image)) if anns_per_image else 0, 2
            ),
            "imbalance_ratio": round(imbalance_ratio, 1),
            "represented_classes": len(class_counter),
            "missing_classes": missing_classes,
            "rare_classes": rare_classes,
            "rare_threshold": rare_threshold,
            "class_distribution": class_dist,
            "images_per_class": images_per_class,
            "group_coverage": group_coverage,
            "bbox_width_stats": _array_stats(bbox_widths),
            "bbox_height_stats": _array_stats(bbox_heights),
            "bbox_area_stats": _array_stats(bbox_areas),
            "image_width_stats": _array_stats([float(w) for w in img_widths]),
            "image_height_stats": _array_stats([float(h) for h in img_heights]),
        }

        log.info(
            f"Stats: {n_images} images | {total_anns} annotations | "
            f"{len(class_counter)}/{self.taxonomy.num_classes} classes represented | "
            f"imbalance ratio={imbalance_ratio:.0f}x"
        )
        return self._report

    def save_json(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self._report, f, indent=2)
        log.info(f"Statistics saved to {path}")

    def generate_markdown_report(self, path: Path) -> None:
        """Write a human-readable Markdown report."""
        if not self._report:
            self.compute()
        r = self._report
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        lines = [
            f"# SPIRO Dataset Report — {r['name']}",
            "",
            "## Summary",
            "",
            f"| Metric | Value |",
            f"|---|---|",
            f"| Total images | {r['total_images']:,} |",
            f"| Total annotations | {r['total_annotations']:,} |",
            f"| Empty images | {r['empty_images']:,} |",
            f"| Avg annotations/image | {r['avg_annotations_per_image']} |",
            f"| Classes represented | {r['represented_classes']}/{self.taxonomy.num_classes} |",
            f"| Imbalance ratio | {r['imbalance_ratio']}x |",
            f"| Rare classes | {len(r['rare_classes'])} |",
            "",
            "## Class Distribution",
            "",
            "| Class | Annotations | Images |",
            "|---|---|---|",
        ]
        for cls_name in sorted(r["class_distribution"]):
            anns = r["class_distribution"][cls_name]
            imgs = r["images_per_class"].get(cls_name, 0)
            lines.append(f"| {cls_name} | {anns:,} | {imgs:,} |")

        lines += [
            "",
            "## Missing Classes",
            "",
        ]
        if r["missing_classes"]:
            for cls in r["missing_classes"]:
                lines.append(f"- {cls}")
        else:
            lines.append("All 109 SPIRO classes represented ✓")

        lines += [
            "",
            "## Rare Classes",
            f"_(fewer than {r['rare_threshold']} annotations)_",
            "",
        ]
        for cls in r.get("rare_classes", []):
            lines.append(f"- {cls}")

        lines += [
            "",
            "## Group Coverage",
            "",
            "| Group | Classes | Represented | Coverage | Annotations |",
            "|---|---|---|---|---|",
        ]
        for group, gc in r["group_coverage"].items():
            lines.append(
                f"| {group} | {gc['total_classes']} | "
                f"{gc['represented']} | {gc['coverage_pct']}% | {gc['annotations']:,} |"
            )

        lines += [
            "",
            "## Bounding Box Statistics",
            "",
            "| Metric | Width | Height | Area |",
            "|---|---|---|---|",
        ]
        bw = r.get("bbox_width_stats", {})
        bh = r.get("bbox_height_stats", {})
        ba = r.get("bbox_area_stats", {})
        for stat in ["mean", "std", "min", "p50", "max"]:
            lines.append(
                f"| {stat} | "
                f"{bw.get(stat, 0):.4f} | "
                f"{bh.get(stat, 0):.4f} | "
                f"{ba.get(stat, 0):.6f} |"
            )

        path.write_text("\n".join(lines), encoding="utf-8")
        log.info(f"Markdown report saved to {path}")
