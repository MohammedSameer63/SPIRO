"""
SPIRO ML — Dataset Visualization
Generates all quality and analysis charts:
  - Class histogram (frequency bar chart)
  - Bounding box size histogram
  - Image resolution scatter/heatmap
  - Dataset composition pie chart
  - Train/Val/Test distribution
  - Class imbalance heatmap
  - Group coverage bar chart
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from lib.ml.core.logger import get_logger
from lib.ml.dataset_engineering.taxonomy import TaxonomyLoader

log = get_logger(__name__)

_COLORS = plt.cm.tab20.colors


class DatasetVisualizer:
    """
    Generates dataset quality visualizations from a statistics report dict.

    Parameters
    ----------
    stats : dict
        Output of DatasetStatistics.compute().
    output_dir : Path
        Where to save chart PNGs.

    Example
    -------
    >>> viz = DatasetVisualizer(stats_report, Path("reports/charts"))
    >>> viz.generate_all()
    """

    def __init__(self, stats: Dict[str, Any], output_dir: Path) -> None:
        self.stats = stats
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.taxonomy = TaxonomyLoader()

    def generate_all(self) -> None:
        """Generate all charts."""
        log.info(f"Generating visualizations → {self.output_dir}")
        self.class_histogram()
        self.bbox_histogram()
        self.resolution_scatter()
        self.dataset_composition()
        self.split_distribution()
        self.group_coverage()
        self.imbalance_heatmap()
        log.info("All charts generated ✓")

    # ------------------------------------------------------------------
    # Individual charts
    # ------------------------------------------------------------------

    def class_histogram(self) -> Path:
        """Bar chart of annotation count per class."""
        dist = self.stats.get("class_distribution", {})
        if not dist:
            log.warning("No class distribution data for histogram")
            return self.output_dir / "class_histogram.png"

        names = list(dist.keys())
        counts = [dist[n] for n in names]
        # Sort by count descending
        order = np.argsort(counts)[::-1]
        names = [names[i] for i in order]
        counts = [counts[i] for i in order]

        fig, ax = plt.subplots(figsize=(max(12, len(names) * 0.18), 7))
        bars = ax.bar(range(len(names)), counts, color="#4C8BF5", edgecolor="none")
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=90, fontsize=6)
        ax.set_ylabel("Annotation count")
        ax.set_title(f"Class Distribution — {self.stats.get('name', '')}")
        ax.grid(axis="y", alpha=0.3)
        # Threshold line for rare classes
        if counts:
            median = np.median(counts) * 0.1
            ax.axhline(y=median, color="red", linestyle="--", alpha=0.6, label=f"Rare threshold ({int(median)})")
            ax.legend()
        plt.tight_layout()
        out = self.output_dir / "class_histogram.png"
        plt.savefig(str(out), dpi=150)
        plt.close()
        log.info(f"Class histogram → {out}")
        return out

    def bbox_histogram(self) -> Path:
        """Histogram of bounding box width, height, and area distributions."""
        bw = self.stats.get("bbox_width_stats", {})
        bh = self.stats.get("bbox_height_stats", {})
        ba = self.stats.get("bbox_area_stats", {})

        if not bw:
            log.warning("No bbox stats for histogram")
            return self.output_dir / "bbox_histogram.png"

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        for ax, stat, label in zip(axes, [bw, bh, ba], ["Width", "Height", "Area"]):
            # Reconstruct approximate distribution from percentiles
            pts = [stat.get(k, 0) for k in ["min", "p25", "p50", "p75", "max"]]
            ax.bar(
                ["min", "p25", "p50", "p75", "max"], pts,
                color="#FF6B35", edgecolor="none"
            )
            ax.set_title(f"BBox {label} (normalised)")
            ax.set_ylabel("Value")
            ax.grid(axis="y", alpha=0.3)
            mean_val = stat.get("mean", 0)
            ax.axhline(mean_val, color="blue", linestyle="--", label=f"mean={mean_val:.3f}")
            ax.legend(fontsize=8)

        fig.suptitle("Bounding Box Size Distribution")
        plt.tight_layout()
        out = self.output_dir / "bbox_histogram.png"
        plt.savefig(str(out), dpi=150)
        plt.close()
        log.info(f"BBox histogram → {out}")
        return out

    def resolution_scatter(self) -> Path:
        """Scatter plot of image widths vs heights."""
        iw = self.stats.get("image_width_stats", {})
        ih = self.stats.get("image_height_stats", {})

        if not iw:
            log.warning("No resolution stats for scatter")
            return self.output_dir / "resolution_scatter.png"

        # Simulate distribution from stats (mean ± std, approximate)
        np.random.seed(42)
        n = min(self.stats.get("total_images", 500), 500)
        w_mean, w_std = iw.get("mean", 640), iw.get("std", 100)
        h_mean, h_std = ih.get("mean", 640), ih.get("std", 100)
        ws = np.clip(np.random.normal(w_mean, w_std, n), iw.get("min", 64), iw.get("max", 4096))
        hs = np.clip(np.random.normal(h_mean, h_std, n), ih.get("min", 64), ih.get("max", 4096))

        fig, ax = plt.subplots(figsize=(7, 6))
        ax.scatter(ws, hs, alpha=0.3, s=10, color="#4C8BF5")
        ax.axvline(w_mean, color="red", linestyle="--", alpha=0.7, label=f"mean W={w_mean:.0f}")
        ax.axhline(h_mean, color="orange", linestyle="--", alpha=0.7, label=f"mean H={h_mean:.0f}")
        ax.set_xlabel("Image Width (px)")
        ax.set_ylabel("Image Height (px)")
        ax.set_title("Image Resolution Distribution")
        ax.legend()
        ax.grid(alpha=0.3)
        plt.tight_layout()
        out = self.output_dir / "resolution_scatter.png"
        plt.savefig(str(out), dpi=150)
        plt.close()
        log.info(f"Resolution scatter → {out}")
        return out

    def dataset_composition(self) -> Path:
        """Pie chart of annotation count per source dataset."""
        per_source = self.stats.get("per_source") or {}
        if not per_source:
            # Use group coverage as fallback
            gc = self.stats.get("group_coverage", {})
            if not gc:
                return self.output_dir / "dataset_composition.png"
            labels = list(gc.keys())
            sizes = [gc[g]["annotations"] for g in labels]
        else:
            labels = list(per_source.keys())
            sizes = [per_source[s].get("annotations", 0) for s in labels]

        # Remove zero slices
        filtered = [(l, s) for l, s in zip(labels, sizes) if s > 0]
        if not filtered:
            return self.output_dir / "dataset_composition.png"
        labels, sizes = zip(*filtered)

        fig, ax = plt.subplots(figsize=(9, 6))
        wedges, texts, autotexts = ax.pie(
            sizes, labels=labels, autopct="%1.1f%%",
            colors=list(_COLORS[:len(labels)]), startangle=140,
        )
        for at in autotexts:
            at.set_fontsize(8)
        ax.set_title("Dataset Composition (by annotations)")
        plt.tight_layout()
        out = self.output_dir / "dataset_composition.png"
        plt.savefig(str(out), dpi=150)
        plt.close()
        log.info(f"Dataset composition → {out}")
        return out

    def split_distribution(self) -> Path:
        """Grouped bar chart of train/val/test sizes."""
        img_counts = self.stats.get("image_counts", {})
        ann_counts = self.stats.get("annotation_counts", {})

        if not img_counts:
            # Try from splits field
            splits_data = self.stats.get("splits", {})
            if not splits_data:
                return self.output_dir / "split_distribution.png"
            img_counts = splits_data

        splits = ["train", "val", "test"]
        img_vals = [img_counts.get(s, 0) for s in splits]
        ann_vals = [ann_counts.get(s, 0) for s in splits]

        x = np.arange(len(splits))
        width = 0.35
        fig, ax = plt.subplots(figsize=(7, 5))
        bars1 = ax.bar(x - width / 2, img_vals, width, label="Images", color="#4C8BF5")
        bars2 = ax.bar(x + width / 2, ann_vals, width, label="Annotations", color="#FF6B35")
        ax.set_xticks(x)
        ax.set_xticklabels([s.capitalize() for s in splits])
        ax.set_ylabel("Count")
        ax.set_title("Train / Validation / Test Distribution")
        ax.legend()
        ax.bar_label(bars1, padding=3, fontsize=8)
        ax.bar_label(bars2, padding=3, fontsize=8)
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        out = self.output_dir / "split_distribution.png"
        plt.savefig(str(out), dpi=150)
        plt.close()
        log.info(f"Split distribution → {out}")
        return out

    def group_coverage(self) -> Path:
        """Bar chart of annotation count per SPIRO group."""
        gc = self.stats.get("group_coverage", {})
        if not gc:
            return self.output_dir / "group_coverage.png"

        groups = list(gc.keys())
        coverages = [gc[g]["coverage_pct"] for g in groups]
        annotations = [gc[g]["annotations"] for g in groups]

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 9))

        # Coverage %
        colors = ["#2ECC71" if c >= 75 else "#F39C12" if c >= 40 else "#E74C3C" for c in coverages]
        bars = ax1.bar(groups, coverages, color=colors)
        ax1.set_ylabel("Class Coverage (%)")
        ax1.set_title("SPIRO Group Coverage")
        ax1.set_xticklabels(groups, rotation=45, ha="right", fontsize=8)
        ax1.axhline(100, color="green", linestyle="--", alpha=0.3)
        ax1.set_ylim(0, 110)
        ax1.bar_label(bars, fmt="%.0f%%", padding=2, fontsize=7)
        ax1.grid(axis="y", alpha=0.3)

        # Annotations
        ax2.bar(groups, annotations, color="#4C8BF5")
        ax2.set_ylabel("Annotation count")
        ax2.set_title("Annotations per SPIRO Group")
        ax2.set_xticklabels(groups, rotation=45, ha="right", fontsize=8)
        ax2.grid(axis="y", alpha=0.3)

        plt.tight_layout()
        out = self.output_dir / "group_coverage.png"
        plt.savefig(str(out), dpi=150)
        plt.close()
        log.info(f"Group coverage → {out}")
        return out

    def imbalance_heatmap(self) -> Path:
        """Heatmap of class annotations organised by taxonomy group."""
        dist = self.stats.get("class_distribution", {})
        if not dist:
            return self.output_dir / "imbalance_heatmap.png"

        groups = self.taxonomy.group_names()
        # Build matrix: groups × max_per_group
        max_cols = max(len(self.taxonomy.group_ids(g)) for g in groups)
        matrix = np.zeros((len(groups), max_cols))
        row_labels = []
        col_labels = [str(i) for i in range(max_cols)]

        for row_idx, group in enumerate(groups):
            row_labels.append(group)
            for col_idx, cls_id in enumerate(self.taxonomy.group_ids(group)):
                name = self.taxonomy.id_to_name(cls_id)
                matrix[row_idx, col_idx] = dist.get(name, 0)

        fig, ax = plt.subplots(figsize=(max(10, max_cols * 0.6), max(6, len(groups) * 0.5)))
        im = ax.imshow(np.log1p(matrix), cmap="YlOrRd", aspect="auto")
        ax.set_yticks(range(len(groups)))
        ax.set_yticklabels(row_labels, fontsize=8)
        ax.set_xlabel("Class index within group")
        ax.set_title("Class Imbalance Heatmap (log scale)")
        plt.colorbar(im, ax=ax, label="log(1 + count)")
        plt.tight_layout()
        out = self.output_dir / "imbalance_heatmap.png"
        plt.savefig(str(out), dpi=150)
        plt.close()
        log.info(f"Imbalance heatmap → {out}")
        return out


# =============================================================================
# Utility: draw YOLO boxes on image
# =============================================================================

def draw_yolo_boxes(
    image: np.ndarray,
    bboxes: List[Tuple[float, float, float, float]],
    class_ids: List[int],
    class_names: Optional[List[str]] = None,
) -> np.ndarray:
    """
    Draw YOLO-format bboxes on an image (RGB or BGR).

    Parameters
    ----------
    image : np.ndarray
        H×W×3 image.
    bboxes : list of (cx, cy, w, h) normalised tuples.
    class_ids : list of int
    class_names : list of str, optional

    Returns
    -------
    np.ndarray — annotated image copy (BGR)
    """
    canvas = image.copy()
    if canvas.shape[2] == 3 and canvas.dtype == np.uint8:
        if canvas.max() <= 1:
            canvas = (canvas * 255).astype(np.uint8)

    ih, iw = canvas.shape[:2]
    cmap = [
        (0, 114, 189), (217, 83, 25), (237, 177, 32),
        (126, 47, 142), (119, 172, 48), (77, 190, 238),
    ]

    for (cx, cy, w, h), cls_id in zip(bboxes, class_ids):
        x1 = int((cx - w / 2) * iw)
        y1 = int((cy - h / 2) * ih)
        x2 = int((cx + w / 2) * iw)
        y2 = int((cy + h / 2) * ih)
        color = cmap[cls_id % len(cmap)]
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        label = class_names[cls_id] if class_names and cls_id < len(class_names) else str(cls_id)
        cv2.putText(canvas, label, (x1, max(y1 - 4, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

    return cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR) if image.dtype == np.uint8 else canvas
