"""
SPIRO ML — ResultsPlotter
Generates training artefact plots from CSV history and Ultralytics results:
  - Loss curves (train + val box/cls/dfl)
  - mAP curve over epochs
  - Precision and recall curves
  - PR curve (precision vs recall)
  - Confusion matrix (from Ultralytics output)
  - Combined results.png dashboard
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from lib.ml.core.logger import get_logger

log = get_logger(__name__)


class ResultsPlotter:
    """
    Generates plots from training_history.csv and Ultralytics run directory.

    Example
    -------
    >>> plotter = ResultsPlotter(
    ...     csv_path=Path("logs/training_history.csv"),
    ...     run_dir=Path("models/checkpoints/spiro_yolo11s"),
    ...     output_dir=Path("reports/training"),
    ... )
    >>> plotter.generate_all()
    """

    def __init__(
        self,
        csv_path: Path,
        run_dir: Optional[Path] = None,
        output_dir: Optional[Path] = None,
    ) -> None:
        self.csv_path = Path(csv_path)
        self.run_dir = Path(run_dir) if run_dir else None
        self.output_dir = Path(output_dir) if output_dir else self.csv_path.parent / "plots"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._df: Optional[pd.DataFrame] = None

    # ------------------------------------------------------------------
    # Main entry
    # ------------------------------------------------------------------

    def generate_all(self) -> List[Path]:
        """Generate all plots. Returns list of created file paths."""
        self._df = self._load_csv()
        outputs: List[Path] = []

        if self._df is not None and not self._df.empty:
            outputs.append(self.plot_loss_curves())
            outputs.append(self.plot_map_curve())
            outputs.append(self.plot_precision_recall())
            outputs.append(self.plot_results_dashboard())

        # Copy Ultralytics-generated plots from run_dir
        if self.run_dir:
            outputs.extend(self._copy_ultralytics_plots())

        log.info(f"Generated {len(outputs)} plots in {self.output_dir}")
        return outputs

    # ------------------------------------------------------------------
    # Individual plots
    # ------------------------------------------------------------------

    def plot_loss_curves(self) -> Path:
        """Plot train and val loss components over epochs."""
        df = self._df
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        components = [
            ("box_loss", "Box Loss"),
            ("cls_loss", "Classification Loss"),
            ("dfl_loss", "DFL Loss"),
        ]
        for ax, (comp, title) in zip(axes, components):
            train_col = f"train/{comp}"
            val_col = f"val/{comp}"
            if train_col in df.columns:
                ax.plot(df["epoch"], df[train_col], label="train", color="#4C8BF5")
            if val_col in df.columns:
                ax.plot(df["epoch"], df[val_col], label="val", color="#FF6B35", linestyle="--")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss")
            ax.set_title(title)
            ax.legend()
            ax.grid(alpha=0.3)
        fig.suptitle("Training Loss Curves", fontsize=13, y=1.02)
        plt.tight_layout()
        out = self.output_dir / "loss_curves.png"
        plt.savefig(str(out), dpi=150, bbox_inches="tight")
        plt.close()
        log.info(f"Loss curves → {out}")
        return out

    def plot_map_curve(self) -> Path:
        """Plot mAP50 and mAP50-95 over epochs."""
        df = self._df
        fig, ax = plt.subplots(figsize=(9, 5))

        if "mAP50" in df.columns:
            ax.plot(df["epoch"], df["mAP50"], label="mAP@50", color="#2ECC71", linewidth=2)
        if "mAP50-95" in df.columns:
            ax.plot(df["epoch"], df["mAP50-95"], label="mAP@50-95",
                    color="#3498DB", linewidth=2, linestyle="--")

        if "mAP50-95" in df.columns and not df["mAP50-95"].isna().all():
            best_idx = df["mAP50-95"].idxmax()
            best_epoch = df.loc[best_idx, "epoch"]
            best_val = df.loc[best_idx, "mAP50-95"]
            ax.axvline(best_epoch, color="red", linestyle=":", alpha=0.6)
            ax.annotate(
                f"Best: {best_val:.4f}\n(epoch {best_epoch})",
                xy=(best_epoch, best_val),
                xytext=(best_epoch + max(1, len(df) * 0.05), best_val - 0.05),
                arrowprops=dict(arrowstyle="->", color="red"),
                color="red", fontsize=9,
            )

        ax.set_xlabel("Epoch")
        ax.set_ylabel("mAP")
        ax.set_title("mAP over Training Epochs")
        ax.legend()
        ax.grid(alpha=0.3)
        ax.set_ylim(0, max(1.05, df.get("mAP50", pd.Series([0])).max() * 1.1))
        plt.tight_layout()
        out = self.output_dir / "map_curve.png"
        plt.savefig(str(out), dpi=150)
        plt.close()
        log.info(f"mAP curve → {out}")
        return out

    def plot_precision_recall(self) -> Path:
        """Plot precision and recall over epochs."""
        df = self._df
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        for ax, col, title, color in [
            (axes[0], "precision", "Precision", "#9B59B6"),
            (axes[1], "recall", "Recall", "#E67E22"),
        ]:
            if col in df.columns:
                ax.plot(df["epoch"], df[col], color=color, linewidth=2)
                ax.set_xlabel("Epoch")
                ax.set_ylabel(title)
                ax.set_title(f"{title} over Epochs")
                ax.set_ylim(0, 1.05)
                ax.grid(alpha=0.3)
                ax.fill_between(df["epoch"], df[col], alpha=0.15, color=color)

        plt.tight_layout()
        out = self.output_dir / "precision_recall_curves.png"
        plt.savefig(str(out), dpi=150)
        plt.close()
        log.info(f"Precision/recall curves → {out}")
        return out

    def plot_results_dashboard(self) -> Path:
        """Combined 3×2 dashboard: loss + metrics + lr."""
        df = self._df
        fig = plt.figure(figsize=(18, 10))
        fig.suptitle("SPIRO YOLOv11 Training Results", fontsize=14, y=1.01)

        panels = [
            ("train/box_loss",         "Train Box Loss",        "#4C8BF5"),
            ("val/box_loss",           "Val Box Loss",          "#FF6B35"),
            ("mAP50",                  "mAP@50",                "#2ECC71"),
            ("mAP50-95",               "mAP@50-95",             "#3498DB"),
            ("precision",              "Precision",             "#9B59B6"),
            ("recall",                 "Recall",                "#E67E22"),
        ]

        for i, (col, title, color) in enumerate(panels, 1):
            ax = fig.add_subplot(2, 3, i)
            if col in df.columns and not df[col].isna().all():
                ax.plot(df["epoch"], df[col], color=color, linewidth=1.5)
                ax.fill_between(df["epoch"], df[col], alpha=0.1, color=color)
            ax.set_title(title, fontsize=10)
            ax.set_xlabel("Epoch", fontsize=8)
            ax.grid(alpha=0.25)
            ax.tick_params(labelsize=7)

        plt.tight_layout()
        out = self.output_dir / "results.png"
        plt.savefig(str(out), dpi=150, bbox_inches="tight")
        plt.close()
        log.info(f"Dashboard → {out}")
        return out

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _load_csv(self) -> Optional[pd.DataFrame]:
        if not self.csv_path.exists():
            log.warning(f"CSV not found: {self.csv_path}")
            return None
        try:
            df = pd.read_csv(self.csv_path)
            # Strip whitespace from column names
            df.columns = [c.strip() for c in df.columns]
            return df
        except Exception as e:
            log.error(f"Failed to load CSV {self.csv_path}: {e}")
            return None

    def _copy_ultralytics_plots(self) -> List[Path]:
        """Copy plots generated by Ultralytics to output_dir."""
        copied = []
        if not self.run_dir:
            return copied
        for pattern in ["confusion_matrix.png", "confusion_matrix_normalized.png",
                         "PR_curve.png", "P_curve.png", "R_curve.png",
                         "F1_curve.png", "results.png"]:
            src = self.run_dir / pattern
            if src.exists():
                import shutil
                dst = self.output_dir / f"yolo_{pattern}"
                shutil.copy2(src, dst)
                copied.append(dst)
        return copied
