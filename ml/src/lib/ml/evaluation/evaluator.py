"""
SPIRO ML — Evaluator
End-to-end evaluation pipeline for both detection and classification models.
Produces:
  - mAP50 / mAP50-95 (detection)
  - Precision, Recall, F1 per class
  - Confusion matrix (PNG)
  - PR curve (PNG)
  - Per-class bar chart (PNG)
  - JSON report
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
)

from lib.ml.core.config import ConfigManager
from lib.ml.core.logger import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Detection evaluation (YOLO)
# ---------------------------------------------------------------------------


class DetectionEvaluator:
    """
    Evaluates YOLOv11 detection results.
    Wraps Ultralytics val() and adds custom plots + JSON report.

    Example
    -------
    >>> ev = DetectionEvaluator(cfg)
    >>> report = ev.evaluate(weights_path="models/checkpoints/best.pt")
    """

    def __init__(self, cfg: ConfigManager) -> None:
        self.cfg = cfg
        self.reports_dir = Path("reports")
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self.class_names = list(cfg.dataset.class_names)

    def evaluate(
        self,
        weights_path: Union[str, Path],
        split: str = "test",
        save_report: bool = True,
    ) -> Dict[str, Any]:
        """
        Run Ultralytics val() on the test split and generate reports.

        Parameters
        ----------
        weights_path : Path
            Path to .pt weights file.
        split : str
            "val" or "test"
        save_report : bool
            Write JSON + plots to reports/.

        Returns
        -------
        dict with all metrics.
        """
        from ultralytics import YOLO
        from lib.ml.core.device import resolve_device

        device = resolve_device(self.cfg.training.device)
        model = YOLO(str(weights_path))

        log.info(f"Evaluating on {split} split — weights: {weights_path}")
        metrics = model.val(
            data=str(Path("configs/dataset.yaml").resolve()),
            split=split,
            imgsz=self.cfg.model.input_size[0],
            conf=self.cfg.evaluation.conf_threshold,
            iou=self.cfg.evaluation.iou_threshold,
            device=str(device) if device.type != "cpu" else "cpu",
            plots=self.cfg.evaluation.save_plots,
            save_json=True,
            verbose=True,
        )

        report: Dict[str, Any] = {
            "split": split,
            "weights": str(weights_path),
            "conf_threshold": self.cfg.evaluation.conf_threshold,
            "iou_threshold": self.cfg.evaluation.iou_threshold,
            "mAP50": float(metrics.box.map50),
            "mAP50-95": float(metrics.box.map),
            "precision": float(metrics.box.mp),
            "recall": float(metrics.box.mr),
        }

        # Per-class metrics
        if hasattr(metrics.box, "ap_class_index") and metrics.box.ap_class_index is not None:
            per_class = {}
            for i, cls_idx in enumerate(metrics.box.ap_class_index):
                name = self.class_names[cls_idx] if cls_idx < len(self.class_names) else str(cls_idx)
                per_class[name] = {
                    "AP50": float(metrics.box.ap50[i]) if i < len(metrics.box.ap50) else 0.0,
                }
            report["per_class"] = per_class
            self._plot_per_class_ap(per_class, self.reports_dir / f"{split}_per_class_ap.png")

        if save_report:
            out = self.reports_dir / f"{split}_detection_report.json"
            with open(out, "w") as f:
                json.dump(report, f, indent=2)
            log.info(f"Detection report saved to {out}")

        log.info(
            f"mAP50={report['mAP50']:.4f}  mAP50-95={report['mAP50-95']:.4f}  "
            f"P={report['precision']:.4f}  R={report['recall']:.4f}"
        )
        return report

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------

    def _plot_per_class_ap(self, per_class: Dict, out: Path) -> None:
        names = list(per_class.keys())
        values = [v["AP50"] for v in per_class.values()]
        fig, ax = plt.subplots(figsize=(max(6, len(names) * 1.2), 5))
        bars = ax.bar(names, values, color="#4C8BF5")
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("AP@50")
        ax.set_title("Per-Class AP@50")
        ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=8)
        plt.tight_layout()
        plt.savefig(str(out), dpi=150)
        plt.close()
        log.info(f"Per-class AP plot saved: {out}")


# ---------------------------------------------------------------------------
# Classification evaluation
# ---------------------------------------------------------------------------


class ClassificationEvaluator:
    """
    Evaluates SPIROClassifier with full sklearn metrics.

    Example
    -------
    >>> ev = ClassificationEvaluator(cfg)
    >>> y_true = [0, 1, 2, ...]
    >>> y_pred = [0, 1, 1, ...]
    >>> y_proba = [[0.9, 0.05, ...], ...]
    >>> report = ev.evaluate(y_true, y_pred, y_proba)
    """

    def __init__(self, cfg: ConfigManager) -> None:
        self.cfg = cfg
        self.class_names = list(cfg.dataset.class_names)
        self.reports_dir = Path("reports")
        self.reports_dir.mkdir(parents=True, exist_ok=True)

    def evaluate(
        self,
        y_true: List[int],
        y_pred: List[int],
        y_proba: Optional[List[List[float]]] = None,
        save_report: bool = True,
    ) -> Dict[str, Any]:
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        report = classification_report(
            y_true, y_pred,
            target_names=self.class_names,
            output_dict=True,
            zero_division=0,
        )

        result: Dict[str, Any] = {
            "classification_report": report,
            "accuracy": float((y_true == y_pred).mean()),
        }

        if y_proba is not None:
            y_proba_arr = np.array(y_proba)
            try:
                roc_auc = roc_auc_score(
                    y_true, y_proba_arr, multi_class="ovr", average="macro"
                )
                result["roc_auc_macro"] = float(roc_auc)
            except Exception:
                pass

        if save_report:
            # Confusion matrix plot
            self._plot_confusion_matrix(
                y_true, y_pred,
                self.reports_dir / "classification_confusion_matrix.png",
            )
            # PR curves
            if y_proba is not None:
                self._plot_pr_curves(
                    y_true, np.array(y_proba),
                    self.reports_dir / "classification_pr_curves.png",
                )
            out = self.reports_dir / "classification_report.json"
            with open(out, "w") as f:
                json.dump(result, f, indent=2)
            log.info(f"Classification report saved to {out}")

        log.info(f"Accuracy: {result['accuracy']:.4f}")
        return result

    def _plot_confusion_matrix(
        self, y_true: np.ndarray, y_pred: np.ndarray, out: Path
    ) -> None:
        cm = confusion_matrix(y_true, y_pred)
        n = len(self.class_names)
        fig, ax = plt.subplots(figsize=(max(6, n), max(5, n - 1)))
        im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
        plt.colorbar(im, ax=ax)
        ax.set_xticks(np.arange(n))
        ax.set_yticks(np.arange(n))
        ax.set_xticklabels(self.class_names, rotation=45, ha="right")
        ax.set_yticklabels(self.class_names)
        thresh = cm.max() / 2.0
        for i in range(n):
            for j in range(n):
                ax.text(j, i, str(cm[i, j]),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black",
                        fontsize=9)
        ax.set_ylabel("True label")
        ax.set_xlabel("Predicted label")
        ax.set_title("Confusion Matrix")
        plt.tight_layout()
        plt.savefig(str(out), dpi=150)
        plt.close()
        log.info(f"Confusion matrix saved: {out}")

    def _plot_pr_curves(
        self, y_true: np.ndarray, y_proba: np.ndarray, out: Path
    ) -> None:
        n = len(self.class_names)
        fig, ax = plt.subplots(figsize=(8, 6))
        colors = plt.cm.tab10(np.linspace(0, 1, n))
        for i, (name, color) in enumerate(zip(self.class_names, colors)):
            y_bin = (y_true == i).astype(int)
            if y_bin.sum() == 0:
                continue
            prec, rec, _ = precision_recall_curve(y_bin, y_proba[:, i])
            ap = average_precision_score(y_bin, y_proba[:, i])
            ax.plot(rec, prec, color=color, label=f"{name} (AP={ap:.2f})")
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title("Precision-Recall Curves")
        ax.legend(loc="lower left", fontsize=8)
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(str(out), dpi=150)
        plt.close()
        log.info(f"PR curves saved: {out}")
