"""
SPIRO ML — VerifyMetrics
Classification metrics for EfficientNetV2 verifier:
  - Top-1 / Top-5 Accuracy
  - Precision / Recall / F1 (macro, weighted, per-class)
  - ROC-AUC (macro OVR)
  - Confusion matrix
  - Classification report JSON
  - Per-class accuracy
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    top_k_accuracy_score,
)

from lib.ml.core.logger import get_logger
from lib.ml.dataset_engineering.taxonomy import TaxonomyLoader

log = get_logger(__name__)


class VerifyMetrics:
    """
    Computes and stores all classification metrics for a verification run.

    Parameters
    ----------
    num_classes : int
    class_names : list of str
    output_dir : Path
        Where to save plots and JSON reports.
    """

    def __init__(
        self,
        num_classes: int = 109,
        class_names: Optional[List[str]] = None,
        output_dir: Path = Path("reports/verification"),
    ) -> None:
        self.num_classes = num_classes
        self.taxonomy = TaxonomyLoader()
        self.class_names = class_names or self.taxonomy.all_names()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Static fast metrics (used during training loop)
    # ------------------------------------------------------------------

    @staticmethod
    def compute_basic(
        logits: torch.Tensor,
        labels: torch.Tensor,
        top_k: int = 5,
    ) -> Dict[str, float]:
        """
        Fast per-batch metrics (no sklearn) suitable for the training loop.

        Parameters
        ----------
        logits : Tensor [N, C]
        labels : Tensor [N]
        top_k : int

        Returns
        -------
        dict with top1_accuracy, top5_accuracy
        """
        with torch.no_grad():
            probs = F.softmax(logits, dim=1)
            top1_correct = (logits.argmax(1) == labels).float().mean().item()

            k = min(top_k, logits.size(1))
            top_k_preds = logits.topk(k, dim=1).indices
            top_k_correct = (
                top_k_preds == labels.unsqueeze(1)
            ).any(dim=1).float().mean().item()

        return {
            "top1_accuracy": float(top1_correct),
            f"top{k}_accuracy": float(top_k_correct),
        }

    # ------------------------------------------------------------------
    # Full evaluation (post-training)
    # ------------------------------------------------------------------

    def evaluate(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        split: str = "test",
        save_reports: bool = True,
    ) -> Dict[str, Any]:
        """
        Full classification metrics suite.

        Parameters
        ----------
        logits : Tensor [N, C]
        labels : Tensor [N] — ground-truth class IDs
        split : str
        save_reports : bool — write JSON + plots

        Returns
        -------
        dict with all metrics
        """
        probs = F.softmax(logits, dim=1).cpu().numpy()
        y_pred = np.argmax(probs, axis=1)
        y_true = labels.cpu().numpy()

        nc = min(self.num_classes, probs.shape[1])
        names_used = self.class_names[:nc]

        # ── Core metrics ──────────────────────────────────────────────
        top1 = float((y_pred == y_true).mean())
        top5 = float(
            top_k_accuracy_score(y_true, probs, k=min(5, nc))
            if probs.shape[1] >= 5 else top1
        )

        present = sorted(set(y_true.tolist()))
        present_names = [self.class_names[i] for i in present if i < len(self.class_names)]

        precision = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
        recall    = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
        f1        = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
        f1_w      = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

        # ROC-AUC (only for classes present in y_true)
        roc_auc = None
        try:
            probs_present = probs[:, present]
            roc_auc = float(roc_auc_score(
                y_true, probs_present, multi_class="ovr",
                labels=present, average="macro",
            ))
        except Exception as e:
            log.debug(f"ROC-AUC failed: {e}")

        # Per-class accuracy
        per_class_acc: Dict[str, float] = {}
        for cls_id in present:
            mask = y_true == cls_id
            if mask.sum() == 0:
                continue
            name = self.class_names[cls_id] if cls_id < len(self.class_names) else str(cls_id)
            per_class_acc[name] = float((y_pred[mask] == cls_id).mean())

        # Classification report
        clf_report = classification_report(
            y_true, y_pred,
            labels=present,
            target_names=present_names,
            output_dict=True,
            zero_division=0,
        )

        result: Dict[str, Any] = {
            "split": split,
            "top1_accuracy": top1,
            "top5_accuracy": top5,
            "precision_macro": precision,
            "recall_macro": recall,
            "f1_macro": f1,
            "f1_weighted": f1_w,
            "roc_auc_macro": roc_auc,
            "per_class_accuracy": per_class_acc,
            "classification_report": clf_report,
            "num_samples": int(len(y_true)),
            "classes_evaluated": len(present),
        }

        log.info(
            f"[{split}] top1={top1:.4f} top5={top5:.4f} "
            f"P={precision:.4f} R={recall:.4f} F1={f1:.4f} "
            f"ROC-AUC={roc_auc:.4f if roc_auc else 'N/A'}"
        )

        if save_reports:
            self._save_classification_report(result, split)
            self._plot_confusion_matrix(y_true, y_pred, present, present_names, split)
            self._plot_per_class_accuracy(per_class_acc, split)
            if roc_auc is not None:
                self._plot_roc_curves(y_true, probs, present, present_names, split)

        return result

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------

    def _save_classification_report(self, result: Dict, split: str) -> None:
        out = self.output_dir / f"{split}_classification_report.json"
        with open(out, "w") as f:
            json.dump(result, f, indent=2, default=str)
        log.info(f"Classification report → {out}")

    def _plot_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        classes: List[int],
        names: List[str],
        split: str,
    ) -> Path:
        cm = confusion_matrix(y_true, y_pred, labels=classes)
        n = len(classes)
        figsize = max(8, n * 0.35)
        fig, ax = plt.subplots(figsize=(figsize, figsize * 0.85))
        im = ax.imshow(cm, cmap="Blues")
        plt.colorbar(im, ax=ax, fraction=0.04)

        tick_labels = names if n <= 30 else [f"C{i}" for i in classes]
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(tick_labels, rotation=90, fontsize=max(4, 7 - n // 20))
        ax.set_yticklabels(tick_labels, fontsize=max(4, 7 - n // 20))

        if n <= 20:
            thresh = cm.max() / 2.0
            for i in range(n):
                for j in range(n):
                    ax.text(j, i, str(cm[i, j]),
                            ha="center", va="center", fontsize=6,
                            color="white" if cm[i, j] > thresh else "black")

        ax.set_ylabel("True label")
        ax.set_xlabel("Predicted label")
        ax.set_title(f"Confusion Matrix — {split}")
        plt.tight_layout()
        out = self.output_dir / f"{split}_confusion_matrix.png"
        plt.savefig(str(out), dpi=150)
        plt.close()
        log.info(f"Confusion matrix → {out}")
        return out

    def _plot_per_class_accuracy(self, per_class: Dict[str, float], split: str) -> Path:
        if not per_class:
            return self.output_dir / "empty.png"
        names = list(per_class.keys())
        accs = [per_class[n] for n in names]
        order = np.argsort(accs)
        names = [names[i] for i in order]
        accs = [accs[i] for i in order]

        fig, ax = plt.subplots(figsize=(10, max(4, len(names) * 0.2)))
        colors = ["#2ECC71" if a >= 0.7 else "#F39C12" if a >= 0.4 else "#E74C3C" for a in accs]
        ax.barh(names, accs, color=colors)
        ax.set_xlim(0, 1.05)
        ax.set_xlabel("Top-1 Accuracy")
        ax.set_title(f"Per-Class Accuracy — {split}")
        ax.axvline(0.5, color="gray", linestyle="--", alpha=0.5)
        plt.tight_layout()
        out = self.output_dir / f"{split}_per_class_accuracy.png"
        plt.savefig(str(out), dpi=150, bbox_inches="tight")
        plt.close()
        log.info(f"Per-class accuracy → {out}")
        return out

    def _plot_roc_curves(
        self,
        y_true: np.ndarray,
        probs: np.ndarray,
        classes: List[int],
        names: List[str],
        split: str,
    ) -> Path:
        from sklearn.metrics import roc_curve
        from sklearn.preprocessing import label_binarize

        y_bin = label_binarize(y_true, classes=classes)
        fig, ax = plt.subplots(figsize=(8, 6))
        n_show = min(15, len(classes))
        cmap = plt.cm.tab20(np.linspace(0, 1, n_show))

        for i, (cls_idx, name, color) in enumerate(
            zip(classes[:n_show], names[:n_show], cmap)
        ):
            fpr, tpr, _ = roc_curve(y_bin[:, i], probs[:, cls_idx])
            ap = average_precision_score(y_bin[:, i], probs[:, cls_idx])
            ax.plot(fpr, tpr, color=color, label=f"{name} (AP={ap:.2f})", linewidth=1.2)

        ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(f"ROC Curves (top {n_show} classes) — {split}")
        ax.legend(loc="lower right", fontsize=6, ncol=2)
        ax.grid(alpha=0.3)
        plt.tight_layout()
        out = self.output_dir / f"{split}_roc_curves.png"
        plt.savefig(str(out), dpi=150)
        plt.close()
        log.info(f"ROC curves → {out}")
        return out

    def plot_training_curves(self, history: Dict[str, List[float]]) -> List[Path]:
        """Plot accuracy and loss curves from training history."""
        outputs = []

        # Accuracy curve
        fig, ax = plt.subplots(figsize=(9, 5))
        if "train_acc" in history:
            ax.plot(history["train_acc"], label="train top-1", color="#4C8BF5")
        if "val_acc" in history:
            ax.plot(history["val_acc"], label="val top-1", color="#FF6B35", linestyle="--")
        if "val_top5_acc" in history:
            ax.plot(history["val_top5_acc"], label="val top-5", color="#2ECC71", linestyle=":")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Accuracy")
        ax.set_title("EfficientNetV2 Accuracy Curves")
        ax.legend()
        ax.grid(alpha=0.3)
        ax.set_ylim(0, 1.05)
        out = self.output_dir / "accuracy_curve.png"
        plt.savefig(str(out), dpi=150)
        plt.close()
        outputs.append(out)

        # Loss curve
        fig, ax = plt.subplots(figsize=(9, 5))
        if "train_loss" in history:
            ax.plot(history["train_loss"], label="train loss", color="#4C8BF5")
        if "val_loss" in history:
            ax.plot(history["val_loss"], label="val loss", color="#FF6B35", linestyle="--")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title("EfficientNetV2 Loss Curves")
        ax.legend()
        ax.grid(alpha=0.3)
        out2 = self.output_dir / "loss_curve.png"
        plt.savefig(str(out2), dpi=150)
        plt.close()
        outputs.append(out2)
        return outputs
