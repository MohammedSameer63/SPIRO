"""
SPIRO ML — Training Callbacks
Ultralytics-compatible callback hooks for:
  - TensorBoard metric/image logging
  - CSV training history
  - Early stopping
  - GPU memory tracking
  - Per-epoch console summary
  - Checkpoint management integration

All callbacks follow the Ultralytics callback signature:
    callback(trainer)  →  None
"""
from __future__ import annotations

import csv
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import torch

from lib.ml.core.logger import get_logger

log = get_logger(__name__)


# =============================================================================
# TensorBoard callback
# =============================================================================

class TensorBoardCallback:
    """
    Writes training metrics to TensorBoard after each epoch.

    Registered as:
      on_train_epoch_end
      on_val_end
      on_fit_epoch_end
    """

    def __init__(self, log_dir: Path) -> None:
        from torch.utils.tensorboard import SummaryWriter
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir=str(self.log_dir))
        log.info(f"TensorBoard: {self.log_dir}")

    def on_fit_epoch_end(self, trainer) -> None:
        """Called by Ultralytics at the end of each fit (train+val) epoch."""
        metrics = trainer.metrics
        epoch = trainer.epoch + 1

        # Loss scalars
        for key, val in trainer.label_loss_items(trainer.tloss, prefix="train").items():
            self.writer.add_scalar(f"Loss/train/{key}", float(val), epoch)

        # Metric scalars
        metric_map = {
            "metrics/precision(B)": "Metrics/Precision",
            "metrics/recall(B)":    "Metrics/Recall",
            "metrics/mAP50(B)":     "Metrics/mAP50",
            "metrics/mAP50-95(B)":  "Metrics/mAP50-95",
            "val/box_loss":         "Loss/val/box",
            "val/cls_loss":         "Loss/val/cls",
            "val/dfl_loss":         "Loss/val/dfl",
        }
        for src_key, tb_key in metric_map.items():
            val = metrics.get(src_key)
            if val is not None:
                self.writer.add_scalar(tb_key, float(val), epoch)

        # Learning rate
        if hasattr(trainer, "optimizer") and trainer.optimizer:
            for i, pg in enumerate(trainer.optimizer.param_groups):
                self.writer.add_scalar(f"LR/pg{i}", pg["lr"], epoch)

        # GPU memory
        if torch.cuda.is_available():
            mem = torch.cuda.memory_reserved() / 1e9
            self.writer.add_scalar("System/GPU_mem_GB", mem, epoch)

        self.writer.flush()

    def close(self) -> None:
        self.writer.close()


# =============================================================================
# CSV logger callback
# =============================================================================

class CSVLoggerCallback:
    """
    Appends per-epoch metrics to a CSV file.

    Columns: epoch, train/box_loss, train/cls_loss, train/dfl_loss,
             val/box_loss, val/cls_loss, val/dfl_loss,
             precision, recall, mAP50, mAP50-95, lr, time_s, gpu_mem_GB
    """

    COLUMNS = [
        "epoch",
        "train/box_loss", "train/cls_loss", "train/dfl_loss",
        "val/box_loss",   "val/cls_loss",   "val/dfl_loss",
        "precision", "recall", "mAP50", "mAP50-95",
        "lr", "time_s", "gpu_mem_GB",
    ]

    def __init__(self, csv_path: Path) -> None:
        self.csv_path = Path(csv_path)
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        self._epoch_start: float = 0.0
        self._wrote_header = self.csv_path.exists()

    def on_train_epoch_start(self, trainer) -> None:
        self._epoch_start = time.perf_counter()

    def on_fit_epoch_end(self, trainer) -> None:
        elapsed = time.perf_counter() - self._epoch_start
        metrics = trainer.metrics
        epoch = trainer.epoch + 1

        def _g(key: str) -> float:
            v = metrics.get(key)
            return float(v) if v is not None else 0.0

        # Learning rate from first param group
        lr = 0.0
        if hasattr(trainer, "optimizer") and trainer.optimizer:
            lr = trainer.optimizer.param_groups[0].get("lr", 0.0)

        gpu_mem = 0.0
        if torch.cuda.is_available():
            gpu_mem = round(torch.cuda.memory_reserved() / 1e9, 3)

        # Fetch train losses (Ultralytics stores them in trainer.tloss)
        train_losses: Dict[str, float] = {}
        if hasattr(trainer, "tloss") and trainer.tloss is not None:
            labeled = trainer.label_loss_items(trainer.tloss, prefix="train")
            for k, v in labeled.items():
                short = k.replace("train/", "")
                train_losses[f"train/{short}"] = float(v)

        row = {
            "epoch": epoch,
            "train/box_loss": train_losses.get("train/box_loss", 0.0),
            "train/cls_loss": train_losses.get("train/cls_loss", 0.0),
            "train/dfl_loss": train_losses.get("train/dfl_loss", 0.0),
            "val/box_loss":   _g("val/box_loss"),
            "val/cls_loss":   _g("val/cls_loss"),
            "val/dfl_loss":   _g("val/dfl_loss"),
            "precision":      _g("metrics/precision(B)"),
            "recall":         _g("metrics/recall(B)"),
            "mAP50":          _g("metrics/mAP50(B)"),
            "mAP50-95":       _g("metrics/mAP50-95(B)"),
            "lr":             lr,
            "time_s":         round(elapsed, 2),
            "gpu_mem_GB":     gpu_mem,
        }

        with open(self.csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.COLUMNS)
            if not self._wrote_header:
                writer.writeheader()
                self._wrote_header = True
            writer.writerow(row)


# =============================================================================
# Early stopping callback
# =============================================================================

class EarlyStoppingCallback:
    """
    Monitors a validation metric and signals early stopping when
    no improvement is seen for `patience` epochs.

    Note: Ultralytics has built-in early stopping via patience=N in train().
    This callback provides an additional layer with custom metric selection
    and verbose logging.
    """

    def __init__(
        self,
        metric: str = "metrics/mAP50-95(B)",
        patience: int = 50,
        min_delta: float = 1e-4,
        maximize: bool = True,
    ) -> None:
        self.metric = metric
        self.patience = patience
        self.min_delta = min_delta
        self.maximize = maximize
        self._best: Optional[float] = None
        self._counter = 0
        self._stop = False

    def on_val_end(self, trainer) -> None:
        if self.patience <= 0:
            return

        val = trainer.metrics.get(self.metric)
        if val is None:
            return
        val = float(val)

        if self._best is None:
            self._best = val
            self._counter = 0
            return

        delta = val - self._best if self.maximize else self._best - val
        if delta > self.min_delta:
            self._best = val
            self._counter = 0
        else:
            self._counter += 1
            log.debug(
                f"EarlyStopping: no improvement for {self._counter}/{self.patience} epochs "
                f"({self.metric}={val:.4f}, best={self._best:.4f})"
            )
            if self._counter >= self.patience:
                log.info(
                    f"Early stopping triggered after {self.patience} epochs "
                    f"without improvement (best {self.metric}={self._best:.4f})"
                )
                trainer.stopper = lambda: True  # signal stop

    @property
    def stopped_early(self) -> bool:
        return self._counter >= self.patience


# =============================================================================
# Console epoch summary callback
# =============================================================================

class EpochSummaryCallback:
    """Prints a concise per-epoch summary to the console."""

    def __init__(self, total_epochs: int) -> None:
        self.total_epochs = total_epochs
        self._epoch_start = 0.0

    def on_train_epoch_start(self, trainer) -> None:
        self._epoch_start = time.perf_counter()

    def on_fit_epoch_end(self, trainer) -> None:
        elapsed = time.perf_counter() - self._epoch_start
        metrics = trainer.metrics
        epoch = trainer.epoch + 1

        def _f(key: str, digits: int = 4) -> str:
            v = metrics.get(key)
            return f"{float(v):.{digits}f}" if v is not None else "N/A"

        gpu_str = ""
        if torch.cuda.is_available():
            mem = torch.cuda.memory_reserved() / 1e9
            gpu_str = f" | GPU {mem:.1f}GB"

        lr = 0.0
        if hasattr(trainer, "optimizer") and trainer.optimizer:
            lr = trainer.optimizer.param_groups[0].get("lr", 0.0)

        log.info(
            f"Epoch [{epoch:>3}/{self.total_epochs}] "
            f"mAP50={_f('metrics/mAP50(B)', 4)} "
            f"mAP50-95={_f('metrics/mAP50-95(B)', 4)} "
            f"P={_f('metrics/precision(B)', 3)} "
            f"R={_f('metrics/recall(B)', 3)} "
            f"lr={lr:.2e} "
            f"| {elapsed:.1f}s{gpu_str}"
        )


# =============================================================================
# Callback registry
# =============================================================================

def build_callbacks(
    tb_log_dir: Optional[Path] = None,
    csv_path: Optional[Path] = None,
    patience: int = 50,
    total_epochs: int = 300,
    metric: str = "metrics/mAP50-95(B)",
) -> Dict[str, List[Callable]]:
    """
    Build and return a dict of Ultralytics-compatible callbacks.

    Returns
    -------
    dict mapping callback event name → list of callables
    """
    callbacks: Dict[str, List[Callable]] = {
        "on_train_epoch_start": [],
        "on_val_end": [],
        "on_fit_epoch_end": [],
    }

    # TensorBoard
    if tb_log_dir:
        tb_cb = TensorBoardCallback(tb_log_dir)
        callbacks["on_fit_epoch_end"].append(tb_cb.on_fit_epoch_end)

    # CSV
    if csv_path:
        csv_cb = CSVLoggerCallback(csv_path)
        callbacks["on_train_epoch_start"].append(csv_cb.on_train_epoch_start)
        callbacks["on_fit_epoch_end"].append(csv_cb.on_fit_epoch_end)

    # Early stopping
    if patience > 0:
        es_cb = EarlyStoppingCallback(
            metric=metric, patience=patience
        )
        callbacks["on_val_end"].append(es_cb.on_val_end)

    # Console summary
    summary_cb = EpochSummaryCallback(total_epochs)
    callbacks["on_train_epoch_start"].append(summary_cb.on_train_epoch_start)
    callbacks["on_fit_epoch_end"].append(summary_cb.on_fit_epoch_end)

    return callbacks
