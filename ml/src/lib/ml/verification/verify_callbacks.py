"""
SPIRO ML — VerifyCallbacks
Training callbacks for EfficientNetV2 verifier:
  - TensorBoard metric logging
  - CSV history writer
  - Early stopping
  - Epoch console summary
  - GPU memory tracking
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
# Base callback
# =============================================================================

class BaseCallback:
    """All verification callbacks inherit from this."""

    def on_epoch_start(self, epoch: int, **kwargs) -> None: pass
    def on_epoch_end(self, epoch: int, metrics: Dict[str, float], **kwargs) -> None: pass
    def on_train_end(self, **kwargs) -> None: pass


# =============================================================================
# TensorBoard callback
# =============================================================================

class VerifyTBCallback(BaseCallback):
    """Writes per-epoch metrics to TensorBoard."""

    def __init__(self, log_dir: Path) -> None:
        from torch.utils.tensorboard import SummaryWriter
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir=str(self.log_dir))
        log.info(f"VerifyTBCallback: {self.log_dir}")

    def on_epoch_end(self, epoch: int, metrics: Dict[str, float], **kwargs) -> None:
        phase = kwargs.get("phase", "")
        for key, val in metrics.items():
            tag = f"{phase}/{key}" if phase else key
            self.writer.add_scalar(tag, float(val), epoch)
        # GPU memory
        if torch.cuda.is_available():
            self.writer.add_scalar(
                "System/GPU_mem_GB",
                torch.cuda.memory_reserved() / 1e9,
                epoch,
            )
        self.writer.flush()

    def on_train_end(self, **kwargs) -> None:
        self.writer.close()


# =============================================================================
# CSV callback
# =============================================================================

class VerifyCSVCallback(BaseCallback):
    """Appends per-epoch training metrics to a CSV file."""

    COLUMNS = [
        "epoch", "phase",
        "train_loss", "train_acc",
        "val_loss", "val_acc", "val_top5_acc",
        "lr", "time_s", "gpu_mem_GB",
    ]

    def __init__(self, csv_path: Path) -> None:
        self.csv_path = Path(csv_path)
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        self._wrote_header = self.csv_path.exists()
        self._epoch_start: float = 0.0

    def on_epoch_start(self, epoch: int, **kwargs) -> None:
        self._epoch_start = time.perf_counter()

    def on_epoch_end(self, epoch: int, metrics: Dict[str, float], **kwargs) -> None:
        elapsed = time.perf_counter() - self._epoch_start
        gpu_mem = 0.0
        if torch.cuda.is_available():
            gpu_mem = round(torch.cuda.memory_reserved() / 1e9, 3)

        row = {
            "epoch": epoch,
            "phase": kwargs.get("phase", ""),
            "train_loss":   round(metrics.get("train_loss", 0.0), 6),
            "train_acc":    round(metrics.get("train_acc", 0.0), 6),
            "val_loss":     round(metrics.get("val_loss", 0.0), 6),
            "val_acc":      round(metrics.get("val_acc", 0.0), 6),
            "val_top5_acc": round(metrics.get("val_top5_acc", 0.0), 6),
            "lr":           round(metrics.get("lr", 0.0), 8),
            "time_s":       round(elapsed, 2),
            "gpu_mem_GB":   gpu_mem,
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

class VerifyEarlyStoppingCallback(BaseCallback):
    """
    Stops training when the tracked metric plateaus.

    Parameters
    ----------
    metric : str
        Key in the metrics dict to monitor.
    patience : int
        Number of epochs without improvement before stopping.
    min_delta : float
        Minimum change to count as improvement.
    maximize : bool
        If True, look for increasing metric (accuracy).
        If False, look for decreasing metric (loss).
    """

    def __init__(
        self,
        metric: str = "val_acc",
        patience: int = 15,
        min_delta: float = 1e-4,
        maximize: bool = True,
    ) -> None:
        self.metric = metric
        self.patience = patience
        self.min_delta = min_delta
        self.maximize = maximize
        self._best: Optional[float] = None
        self._counter = 0
        self.should_stop = False

    def on_epoch_end(self, epoch: int, metrics: Dict[str, float], **kwargs) -> None:
        val = metrics.get(self.metric)
        if val is None or self.patience <= 0:
            return
        val = float(val)

        if self._best is None:
            self._best = val
            return

        delta = val - self._best if self.maximize else self._best - val
        if delta > self.min_delta:
            self._best = val
            self._counter = 0
        else:
            self._counter += 1
            log.debug(
                f"EarlyStopping [{self.metric}]: no improvement "
                f"{self._counter}/{self.patience} (best={self._best:.4f})"
            )
            if self._counter >= self.patience:
                log.info(
                    f"Early stopping triggered "
                    f"(best {self.metric}={self._best:.4f})"
                )
                self.should_stop = True

    @property
    def stopped_early(self) -> bool:
        return self._counter >= self.patience


# =============================================================================
# Epoch summary callback
# =============================================================================

class VerifyEpochSummaryCallback(BaseCallback):
    """Prints a clean per-epoch summary to the console."""

    def __init__(self, total_epochs: int) -> None:
        self.total_epochs = total_epochs
        self._start = 0.0

    def on_epoch_start(self, epoch: int, **kwargs) -> None:
        self._start = time.perf_counter()

    def on_epoch_end(self, epoch: int, metrics: Dict[str, float], **kwargs) -> None:
        elapsed = time.perf_counter() - self._start
        phase = kwargs.get("phase", "")
        prefix = f"[{phase}] " if phase else ""
        gpu_str = ""
        if torch.cuda.is_available():
            gpu_str = f" GPU {torch.cuda.memory_reserved()/1e9:.1f}GB"

        def _f(k): return f"{metrics.get(k, 0):.4f}"

        log.info(
            f"{prefix}Epoch [{epoch:>3}/{self.total_epochs}] "
            f"loss={_f('train_loss')}/{_f('val_loss')} "
            f"top1={_f('train_acc')}/{_f('val_acc')} "
            f"top5={_f('val_top5_acc')} "
            f"lr={metrics.get('lr', 0):.2e} "
            f"| {elapsed:.1f}s{gpu_str}"
        )


# =============================================================================
# Callback registry
# =============================================================================

def build_verify_callbacks(
    tb_log_dir: Optional[Path] = None,
    csv_path: Optional[Path] = None,
    patience: int = 15,
    total_epochs: int = 50,
    metric: str = "val_acc",
) -> List[BaseCallback]:
    """
    Build and return a list of verification training callbacks.

    Parameters
    ----------
    tb_log_dir : Path, optional
    csv_path : Path, optional
    patience : int
    total_epochs : int
    metric : str

    Returns
    -------
    List[BaseCallback]
    """
    callbacks: List[BaseCallback] = []

    if tb_log_dir:
        callbacks.append(VerifyTBCallback(tb_log_dir))

    if csv_path:
        callbacks.append(VerifyCSVCallback(csv_path))

    if patience > 0:
        callbacks.append(
            VerifyEarlyStoppingCallback(metric=metric, patience=patience)
        )

    callbacks.append(VerifyEpochSummaryCallback(total_epochs=total_epochs))

    return callbacks
