"""
SPIRO ML — CheckpointManager
Handles all checkpoint I/O for YOLOv11 training:
  - Saving best.pt and last.pt
  - Automatic resume detection
  - Checkpoint metadata (epoch, metrics, config snapshot)
  - Checkpoint history pruning
"""
from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

from lib.ml.core.logger import get_logger

log = get_logger(__name__)

_META_FILE = "checkpoint_meta.json"


class CheckpointManager:
    """
    Manages training checkpoints under a run directory.

    Directory structure
    -------------------
    <run_dir>/
      weights/
        best.pt          ← best model (tracked metric)
        last.pt          ← most recent epoch
        epoch_050.pt     ← periodic saves
      checkpoint_meta.json

    Example
    -------
    >>> cm = CheckpointManager(run_dir=Path("models/checkpoints/spiro_yolo11s"))
    >>> cm.save_best("best.pt", metrics={"mAP50-95": 0.62}, epoch=100)
    >>> resume_path = cm.auto_resume()
    """

    def __init__(
        self,
        run_dir: Path,
        metric: str = "mAP50-95",
        maximize: bool = True,
        keep_last_n: int = 3,
    ) -> None:
        self.run_dir = Path(run_dir)
        self.weights_dir = self.run_dir / "weights"
        self.weights_dir.mkdir(parents=True, exist_ok=True)
        self.metric = metric
        self.maximize = maximize
        self.keep_last_n = keep_last_n

        self._meta: Dict[str, Any] = self._load_meta()
        self._best_value: Optional[float] = self._meta.get("best_value")
        self._periodic: List[str] = self._meta.get("periodic_checkpoints", [])

    # ------------------------------------------------------------------
    # Save operations
    # ------------------------------------------------------------------

    def save_last(self, src_path: Path, epoch: int, metrics: Dict[str, float]) -> Path:
        """Copy src_path to weights/last.pt and update metadata."""
        dst = self.weights_dir / "last.pt"
        shutil.copy2(src_path, dst)
        self._meta["last_epoch"] = epoch
        self._meta["last_metrics"] = metrics
        self._meta["last_saved_at"] = datetime.utcnow().isoformat()
        self._save_meta()
        log.debug(f"last.pt updated (epoch {epoch})")
        return dst

    def save_best(
        self, src_path: Path, epoch: int, metrics: Dict[str, float]
    ) -> Tuple[Path, bool]:
        """
        Copy src_path to weights/best.pt if metrics improved.

        Returns (path, improved: bool).
        """
        value = metrics.get(self.metric, 0.0)
        improved = self._is_better(value)

        if improved:
            dst = self.weights_dir / "best.pt"
            shutil.copy2(src_path, dst)
            self._best_value = value
            self._meta["best_epoch"] = epoch
            self._meta["best_value"] = value
            self._meta["best_metrics"] = metrics
            self._meta["best_saved_at"] = datetime.utcnow().isoformat()
            self._save_meta()
            log.info(f"✓ New best {self.metric}={value:.4f} (epoch {epoch}) → best.pt")
            return dst, True

        return self.weights_dir / "best.pt", False

    def save_periodic(self, src_path: Path, epoch: int) -> Optional[Path]:
        """Save a periodic checkpoint as weights/epoch_{epoch:05d}.pt."""
        fname = f"epoch_{epoch:05d}.pt"
        dst = self.weights_dir / fname
        shutil.copy2(src_path, dst)
        self._periodic.append(fname)
        self._meta["periodic_checkpoints"] = self._periodic
        self._save_meta()

        # Prune old periodic checkpoints
        if len(self._periodic) > self.keep_last_n:
            to_prune = self._periodic[: -self.keep_last_n]
            for old in to_prune:
                old_path = self.weights_dir / old
                old_path.unlink(missing_ok=True)
            self._periodic = self._periodic[-self.keep_last_n :]
            self._meta["periodic_checkpoints"] = self._periodic
            self._save_meta()

        log.debug(f"Periodic checkpoint saved: {fname}")
        return dst

    # ------------------------------------------------------------------
    # Resume
    # ------------------------------------------------------------------

    def auto_resume(self) -> Optional[Path]:
        """
        Return the path to resume from (last.pt if it exists).
        Returns None if no checkpoint found.
        """
        last = self.weights_dir / "last.pt"
        if last.exists():
            epoch = self._meta.get("last_epoch", "?")
            log.info(f"Auto-resume: found last.pt (epoch {epoch})")
            return last
        log.info("No existing last.pt — starting fresh")
        return None

    def best_path(self) -> Optional[Path]:
        p = self.weights_dir / "best.pt"
        return p if p.exists() else None

    def last_path(self) -> Optional[Path]:
        p = self.weights_dir / "last.pt"
        return p if p.exists() else None

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    def update_meta(self, **kwargs) -> None:
        self._meta.update(kwargs)
        self._save_meta()

    def get_meta(self) -> Dict[str, Any]:
        return dict(self._meta)

    def summary(self) -> str:
        b = self._meta.get("best_value", "N/A")
        be = self._meta.get("best_epoch", "N/A")
        le = self._meta.get("last_epoch", "N/A")
        return f"best {self.metric}={b} @ epoch {be} | last epoch={le}"

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _is_better(self, value: float) -> bool:
        if self._best_value is None:
            return True
        if self.maximize:
            return value > self._best_value
        return value < self._best_value

    def _load_meta(self) -> Dict[str, Any]:
        meta_path = self.run_dir / _META_FILE
        if meta_path.exists():
            with open(meta_path) as f:
                return json.load(f)
        return {
            "run_dir": str(self.run_dir),
            "metric": self.metric,
            "created_at": datetime.utcnow().isoformat(),
        }

    def _save_meta(self) -> None:
        with open(self.run_dir / _META_FILE, "w") as f:
            json.dump(self._meta, f, indent=2)
