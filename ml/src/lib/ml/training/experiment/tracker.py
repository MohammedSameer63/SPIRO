"""
SPIRO ML — ExperimentTracker
Structured experiment tracking using MLflow.

Logs:
  - All training hyperparameters (flat dict)
  - Per-epoch metrics (step-based)
  - Artefacts: best.pt, last.pt, best.onnx, CSV, confusion matrix
  - System info (GPU, torch version, dataset stats)
  - Final summary
"""
from __future__ import annotations

import json
import platform
import socket
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from lib.ml.core.logger import get_logger

log = get_logger(__name__)


class ExperimentTracker:
    """
    MLflow experiment tracker for SPIRO training runs.

    Parameters
    ----------
    experiment_name : str
        MLflow experiment name (created if not exists).
    run_name : str
        MLflow run name.
    tracking_uri : str
        MLflow tracking server URI (default: local file store).
    tags : list[str]
        Additional tags for this run.

    Example
    -------
    >>> tracker = ExperimentTracker("spiro_ml", "yolo11s_run1")
    >>> tracker.start(params={"lr0": 0.01, "epochs": 300})
    >>> tracker.log_metrics({"mAP50": 0.82}, step=100)
    >>> tracker.log_artifact(Path("models/checkpoints/best.pt"))
    >>> tracker.end(status="FINISHED")
    """

    def __init__(
        self,
        experiment_name: str = "spiro_ml",
        run_name: str = "training_run",
        tracking_uri: str = "logs/mlflow",
        tags: Optional[List[str]] = None,
    ) -> None:
        self.experiment_name = experiment_name
        self.run_name = run_name
        self.tracking_uri = tracking_uri
        self.tags = tags or []
        self._run = None
        self._active = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self, params: Optional[Dict[str, Any]] = None) -> "ExperimentTracker":
        """Start an MLflow run and log initial parameters."""
        try:
            import mlflow

            mlflow.set_tracking_uri(str(Path(self.tracking_uri).resolve()))
            mlflow.set_experiment(self.experiment_name)
            self._run = mlflow.start_run(run_name=self.run_name)
            self._active = True

            # Tags
            for tag in self.tags:
                mlflow.set_tag("tag", tag)
            mlflow.set_tag("host", socket.gethostname())
            mlflow.set_tag("platform", platform.system())
            mlflow.set_tag("started_at", datetime.utcnow().isoformat())

            # System info
            mlflow.log_param("torch_version", torch.__version__)
            mlflow.log_param("cuda_available", torch.cuda.is_available())
            if torch.cuda.is_available():
                mlflow.log_param("gpu", torch.cuda.get_device_name(0))
                mlflow.log_param(
                    "gpu_mem_GB",
                    round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1),
                )

            # Hyperparameters
            if params:
                # MLflow has 500 char limit per param value
                for k, v in params.items():
                    try:
                        mlflow.log_param(k, str(v)[:490])
                    except Exception:
                        pass

            log.info(f"MLflow run started: {self.run_name} | {self.tracking_uri}")
        except ImportError:
            log.warning("mlflow not installed — experiment tracking disabled")
        except Exception as e:
            log.warning(f"MLflow start failed: {e} — continuing without tracking")
        return self

    def end(self, status: str = "FINISHED") -> None:
        """End the MLflow run."""
        if not self._active:
            return
        try:
            import mlflow
            mlflow.end_run(status=status)
            self._active = False
            log.info(f"MLflow run ended: {status}")
        except Exception as e:
            log.warning(f"MLflow end failed: {e}")

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def log_metrics(self, metrics: Dict[str, float], step: int) -> None:
        """Log a dict of metrics at a given step."""
        if not self._active:
            return
        try:
            import mlflow
            # Clean keys (MLflow dislikes some characters)
            clean = {k.replace("(", "").replace(")", "").replace("/", "_"): float(v)
                     for k, v in metrics.items() if v is not None}
            mlflow.log_metrics(clean, step=step)
        except Exception as e:
            log.debug(f"MLflow log_metrics failed at step {step}: {e}")

    def log_artifact(self, path: Path) -> None:
        """Upload a file to the MLflow run artefacts."""
        if not self._active:
            return
        try:
            import mlflow
            path = Path(path)
            if path.exists():
                mlflow.log_artifact(str(path))
                log.debug(f"MLflow artifact logged: {path.name}")
        except Exception as e:
            log.debug(f"MLflow log_artifact failed ({path}): {e}")

    def log_artifacts_dir(self, directory: Path) -> None:
        """Upload an entire directory to MLflow artefacts."""
        if not self._active:
            return
        try:
            import mlflow
            directory = Path(directory)
            if directory.exists():
                mlflow.log_artifacts(str(directory))
        except Exception as e:
            log.debug(f"MLflow log_artifacts_dir failed: {e}")

    def set_tag(self, key: str, value: str) -> None:
        if not self._active:
            return
        try:
            import mlflow
            mlflow.set_tag(key, str(value)[:490])
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> "ExperimentTracker":
        return self

    def __exit__(self, exc_type, *_) -> None:
        status = "FAILED" if exc_type else "FINISHED"
        self.end(status=status)
