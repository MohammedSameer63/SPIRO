"""
SPIRO ML — ExperimentTracker (MLOps)
Records every training run with its full parameter set, metrics,
hardware profile, and model artefact links.

Uses MLflow when available; falls back to a local JSON store.
Integrates with ModelRegistry on run completion.
"""
from __future__ import annotations

import json
import os
import platform
import time
import uuid
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from lib.ml.core.logger import get_logger

log = get_logger(__name__)


@dataclass
class ExperimentRecord:
    """Full record of one training experiment."""
    experiment_id: str
    experiment_name: str
    model_id: str
    model_version: str
    dataset_version_id: str
    status: str = "running"           # running | completed | failed
    started_at: str = field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ"))
    finished_at: str = ""
    duration_s: float = 0.0

    # Configuration
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    random_seed: int = 42
    num_classes: int = 109

    # Hardware
    hardware: Dict[str, Any] = field(default_factory=dict)

    # Metrics (populated at end)
    metrics: Dict[str, float] = field(default_factory=dict)
    metric_history: Dict[str, List[float]] = field(default_factory=dict)

    # Artefacts
    tensorboard_dir: str = ""
    csv_path: str = ""
    onnx_path: str = ""
    pytorch_path: str = ""

    # Links
    mlflow_run_id: str = ""
    mlflow_experiment_id: str = ""
    tags: Dict[str, str] = field(default_factory=dict)
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, d: Dict) -> "ExperimentRecord":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


def _collect_hardware() -> Dict[str, Any]:
    hw: Dict[str, Any] = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cpu_count": os.cpu_count(),
    }
    try:
        import torch
        hw["pytorch"] = torch.__version__
        hw["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            hw["gpu_count"] = torch.cuda.device_count()
            hw["gpu_0"] = torch.cuda.get_device_name(0)
            hw["gpu_0_mem_GB"] = round(
                torch.cuda.get_device_properties(0).total_memory / 1e9, 1
            )
    except ImportError:
        pass
    try:
        import psutil
        hw["ram_GB"] = round(psutil.virtual_memory().total / 1e9, 1)
    except ImportError:
        pass
    return hw


class ExperimentTracker:
    """
    Records and persists training experiment metadata.

    Parameters
    ----------
    experiments_dir : Path
        Directory for JSON experiment records.
    mlflow_tracking_uri : str, optional
        If set, also logs to MLflow.

    Example
    -------
    >>> tracker = ExperimentTracker()
    >>> run_id = tracker.start_run("yolov11s_run1", "yolov11s", "v1.0",
    ...                            "dataset_v1.0.0", {"lr": 0.01})
    >>> tracker.log_metrics(run_id, {"mAP50": 0.72, "val_loss": 0.43})
    >>> tracker.end_run(run_id, status="completed")
    """

    def __init__(
        self,
        experiments_dir: Path = Path("mlops/experiments"),
        mlflow_tracking_uri: Optional[str] = None,
    ) -> None:
        self.experiments_dir = Path(experiments_dir)
        self.experiments_dir.mkdir(parents=True, exist_ok=True)
        self._index_path = self.experiments_dir / "index.json"
        self._index: Dict[str, str] = self._load_index()
        self._active: Dict[str, ExperimentRecord] = {}
        self._mlflow_uri = mlflow_tracking_uri
        self._mlflow_client = None
        if mlflow_tracking_uri:
            self._init_mlflow(mlflow_tracking_uri)

    # ------------------------------------------------------------------
    # Run lifecycle
    # ------------------------------------------------------------------

    def start_run(
        self,
        experiment_name: str,
        model_id: str,
        model_version: str,
        dataset_version_id: str,
        hyperparameters: Optional[Dict[str, Any]] = None,
        tags: Optional[Dict[str, str]] = None,
        notes: str = "",
    ) -> str:
        """Start a new experiment run. Returns experiment_id."""
        exp_id = f"exp_{int(time.time())}_{uuid.uuid4().hex[:6]}"
        record = ExperimentRecord(
            experiment_id=exp_id,
            experiment_name=experiment_name,
            model_id=model_id,
            model_version=model_version,
            dataset_version_id=dataset_version_id,
            hyperparameters=hyperparameters or {},
            hardware=_collect_hardware(),
            tags=tags or {},
            notes=notes,
            status="running",
        )
        self._active[exp_id] = record
        self._save_record(record)
        log.info(f"Experiment started: {exp_id} ({experiment_name})")

        if self._mlflow_client:
            try:
                import mlflow
                mlflow.set_tracking_uri(self._mlflow_uri)
                mlflow.set_experiment(experiment_name)
                run = mlflow.start_run(run_name=f"{model_id}_{model_version}")
                record.mlflow_run_id = run.info.run_id
                record.mlflow_experiment_id = run.info.experiment_id
                mlflow.log_params(hyperparameters or {})
            except Exception as e:
                log.debug(f"MLflow start_run failed: {e}")

        return exp_id

    def log_metrics(
        self,
        experiment_id: str,
        metrics: Dict[str, float],
        step: Optional[int] = None,
    ) -> None:
        """Log a dict of metric values for the current step."""
        record = self._active.get(experiment_id)
        if record is None:
            return
        for k, v in metrics.items():
            record.metrics[k] = float(v)
            if k not in record.metric_history:
                record.metric_history[k] = []
            record.metric_history[k].append(float(v))

        self._save_record(record)

        if self._mlflow_client:
            try:
                import mlflow
                mlflow.log_metrics(metrics, step=step)
            except Exception:
                pass

    def log_param(self, experiment_id: str, key: str, value: Any) -> None:
        record = self._active.get(experiment_id)
        if record:
            record.hyperparameters[key] = value
            self._save_record(record)

    def log_artefact(
        self,
        experiment_id: str,
        onnx_path: str = "",
        pytorch_path: str = "",
        tensorboard_dir: str = "",
        csv_path: str = "",
    ) -> None:
        record = self._active.get(experiment_id)
        if record:
            if onnx_path:    record.onnx_path    = onnx_path
            if pytorch_path: record.pytorch_path = pytorch_path
            if tensorboard_dir: record.tensorboard_dir = tensorboard_dir
            if csv_path:     record.csv_path = csv_path
            self._save_record(record)

    def end_run(
        self,
        experiment_id: str,
        status: str = "completed",
        final_metrics: Optional[Dict[str, float]] = None,
    ) -> None:
        """Finalise a run."""
        record = self._active.get(experiment_id)
        if record is None:
            log.warning(f"Unknown experiment_id: {experiment_id}")
            return
        record.status = status
        record.finished_at = time.strftime("%Y-%m-%dT%H:%M:%SZ")
        try:
            from datetime import datetime
            t0 = datetime.fromisoformat(record.started_at.replace("Z", ""))
            t1 = datetime.fromisoformat(record.finished_at.replace("Z", ""))
            record.duration_s = round((t1 - t0).total_seconds(), 1)
        except Exception:
            pass
        if final_metrics:
            record.metrics.update(final_metrics)
        self._save_record(record)
        self._index[experiment_id] = str(self._record_path(record))
        self._save_index()
        self._active.pop(experiment_id, None)
        log.info(f"Experiment {experiment_id} ended [{status}] in {record.duration_s}s")

        if self._mlflow_client:
            try:
                import mlflow
                mlflow.end_run()
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Querying
    # ------------------------------------------------------------------

    def get(self, experiment_id: str) -> Optional[ExperimentRecord]:
        if experiment_id in self._active:
            return self._active[experiment_id]
        path = self._index.get(experiment_id)
        if path and Path(path).exists():
            return ExperimentRecord.from_dict(
                json.loads(Path(path).read_text())
            )
        return None

    def list_experiments(self, model_id: Optional[str] = None) -> List[ExperimentRecord]:
        records = []
        for exp_id, path in self._index.items():
            p = Path(path)
            if p.exists():
                try:
                    r = ExperimentRecord.from_dict(json.loads(p.read_text()))
                    if model_id is None or r.model_id == model_id:
                        records.append(r)
                except Exception:
                    pass
        return sorted(records, key=lambda r: r.started_at, reverse=True)

    def export_history_json(self, output_path: Optional[Path] = None) -> Path:
        output_path = output_path or Path("mlops/reports/training_history.json")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        records = [r.to_dict() for r in self.list_experiments()]
        with open(output_path, "w") as f:
            json.dump({"experiments": records, "total": len(records)}, f, indent=2)
        log.info(f"Training history → {output_path}")
        return output_path

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _record_path(self, record: ExperimentRecord) -> Path:
        d = self.experiments_dir / record.model_id
        d.mkdir(parents=True, exist_ok=True)
        return d / f"{record.experiment_id}.json"

    def _save_record(self, record: ExperimentRecord) -> None:
        self._record_path(record).write_text(record.to_json())

    def _load_index(self) -> Dict[str, str]:
        if self._index_path.exists():
            try:
                return json.loads(self._index_path.read_text())
            except Exception:
                pass
        return {}

    def _save_index(self) -> None:
        self._index_path.write_text(json.dumps(self._index, indent=2))

    def _init_mlflow(self, uri: str) -> None:
        try:
            import mlflow
            mlflow.set_tracking_uri(uri)
            self._mlflow_client = mlflow.tracking.MlflowClient()
            log.info(f"MLflow tracking: {uri}")
        except ImportError:
            log.warning("mlflow not installed — using JSON-only experiment tracking")
