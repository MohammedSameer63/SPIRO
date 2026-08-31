"""
SPIRO ML — ModelMetadata
Canonical data class for a registered model version.
Serialises to/from JSON for storage in the model registry.
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional


def _sha256(path: Path) -> str:
    """Compute SHA-256 of a file."""
    h = hashlib.sha256()
    try:
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return ""


@dataclass
class ModelMetrics:
    top1_accuracy: float = 0.0
    top5_accuracy: float = 0.0
    precision_macro: float = 0.0
    recall_macro: float = 0.0
    f1_macro: float = 0.0
    mAP50: float = 0.0
    mAP50_95: float = 0.0
    val_loss: float = 0.0
    latency_cpu_ms: float = 0.0
    latency_gpu_ms: float = 0.0
    throughput_fps: float = 0.0
    memory_mb: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {k: round(v, 6) for k, v in asdict(self).items()}

    @classmethod
    def from_dict(cls, d: Dict) -> "ModelMetrics":
        return cls(**{k: float(v) for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class ModelMetadata:
    """Complete record for a single trained model version."""

    # Identity
    model_id: str
    version: str
    architecture: str                # "yolov11s" | "efficientnetv2_s" etc.
    task: str                        # "detection" | "verification"

    # Lineage
    dataset_version_id: str
    experiment_id: str
    git_commit: str = ""
    author: str = "spiro_ml"
    created_at: str = field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ"))

    # Artefacts
    onnx_path: str = ""
    pytorch_path: str = ""
    onnx_sha256: str = ""
    pytorch_sha256: str = ""
    model_size_mb: float = 0.0

    # Training config
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    num_classes: int = 109
    input_size: int = 640
    training_epochs: int = 0
    training_time_s: float = 0.0
    random_seed: int = 42

    # Metrics
    metrics: ModelMetrics = field(default_factory=ModelMetrics)

    # Lifecycle
    stage: str = "candidate"   # candidate | staging | production | archived
    approved: bool = False
    approved_by: str = ""
    approved_at: str = ""
    deployed_at: str = ""
    deployment_strategy: str = ""
    tags: List[str] = field(default_factory=list)
    notes: str = ""

    def __post_init__(self) -> None:
        if self.onnx_path and not self.onnx_sha256:
            p = Path(self.onnx_path)
            if p.exists():
                self.onnx_sha256 = _sha256(p)
                self.model_size_mb = round(p.stat().st_size / 1e6, 2)
        if self.pytorch_path and not self.pytorch_sha256:
            p = Path(self.pytorch_path)
            if p.exists():
                self.pytorch_sha256 = _sha256(p)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["metrics"] = self.metrics.to_dict()
        return d

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, d: Dict) -> "ModelMetadata":
        metrics_d = d.pop("metrics", {})
        obj = cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})
        obj.metrics = ModelMetrics.from_dict(metrics_d)
        return obj

    @classmethod
    def from_json(cls, text: str) -> "ModelMetadata":
        return cls.from_dict(json.loads(text))

    def is_better_than(self, other: "ModelMetadata", primary_metric: str = "mAP50_95") -> bool:
        """Return True if this model outperforms `other` on the primary metric."""
        mine = getattr(self.metrics, primary_metric, 0.0)
        theirs = getattr(other.metrics, primary_metric, 0.0)
        return float(mine) > float(theirs)
