"""
SPIRO ML — ModelRegistry
Thread-safe, JSON-backed registry of every trained model version.

Stores a versioned history of all model artefacts, metrics, and lifecycle
state. Supports promotion (candidate → staging → production → archived),
rollback, and structured querying.

Storage layout
--------------
mlops/models/
  registry.json           — master index
  <model_id>/
    <version>/
      metadata.json       — full ModelMetadata record
"""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from lib.ml.core.logger import get_logger
from lib.ml.mlops.registry.model_metadata import ModelMetadata, ModelMetrics

log = get_logger(__name__)

_LOCK = threading.Lock()


class ModelRegistry:
    """
    Versioned registry of SPIRO ML model artefacts.

    Parameters
    ----------
    root : Path
        Registry root directory (default: mlops/models).

    Example
    -------
    >>> registry = ModelRegistry()
    >>> meta = ModelMetadata(model_id="yolov11s", version="v1.0.0", ...)
    >>> registry.register(meta)
    >>> prod = registry.get_production_model("yolov11s")
    >>> registry.promote(meta.model_id, meta.version, "production")
    """

    def __init__(self, root: Path = Path("mlops/models")) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._index_path = self.root / "registry.json"
        self._index: Dict[str, Any] = self._load_index()

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(self, meta: ModelMetadata) -> None:
        """Register a new model version."""
        with _LOCK:
            meta_dir = self.root / meta.model_id / meta.version
            meta_dir.mkdir(parents=True, exist_ok=True)
            meta_path = meta_dir / "metadata.json"
            meta_path.write_text(meta.to_json())

            # Update master index
            if meta.model_id not in self._index:
                self._index[meta.model_id] = {}
            self._index[meta.model_id][meta.version] = {
                "stage": meta.stage,
                "architecture": meta.architecture,
                "task": meta.task,
                "created_at": meta.created_at,
                "mAP50_95": meta.metrics.mAP50_95,
                "top1_accuracy": meta.metrics.top1_accuracy,
                "path": str(meta_path),
            }
            self._save_index()
        log.info(f"Registered {meta.model_id} v{meta.version} [{meta.stage}]")

    # ------------------------------------------------------------------
    # Querying
    # ------------------------------------------------------------------

    def get(self, model_id: str, version: str) -> Optional[ModelMetadata]:
        """Retrieve a specific model version."""
        meta_path = self.root / model_id / version / "metadata.json"
        if not meta_path.exists():
            return None
        return ModelMetadata.from_json(meta_path.read_text())

    def get_all_versions(self, model_id: str) -> List[ModelMetadata]:
        """Return all versions of a model, sorted by creation time."""
        versions = self._index.get(model_id, {})
        metas = []
        for ver in sorted(versions.keys()):
            m = self.get(model_id, ver)
            if m:
                metas.append(m)
        return metas

    def get_production_model(self, model_id: str) -> Optional[ModelMetadata]:
        """Return the currently production-staged model."""
        for ver, info in self._index.get(model_id, {}).items():
            if info.get("stage") == "production":
                return self.get(model_id, ver)
        return None

    def get_staging_models(self, model_id: str) -> List[ModelMetadata]:
        return [
            m for m in self.get_all_versions(model_id)
            if m.stage == "staging"
        ]

    def get_candidates(self, model_id: str) -> List[ModelMetadata]:
        return [
            m for m in self.get_all_versions(model_id)
            if m.stage == "candidate"
        ]

    def list_models(self) -> List[str]:
        """Return all registered model IDs."""
        return list(self._index.keys())

    # ------------------------------------------------------------------
    # Lifecycle transitions
    # ------------------------------------------------------------------

    def promote(
        self,
        model_id: str,
        version: str,
        target_stage: str,
        approved_by: str = "system",
    ) -> None:
        """
        Promote a model to target_stage.
        Valid transitions: candidate → staging → production.
        Demotes any existing production model to archived.
        """
        valid = {"candidate", "staging", "production", "archived"}
        if target_stage not in valid:
            raise ValueError(f"Invalid stage: {target_stage}")

        with _LOCK:
            meta = self.get(model_id, version)
            if meta is None:
                raise KeyError(f"Model {model_id} v{version} not found")

            # Demote existing production → archived
            if target_stage == "production":
                current_prod = self.get_production_model(model_id)
                if current_prod and current_prod.version != version:
                    self._update_stage(model_id, current_prod.version, "archived")
                    log.info(f"Archived previous production: {model_id} v{current_prod.version}")

            meta.stage = target_stage
            meta.approved = True
            meta.approved_by = approved_by
            meta.approved_at = time.strftime("%Y-%m-%dT%H:%M:%SZ")
            if target_stage == "production":
                meta.deployed_at = meta.approved_at

            # Write updated metadata
            meta_path = self.root / model_id / version / "metadata.json"
            meta_path.write_text(meta.to_json())
            self._index[model_id][version]["stage"] = target_stage
            self._save_index()

        log.info(f"Promoted {model_id} v{version} → {target_stage} (by {approved_by})")

    def archive(self, model_id: str, version: str) -> None:
        self.promote(model_id, version, "archived", approved_by="system")

    def _update_stage(self, model_id: str, version: str, stage: str) -> None:
        meta = self.get(model_id, version)
        if meta:
            meta.stage = stage
            (self.root / model_id / version / "metadata.json").write_text(meta.to_json())
            self._index[model_id][version]["stage"] = stage

    # ------------------------------------------------------------------
    # Reports
    # ------------------------------------------------------------------

    def export_registry_json(self, output_path: Optional[Path] = None) -> Path:
        """Export the full registry as a human-readable JSON report."""
        output_path = output_path or Path("mlops/reports/model_registry.json")
        output_path.parent.mkdir(parents=True, exist_ok=True)

        records = []
        for model_id in self.list_models():
            for meta in self.get_all_versions(model_id):
                records.append(meta.to_dict())

        with open(output_path, "w") as f:
            json.dump({"models": records, "total": len(records)}, f, indent=2)
        log.info(f"Registry report → {output_path}")
        return output_path

    def summary(self) -> Dict[str, Any]:
        """Return a brief summary of registry contents."""
        total = sum(len(v) for v in self._index.values())
        prod = {}
        for mid in self.list_models():
            m = self.get_production_model(mid)
            if m:
                prod[mid] = {"version": m.version, "mAP50_95": m.metrics.mAP50_95}
        return {"total_versions": total, "models": list(self._index.keys()), "production": prod}

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load_index(self) -> Dict:
        if self._index_path.exists():
            try:
                return json.loads(self._index_path.read_text())
            except json.JSONDecodeError:
                log.warning("registry.json corrupt — starting fresh")
        return {}

    def _save_index(self) -> None:
        self._index_path.write_text(json.dumps(self._index, indent=2))
