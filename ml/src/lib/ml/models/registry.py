"""
SPIRO ML — ModelRegistry
Tracks trained model versions, their metadata, and performance metrics.
Supports promotion of a "production" model version.
"""
from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from lib.ml.core.logger import get_logger

log = get_logger(__name__)

_REGISTRY_FILE = "registry.json"


class ModelRegistry:
    """
    File-based model registry stored under models/registry/.

    Schema per entry
    ----------------
    {
      "version": "v1",
      "architecture": "yolov11",
      "created_at": "2024-01-15T10:23:00",
      "weights_path": "models/registry/v1/best.pt",
      "onnx_path": "models/registry/v1/model.onnx",
      "metrics": {"mAP50": 0.87, "mAP50-95": 0.62},
      "tags": ["baseline"],
      "notes": "",
      "production": false
    }
    """

    def __init__(self, registry_dir: str | Path = "models/registry") -> None:
        self.registry_dir = Path(registry_dir)
        self.registry_dir.mkdir(parents=True, exist_ok=True)
        self._registry_path = self.registry_dir / _REGISTRY_FILE
        self._data: Dict[str, Any] = self._load()

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def register(
        self,
        version: str,
        architecture: str,
        weights_path: str | Path,
        metrics: Dict[str, float],
        onnx_path: Optional[str | Path] = None,
        tags: Optional[List[str]] = None,
        notes: str = "",
        copy_weights: bool = True,
    ) -> Dict[str, Any]:
        """
        Register a model version.

        Parameters
        ----------
        version : str
            Unique version string, e.g. "v1", "v2_augmented".
        architecture : str
            Model architecture name.
        weights_path : Path
            Source .pt file.
        metrics : dict
            Evaluation metrics to store (mAP50, precision, etc.).
        onnx_path : Path, optional
            Source .onnx file.
        copy_weights : bool
            If True, copies weights into the registry directory.
        """
        if version in self._data:
            log.warning(f"Version {version!r} already exists — overwriting")

        version_dir = self.registry_dir / version
        version_dir.mkdir(exist_ok=True)

        # Copy artefacts
        stored_weights = str(weights_path)
        if copy_weights:
            dst = version_dir / Path(weights_path).name
            shutil.copy2(weights_path, dst)
            stored_weights = str(dst)

        stored_onnx = None
        if onnx_path and copy_weights:
            dst_onnx = version_dir / Path(onnx_path).name
            shutil.copy2(onnx_path, dst_onnx)
            stored_onnx = str(dst_onnx)

        entry = {
            "version": version,
            "architecture": architecture,
            "created_at": datetime.utcnow().isoformat(),
            "weights_path": stored_weights,
            "onnx_path": stored_onnx,
            "metrics": metrics,
            "tags": tags or [],
            "notes": notes,
            "production": False,
        }
        self._data[version] = entry
        self._save()
        log.info(f"Registered model version {version!r}")
        return entry

    def promote(self, version: str) -> None:
        """Mark a version as production (demotes current production)."""
        if version not in self._data:
            raise KeyError(f"Version {version!r} not found in registry")
        for v in self._data.values():
            v["production"] = False
        self._data[version]["production"] = True
        self._save()
        log.info(f"Promoted {version!r} to production")

    def get_production(self) -> Optional[Dict[str, Any]]:
        """Return the current production model entry, or None."""
        for entry in self._data.values():
            if entry.get("production"):
                return entry
        return None

    def get(self, version: str) -> Dict[str, Any]:
        if version not in self._data:
            raise KeyError(f"Version {version!r} not found")
        return self._data[version]

    def list_versions(self) -> List[Dict[str, Any]]:
        return sorted(self._data.values(), key=lambda e: e["created_at"])

    def delete(self, version: str, remove_files: bool = False) -> None:
        if version not in self._data:
            raise KeyError(f"Version {version!r} not found")
        entry = self._data.pop(version)
        if remove_files:
            version_dir = self.registry_dir / version
            if version_dir.exists():
                shutil.rmtree(version_dir)
        self._save()
        log.info(f"Deleted version {version!r}")

    def best_version(self, metric: str = "mAP50") -> Optional[Dict[str, Any]]:
        """Return the version with the highest value for a given metric."""
        candidates = [
            v for v in self._data.values() if metric in v.get("metrics", {})
        ]
        if not candidates:
            return None
        return max(candidates, key=lambda e: e["metrics"][metric])

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load(self) -> Dict[str, Any]:
        if self._registry_path.exists():
            with open(self._registry_path) as f:
                return json.load(f)
        return {}

    def _save(self) -> None:
        with open(self._registry_path, "w") as f:
            json.dump(self._data, f, indent=2)

    def __repr__(self) -> str:
        return f"ModelRegistry({len(self._data)} versions @ {self.registry_dir})"
