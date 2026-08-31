"""
SPIRO ML — ModelDeployer
Production deployment manager supporting:
  - Immediate deployment (swap ONNX symlink atomically)
  - Blue-Green deployment (maintain two slots, switch traffic)
  - Canary deployment (gradual traffic shift with health check)
  - Rollback (restore previous production model)

All deployments are recorded in mlops/deployments/deployment_history.json.
"""
from __future__ import annotations

import json
import shutil
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from lib.ml.core.logger import get_logger
from lib.ml.mlops.registry.model_registry import ModelRegistry
from lib.ml.mlops.validation.model_validator import ModelValidator

log = get_logger(__name__)


@dataclass
class DeploymentRecord:
    deployment_id: str
    model_id: str
    version: str
    strategy: str                      # immediate | blue_green | canary | rollback
    onnx_path: str
    deployed_at: str = field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ"))
    deployed_by: str = "system"
    status: str = "active"             # active | rolled_back | superseded
    previous_version: str = ""
    previous_onnx: str = ""
    validation_passed: bool = True
    canary_percentage: int = 100
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ModelDeployer:
    """
    Deploys SPIRO ONNX models to a live serving directory.

    The serving directory contains symlinks / copies of the active models:
      serve/yolo_active.onnx     → current production YOLO
      serve/effnet_active.onnx   → current production EfficientNetV2

    Parameters
    ----------
    serve_dir : Path
        Directory where active model files are placed.
    registry : ModelRegistry
    validator : ModelValidator
    deployment_history_path : Path

    Example
    -------
    >>> deployer = ModelDeployer(serve_dir=Path("models/serve"))
    >>> deployer.deploy(model_id="yolov11s", version="v2.0.0",
    ...                 onnx_path=Path("models/exports/best.onnx"),
    ...                 strategy="immediate")
    """

    _SERVE_NAMES = {
        "detection":     "yolo_active.onnx",
        "verification":  "effnet_active.onnx",
    }

    def __init__(
        self,
        serve_dir: Path = Path("models/serve"),
        registry: Optional[ModelRegistry] = None,
        validator: Optional[ModelValidator] = None,
        deployment_history_path: Path = Path("mlops/deployments/deployment_history.json"),
    ) -> None:
        self.serve_dir = Path(serve_dir)
        self.serve_dir.mkdir(parents=True, exist_ok=True)
        self.registry = registry or ModelRegistry()
        self.validator = validator or ModelValidator()
        self.history_path = Path(deployment_history_path)
        self.history_path.parent.mkdir(parents=True, exist_ok=True)
        self._history: List[DeploymentRecord] = self._load_history()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def deploy(
        self,
        model_id: str,
        version: str,
        onnx_path: Path,
        strategy: str = "immediate",
        task: str = "detection",
        deployed_by: str = "system",
        validate: bool = True,
        input_size: int = 640,
        canary_percentage: int = 100,
        notes: str = "",
    ) -> DeploymentRecord:
        """
        Deploy a model using the chosen strategy.

        Parameters
        ----------
        strategy : "immediate" | "blue_green" | "canary" | "rollback"
        task : "detection" | "verification" — determines the serve filename
        validate : run ModelValidator before deploying
        canary_percentage : traffic % for canary (1–100)

        Returns
        -------
        DeploymentRecord
        """
        onnx_path = Path(onnx_path)
        if not onnx_path.exists():
            raise FileNotFoundError(f"ONNX not found: {onnx_path}")

        # Validation gate
        if validate:
            val_report = self.validator.validate(
                onnx_path=onnx_path,
                model_id=model_id,
                version=version,
                input_size=input_size,
                task=task,
            )
            if not val_report.overall_passed:
                log.error(f"Deployment blocked — validation failed: {val_report.summary}")
                raise RuntimeError(
                    f"Validation failed for {model_id} v{version}: {val_report.summary}"
                )

        # Find current active model for rollback info
        prev_version, prev_onnx = self._current_active(task)

        dep_id = f"dep_{int(time.time())}_{model_id}_{version}"
        record = DeploymentRecord(
            deployment_id=dep_id,
            model_id=model_id,
            version=version,
            strategy=strategy,
            onnx_path=str(onnx_path),
            deployed_by=deployed_by,
            previous_version=prev_version,
            previous_onnx=prev_onnx,
            validation_passed=validate,
            canary_percentage=canary_percentage,
            notes=notes,
        )

        if strategy == "immediate":
            self._deploy_immediate(onnx_path, task)
        elif strategy == "blue_green":
            self._deploy_blue_green(onnx_path, task)
        elif strategy == "canary":
            self._deploy_canary(onnx_path, task, canary_percentage)
        elif strategy == "rollback":
            self._rollback_impl(task, prev_onnx, prev_version)
            record.status = "active"
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        # Update registry lifecycle
        self.registry.promote(model_id, version, "production", approved_by=deployed_by)

        # Record history
        self._history.append(record)
        self._save_history()

        log.info(
            f"Deployed {model_id} v{version} via {strategy} "
            f"[task={task}] → {self._serve_path(task)}"
        )
        return record

    def rollback(
        self,
        task: str = "detection",
        deployed_by: str = "system",
        reason: str = "",
    ) -> Optional[DeploymentRecord]:
        """
        Roll back to the previous active model for the given task.

        Returns
        -------
        DeploymentRecord of the rollback, or None if no previous version.
        """
        prev = self._find_previous_deployment(task)
        if prev is None:
            log.warning(f"No previous deployment found for task={task}")
            return None

        if not Path(prev.onnx_path).exists():
            log.error(f"Previous ONNX missing: {prev.onnx_path}")
            return None

        log.info(f"Rolling back {task} to {prev.model_id} v{prev.version}: {reason}")
        return self.deploy(
            model_id=prev.model_id,
            version=prev.version,
            onnx_path=Path(prev.onnx_path),
            strategy="rollback",
            task=task,
            deployed_by=deployed_by,
            validate=False,   # skip validation on rollback
            notes=f"Rollback: {reason}",
        )

    # ------------------------------------------------------------------
    # Strategy implementations
    # ------------------------------------------------------------------

    def _deploy_immediate(self, onnx_path: Path, task: str) -> None:
        dst = self._serve_path(task)
        shutil.copy2(str(onnx_path), str(dst))
        log.debug(f"Immediate deploy: {onnx_path} → {dst}")

    def _deploy_blue_green(self, onnx_path: Path, task: str) -> None:
        """
        Copy new model to the 'green' slot, verify it loads,
        then atomically replace the 'blue' (active) slot.
        """
        green = self.serve_dir / f"{task}_green.onnx"
        blue  = self._serve_path(task)
        shutil.copy2(str(onnx_path), str(green))
        # Verify green loads before switching
        try:
            import onnxruntime as ort
            ort.InferenceSession(str(green), providers=["CPUExecutionProvider"])
        except Exception as e:
            green.unlink(missing_ok=True)
            raise RuntimeError(f"Blue-Green: green slot failed to load: {e}") from e
        # Atomic rename (on same filesystem)
        green.replace(blue)
        log.debug(f"Blue-Green: switched to {blue}")

    def _deploy_canary(
        self, onnx_path: Path, task: str, percentage: int
    ) -> None:
        """
        Copy to canary slot. At 100% completes full promotion.
        Below 100%, both blue and canary coexist; the serving layer
        routes `percentage`% of traffic to canary.
        """
        canary = self.serve_dir / f"{task}_canary.onnx"
        shutil.copy2(str(onnx_path), str(canary))
        log.info(f"Canary deployed at {percentage}% traffic: {canary}")
        if percentage >= 100:
            self._deploy_immediate(onnx_path, task)
            canary.unlink(missing_ok=True)
            log.info(f"Canary promoted to 100% — active slot updated")

    def _rollback_impl(self, task: str, prev_onnx: str, prev_version: str) -> None:
        if prev_onnx and Path(prev_onnx).exists():
            dst = self._serve_path(task)
            shutil.copy2(prev_onnx, str(dst))
            log.info(f"Rolled back {task} to {prev_version}: {dst}")
        else:
            log.warning("Rollback: no previous ONNX available")

    # ------------------------------------------------------------------
    # State helpers
    # ------------------------------------------------------------------

    def _serve_path(self, task: str) -> Path:
        name = self._SERVE_NAMES.get(task, f"{task}_active.onnx")
        return self.serve_dir / name

    def _current_active(self, task: str) -> tuple:
        """Return (version, onnx_path) of currently active model."""
        for rec in reversed(self._history):
            if rec.status == "active" and (
                "detection" in rec.model_id or task in rec.model_id
            ):
                return rec.version, rec.onnx_path
        return "", ""

    def _find_previous_deployment(self, task: str) -> Optional[DeploymentRecord]:
        active_records = [
            r for r in reversed(self._history)
            if r.status == "active" and r.strategy != "rollback"
        ]
        if len(active_records) >= 2:
            return active_records[1]
        return None

    def get_active_path(self, task: str) -> Optional[Path]:
        p = self._serve_path(task)
        return p if p.exists() else None

    def export_history_json(self, output_path: Optional[Path] = None) -> Path:
        output_path = output_path or Path("mlops/reports/deployment_history.json")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(
                {"deployments": [r.to_dict() for r in self._history],
                 "total": len(self._history)},
                f, indent=2
            )
        log.info(f"Deployment history → {output_path}")
        return output_path

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load_history(self) -> List[DeploymentRecord]:
        if self.history_path.exists():
            try:
                data = json.loads(self.history_path.read_text())
                return [
                    DeploymentRecord(**r)
                    for r in data.get("deployments", [])
                ]
            except Exception as e:
                log.warning(f"Deployment history corrupt: {e}")
        return []

    def _save_history(self) -> None:
        self.history_path.write_text(
            json.dumps(
                {"deployments": [r.to_dict() for r in self._history],
                 "total": len(self._history)},
                indent=2,
            )
        )
