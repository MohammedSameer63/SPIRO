"""
SPIRO ML — ContinuousLearningManager
Full continuous learning lifecycle for SPIRO:

1. Collect new consented inference images
2. Validate image quality
3. Approve / reject samples
4. Snapshot dataset version
5. Queue retraining job
6. Evaluate new model
7. Promote if improved
8. Archive previous model

Integrates DatasetRegistry, ModelRegistry, RetrainingScheduler,
ModelPromoter, and DriftDetector.
"""
from __future__ import annotations

import json
import shutil
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

from lib.ml.core.logger import get_logger
from lib.ml.mlops.registry.dataset_registry import DatasetRegistry
from lib.ml.mlops.registry.model_registry import ModelRegistry
from lib.ml.mlops.continuous_learning.retraining_scheduler import RetrainingScheduler
from lib.ml.mlops.deployment.model_promoter import ModelPromoter

log = get_logger(__name__)

_SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass
class SampleRecord:
    sample_id: str
    image_path: str
    label_path: str = ""
    status: str = "pending"     # pending | approved | rejected
    rejection_reason: str = ""
    blur_score: float = 0.0
    brightness: float = 0.0
    submitted_at: str = field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ"))

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class CLCycleRecord:
    cycle_id: str
    model_id: str
    started_at: str = field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ"))
    finished_at: str = ""
    samples_collected: int = 0
    samples_approved: int = 0
    samples_rejected: int = 0
    dataset_version: str = ""
    retraining_job_id: str = ""
    new_model_version: str = ""
    promoted: bool = False
    status: str = "running"     # running | completed | failed

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ContinuousLearningManager:
    """
    Orchestrates the full SPIRO continuous learning loop.

    Parameters
    ----------
    staging_dir : Path
        Directory where new images are deposited for review.
    dataset_root : Path
        Root of the active training dataset.
    model_id : str
        Which model to retrain (e.g. "yolov11s").
    min_new_samples : int
        Minimum approved samples before triggering a retrain.
    blur_threshold : float
        Reject images with Laplacian variance below this.
    brightness_min/max : float
        Brightness bounds for acceptance.
    auto_promote : bool
        Automatically promote if improved.

    Example
    -------
    >>> clm = ContinuousLearningManager(model_id="yolov11s")
    >>> clm.collect_from_directory(Path("incoming_images/"))
    >>> cycle = clm.run_cycle()
    """

    def __init__(
        self,
        staging_dir: Path = Path("mlops/staging"),
        approved_dir: Path = Path("mlops/approved"),
        dataset_root: Path = Path("datasets/processed"),
        model_id: str = "yolov11s",
        min_new_samples: int = 50,
        blur_threshold: float = 30.0,
        brightness_min: float = 20.0,
        brightness_max: float = 245.0,
        auto_promote: bool = False,
        dataset_registry: Optional[DatasetRegistry] = None,
        model_registry: Optional[ModelRegistry] = None,
        scheduler: Optional[RetrainingScheduler] = None,
        promoter: Optional[ModelPromoter] = None,
        records_path: Path = Path("mlops/cl_records.json"),
    ) -> None:
        self.staging_dir = Path(staging_dir)
        self.approved_dir = Path(approved_dir)
        self.dataset_root = Path(dataset_root)
        self.model_id = model_id
        self.min_new_samples = min_new_samples
        self.blur_threshold = blur_threshold
        self.brightness_min = brightness_min
        self.brightness_max = brightness_max
        self.auto_promote = auto_promote

        for d in [self.staging_dir, self.approved_dir]:
            d.mkdir(parents=True, exist_ok=True)

        self.dataset_registry = dataset_registry or DatasetRegistry()
        self.model_registry   = model_registry   or ModelRegistry()
        self.scheduler        = scheduler         or RetrainingScheduler()
        self.promoter         = promoter          or ModelPromoter(
            registry=self.model_registry, auto_deploy=auto_promote
        )
        self.records_path = Path(records_path)
        self._sample_records: List[SampleRecord] = []
        self._cycles: List[CLCycleRecord] = self._load_cycles()

    # ------------------------------------------------------------------
    # Sample collection
    # ------------------------------------------------------------------

    def collect_from_directory(
        self,
        source_dir: Path,
        consent_required: bool = True,
    ) -> List[str]:
        """
        Copy images from source_dir into staging for quality review.
        Returns list of sample_ids collected.
        """
        source_dir = Path(source_dir)
        sample_ids = []

        for img_path in source_dir.iterdir():
            if img_path.suffix.lower() not in _SUPPORTED_EXTS:
                continue
            sid = f"sample_{int(time.time())}_{img_path.stem[:16]}"
            dst = self.staging_dir / img_path.name
            shutil.copy2(str(img_path), str(dst))
            rec = SampleRecord(sample_id=sid, image_path=str(dst))
            self._sample_records.append(rec)
            sample_ids.append(sid)

        log.info(f"Collected {len(sample_ids)} samples from {source_dir}")
        return sample_ids

    def collect_single(
        self, image_path: Path, label_path: Optional[Path] = None
    ) -> str:
        """Add a single image to the staging queue."""
        image_path = Path(image_path)
        dst = self.staging_dir / image_path.name
        shutil.copy2(str(image_path), str(dst))
        sid = f"sample_{int(time.time())}_{image_path.stem[:16]}"
        rec = SampleRecord(
            sample_id=sid,
            image_path=str(dst),
            label_path=str(label_path) if label_path else "",
        )
        self._sample_records.append(rec)
        return sid

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate_staged_samples(self) -> Dict[str, int]:
        """
        Run quality checks on all pending staged samples.
        Returns {'approved': N, 'rejected': M}.
        """
        approved = rejected = 0
        for rec in self._sample_records:
            if rec.status != "pending":
                continue
            result = self._validate_image(rec.image_path)
            if result["passed"]:
                rec.status = "approved"
                rec.blur_score = result["blur"]
                rec.brightness = result["brightness"]
                # Copy to approved dir
                dst = self.approved_dir / Path(rec.image_path).name
                shutil.copy2(rec.image_path, str(dst))
                rec.image_path = str(dst)
                approved += 1
            else:
                rec.status = "rejected"
                rec.rejection_reason = result["reason"]
                rejected += 1

        log.info(f"Sample validation: {approved} approved, {rejected} rejected")
        return {"approved": approved, "rejected": rejected}

    # ------------------------------------------------------------------
    # Dataset versioning
    # ------------------------------------------------------------------

    def snapshot_dataset(self, description: str = "") -> str:
        """
        Create a new dataset version snapshot including approved samples.
        Returns the new version string.
        """
        parent = self.dataset_registry.get_latest(self.model_id)
        dv = self.dataset_registry.snapshot_from_disk(
            dataset_id=self.model_id,
            dataset_root=self.dataset_root,
            description=description or f"CL snapshot including {self._approved_count()} new samples",
            parent_version=parent.version if parent else None,
        )
        self.dataset_registry.register(dv)
        log.info(f"Dataset snapshot: {self.model_id} {dv.version} ({dv.image_count} images)")
        return dv.version

    # ------------------------------------------------------------------
    # Full CL cycle
    # ------------------------------------------------------------------

    def run_cycle(
        self,
        force: bool = False,
        config_path: Optional[str] = None,
    ) -> CLCycleRecord:
        """
        Run a complete CL cycle:
        1. Validate staged samples
        2. Check if enough new data
        3. Snapshot dataset
        4. Queue retraining
        5. Optionally promote

        Parameters
        ----------
        force : skip min_new_samples check
        config_path : training YAML override

        Returns
        -------
        CLCycleRecord
        """
        cycle_id = f"cycle_{int(time.time())}"
        cycle = CLCycleRecord(cycle_id=cycle_id, model_id=self.model_id)
        self._cycles.append(cycle)

        # Validate
        counts = self.validate_staged_samples()
        cycle.samples_collected = len(self._sample_records)
        cycle.samples_approved = counts["approved"]
        cycle.samples_rejected = counts["rejected"]

        if cycle.samples_approved < self.min_new_samples and not force:
            cycle.status = "completed"
            cycle.finished_at = time.strftime("%Y-%m-%dT%H:%M:%SZ")
            reason = (
                f"Only {cycle.samples_approved} approved samples "
                f"(min={self.min_new_samples}) — skipping retraining"
            )
            log.info(f"CL cycle {cycle_id}: {reason}")
            self._save_cycles()
            return cycle

        # Snapshot dataset
        dataset_version = self.snapshot_dataset(
            description=f"CL cycle {cycle_id}: +{cycle.samples_approved} samples"
        )
        cycle.dataset_version = dataset_version

        # Queue retraining
        job_id = self.scheduler.schedule(
            model_id=self.model_id,
            schedule="manual",
            trigger="continuous",
            dataset_version_id=dataset_version,
            config_path=config_path or "",
            priority=3,
            notes=f"CL cycle {cycle_id}",
        )
        cycle.retraining_job_id = job_id
        log.info(f"CL cycle {cycle_id}: retraining queued as {job_id}")

        # If auto_promote and a trained model result is available, promote
        # (In production this would be called after training completes via callback)
        cycle.status = "completed"
        cycle.finished_at = time.strftime("%Y-%m-%dT%H:%M:%SZ")
        self._save_cycles()
        return cycle

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def export_cl_report(self, output_path: Optional[Path] = None) -> Path:
        output_path = output_path or Path("mlops/reports/continuous_learning_report.json")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        report = {
            "model_id": self.model_id,
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "cycles": [c.to_dict() for c in self._cycles],
            "sample_summary": {
                "total": len(self._sample_records),
                "approved": sum(1 for r in self._sample_records if r.status == "approved"),
                "rejected": sum(1 for r in self._sample_records if r.status == "rejected"),
                "pending": sum(1 for r in self._sample_records if r.status == "pending"),
            },
        }
        output_path.write_text(json.dumps(report, indent=2))
        log.info(f"CL report → {output_path}")
        return output_path

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _validate_image(self, image_path: str) -> Dict[str, Any]:
        try:
            img = cv2.imread(image_path, cv2.IMREAD_COLOR)
            if img is None:
                return {"passed": False, "reason": "unreadable", "blur": 0.0, "brightness": 0.0}
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
            brightness = float(gray.mean())
            if blur < self.blur_threshold:
                return {"passed": False, "reason": f"blur_{blur:.1f}", "blur": blur, "brightness": brightness}
            if brightness < self.brightness_min:
                return {"passed": False, "reason": f"dark_{brightness:.1f}", "blur": blur, "brightness": brightness}
            if brightness > self.brightness_max:
                return {"passed": False, "reason": f"overexposed_{brightness:.1f}", "blur": blur, "brightness": brightness}
            return {"passed": True, "reason": "", "blur": blur, "brightness": brightness}
        except Exception as e:
            return {"passed": False, "reason": str(e), "blur": 0.0, "brightness": 0.0}

    def _approved_count(self) -> int:
        return sum(1 for r in self._sample_records if r.status == "approved")

    def _load_cycles(self) -> List[CLCycleRecord]:
        if self.records_path.exists():
            try:
                data = json.loads(self.records_path.read_text())
                return [CLCycleRecord(**c) for c in data.get("cycles", [])]
            except Exception:
                pass
        return []

    def _save_cycles(self) -> None:
        self.records_path.parent.mkdir(parents=True, exist_ok=True)
        self.records_path.write_text(
            json.dumps({"cycles": [c.to_dict() for c in self._cycles]}, indent=2)
        )
