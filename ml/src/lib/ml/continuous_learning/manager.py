"""
SPIRO ML — ContinuousLearningManager
Monitors live inference streams for:
  - Confidence distribution drift (Page-Hinkley test)
  - mAP degradation triggers
  - Automated retraining on accumulated new samples
  - Model version promotion
"""
from __future__ import annotations

import json
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Deque, Dict, List, Optional, Tuple, Union

import numpy as np

from lib.ml.core.config import ConfigManager
from lib.ml.core.logger import get_logger
from lib.ml.models.registry import ModelRegistry

log = get_logger(__name__)


class DriftDetector:
    """
    Page-Hinkley drift detector.
    Signals when the running mean of a stream shifts beyond a threshold.

    Parameters
    ----------
    delta : float
        Magnitude parameter (sensitivity to change).
    lambda_ : float
        Detection threshold — larger = fewer false alarms.
    window : int
        Sliding window size for recent statistics.
    """

    def __init__(
        self,
        delta: float = 0.005,
        lambda_: float = 50.0,
        window: int = 500,
    ) -> None:
        self.delta = delta
        self.lambda_ = lambda_
        self.window: Deque[float] = deque(maxlen=window)
        self._sum = 0.0
        self._min_sum = 0.0
        self._n = 0

    def update(self, value: float) -> bool:
        """
        Feed a new observation.

        Returns True if drift is detected.
        """
        self.window.append(value)
        self._n += 1
        mean = np.mean(self.window) if self.window else value
        self._sum += value - mean - self.delta
        self._min_sum = min(self._min_sum, self._sum)
        ph = self._sum - self._min_sum
        return ph > self.lambda_

    def reset(self) -> None:
        self.window.clear()
        self._sum = 0.0
        self._min_sum = 0.0
        self._n = 0

    @property
    def n_observations(self) -> int:
        return self._n


class NewSampleBuffer:
    """
    Buffers images + pseudo-labels from live inference for retraining.
    Persists to disk in YOLO format so they can be merged with the base dataset.
    """

    def __init__(self, buffer_dir: str | Path) -> None:
        self.buffer_dir = Path(buffer_dir)
        self.images_dir = self.buffer_dir / "images"
        self.labels_dir = self.buffer_dir / "labels"
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.labels_dir.mkdir(parents=True, exist_ok=True)
        self._count = len(list(self.images_dir.iterdir()))

    def add(
        self,
        image: np.ndarray,
        detections: List[Dict[str, Any]],
        stem: Optional[str] = None,
    ) -> None:
        """
        Buffer an image + its detections.

        Parameters
        ----------
        image : np.ndarray
            BGR image.
        detections : list of detection dicts (from ONNXInferenceEngine).
        stem : str, optional
            Filename stem. Auto-generated if None.
        """
        import cv2

        stem = stem or f"sample_{self._count:06d}"
        img_path = self.images_dir / f"{stem}.jpg"
        lbl_path = self.labels_dir / f"{stem}.txt"

        cv2.imwrite(str(img_path), image, [cv2.IMWRITE_JPEG_QUALITY, 90])

        h, w = image.shape[:2]
        with open(lbl_path, "w") as f:
            for det in detections:
                x1, y1, x2, y2 = det["bbox_xyxy"]
                cx = ((x1 + x2) / 2) / w
                cy = ((y1 + y2) / 2) / h
                bw = (x2 - x1) / w
                bh = (y2 - y1) / h
                f.write(f"{det['class_id']} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")

        self._count += 1

    @property
    def count(self) -> int:
        return self._count

    def clear(self) -> None:
        import shutil
        shutil.rmtree(self.buffer_dir)
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.labels_dir.mkdir(parents=True, exist_ok=True)
        self._count = 0


class ContinuousLearningManager:
    """
    Orchestrates the continuous learning loop for SPIRO.

    Workflow
    --------
    1. Live inference results are fed via ``observe()``.
    2. DriftDetector flags distribution shift.
    3. When drift is detected OR min_new_samples threshold is hit,
       ``should_retrain()`` returns True.
    4. Caller invokes ``trigger_retrain()`` which:
       a. Merges buffered samples into the dataset.
       b. Runs training (YOLOTrainer).
       c. Evaluates new model.
       d. Promotes if metrics improve.
       e. Logs to registry.

    Example
    -------
    >>> clm = ContinuousLearningManager(cfg, registry)
    >>> # In your inference loop:
    >>> clm.observe(image, detections, confidence=0.82)
    >>> if clm.should_retrain():
    ...     clm.trigger_retrain()
    """

    def __init__(
        self,
        cfg: ConfigManager,
        registry: ModelRegistry,
        retrain_callback: Optional[Callable[[], Dict[str, float]]] = None,
    ) -> None:
        self.cfg = cfg
        self.registry = registry
        self.retrain_callback = retrain_callback
        cl_cfg = cfg.continuous_learning

        self.drift_detector = DriftDetector(
            window=cl_cfg.drift_window,
        )
        self.buffer = NewSampleBuffer("datasets/continuous_buffer")
        self.min_new_samples = cl_cfg.min_new_samples
        self.trigger_threshold = cl_cfg.trigger_threshold
        self._drift_detected = False
        self._retrain_history: List[Dict[str, Any]] = []
        self._baseline_map: Optional[float] = None

        # Load baseline from production model
        prod = registry.get_production()
        if prod and "mAP50" in prod.get("metrics", {}):
            self._baseline_map = prod["metrics"]["mAP50"]
            log.info(f"Baseline mAP50 from production: {self._baseline_map:.4f}")

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------

    def observe(
        self,
        image: np.ndarray,
        detections: List[Dict[str, Any]],
        confidence: Optional[float] = None,
        buffer_sample: bool = True,
    ) -> bool:
        """
        Feed one inference result into the monitoring stream.

        Parameters
        ----------
        image : np.ndarray
            BGR image that was inferred.
        detections : list
            Detections from ONNXInferenceEngine.
        confidence : float, optional
            Mean confidence of this prediction (used for drift detection).
        buffer_sample : bool
            If True, image + pseudo-labels are buffered for potential retraining.

        Returns
        -------
        bool — True if drift was detected on this observation.
        """
        if confidence is None:
            if detections:
                confidence = float(np.mean([d["confidence"] for d in detections]))
            else:
                confidence = 0.0

        drift = self.drift_detector.update(confidence)
        if drift and not self._drift_detected:
            log.warning(
                f"⚠ Confidence drift detected at observation "
                f"{self.drift_detector.n_observations} "
                f"(mean conf in window: {np.mean(self.drift_detector.window):.3f})"
            )
            self._drift_detected = True

        if buffer_sample:
            self.buffer.add(image, detections)

        return drift

    # ------------------------------------------------------------------
    # Retrain decision
    # ------------------------------------------------------------------

    def should_retrain(self) -> bool:
        """
        Returns True when retraining is warranted.
        Triggered by:
          - Drift detection + sufficient new samples
          - mAP drop beyond trigger_threshold
        """
        has_samples = self.buffer.count >= self.min_new_samples

        if self._drift_detected and has_samples:
            log.info(
                f"Retrain trigger: drift_detected=True, "
                f"buffered_samples={self.buffer.count}"
            )
            return True

        return False

    # ------------------------------------------------------------------
    # Retrain execution
    # ------------------------------------------------------------------

    def trigger_retrain(
        self,
        new_version: Optional[str] = None,
        auto_promote: bool = True,
    ) -> Optional[Dict[str, Any]]:
        """
        Execute the retraining pipeline.

        Parameters
        ----------
        new_version : str, optional
            Version string for new model. Auto-generated if None.
        auto_promote : bool
            Promote new model to production if it beats baseline.

        Returns
        -------
        dict with new model's registry entry, or None on failure.
        """
        if not self.retrain_callback:
            log.error("No retrain_callback set — cannot trigger retraining")
            return None

        version = new_version or f"v{len(self._retrain_history) + 2}"
        log.info(f"=== Triggering Continuous Retraining → {version} ===")

        # Merge buffer into dataset
        self._merge_buffer_to_dataset()

        # Execute training
        try:
            metrics = self.retrain_callback()
        except Exception as e:
            log.error(f"Retraining failed: {e}")
            return None

        # Register
        new_map = metrics.get("mAP50", 0.0)
        entry = self.registry.register(
            version=version,
            architecture="yolov11",
            weights_path=f"models/checkpoints/best.pt",
            metrics=metrics,
            tags=["continuous_learning"],
            notes=f"Auto-retrain at {datetime.utcnow().isoformat()}",
            copy_weights=True,
        )

        # Promote?
        improved = self._baseline_map is None or (
            new_map > self._baseline_map + self.trigger_threshold
        )
        if auto_promote and improved:
            self.registry.promote(version)
            log.info(
                f"✓ Promoted {version} — mAP50 {self._baseline_map or 0:.4f} → {new_map:.4f}"
            )
            self._baseline_map = new_map

        # Reset state
        self._drift_detected = False
        self.drift_detector.reset()
        self.buffer.clear()

        self._retrain_history.append({
            "version": version,
            "metrics": metrics,
            "promoted": improved,
            "timestamp": datetime.utcnow().isoformat(),
        })

        return entry

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _merge_buffer_to_dataset(self) -> None:
        """Copy buffered samples into the processed training split."""
        import shutil
        dst_img = Path("datasets/processed/train/images")
        dst_lbl = Path("datasets/processed/train/labels")
        dst_img.mkdir(parents=True, exist_ok=True)
        dst_lbl.mkdir(parents=True, exist_ok=True)

        n = 0
        for img_path in self.buffer.images_dir.iterdir():
            shutil.copy2(img_path, dst_img / img_path.name)
            lbl = self.buffer.labels_dir / f"{img_path.stem}.txt"
            if lbl.exists():
                shutil.copy2(lbl, dst_lbl / lbl.name)
            n += 1

        log.info(f"Merged {n} buffered samples into training split")

    def summary(self) -> Dict[str, Any]:
        return {
            "observations": self.drift_detector.n_observations,
            "buffered_samples": self.buffer.count,
            "drift_detected": self._drift_detected,
            "retrain_cycles": len(self._retrain_history),
            "baseline_mAP50": self._baseline_map,
            "history": self._retrain_history,
        }
