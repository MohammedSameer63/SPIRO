"""
SPIRO ML — ConceptDriftDetector
Detects when the relationship between inputs and outputs changes —
i.e., the model's predictions become systematically wrong even when
the input distribution looks stable.

Uses Page-Hinkley test on model confidence as a proxy for concept drift:
a sustained drop in confidence on the same input types indicates the
model's decision boundary no longer matches reality.

Also provides Entropy-based detection and CUSUM monitoring.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from lib.ml.core.logger import get_logger

log = get_logger(__name__)


@dataclass
class ConceptDriftEvent:
    detected_at: str
    detector: str
    statistic: float
    threshold: float
    n_samples: int
    details: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class PageHinkleyDetector:
    """
    Page-Hinkley test for detecting a persistent change in the mean
    of a stream of values (e.g., model confidence per class).

    Parameters
    ----------
    delta : float
        Allowed deviation from the running mean (insensitivity parameter).
    lambda_ : float
        Detection threshold (higher = less sensitive).
    burn_in : int
        Minimum samples before detection is active.
    """

    def __init__(
        self,
        delta: float = 0.005,
        lambda_: float = 50.0,
        burn_in: int = 30,
    ) -> None:
        self.delta = delta
        self.lambda_ = lambda_
        self.burn_in = burn_in
        self._sum: float = 0.0
        self._min_sum: float = 0.0
        self._n: int = 0
        self._mean: float = 0.0
        self.drift_detected: bool = False

    def update(self, value: float) -> bool:
        """
        Add a new observation.
        Returns True if drift is detected.
        """
        self._n += 1
        self._mean += (value - self._mean) / self._n
        self._sum += value - self._mean - self.delta
        self._min_sum = min(self._min_sum, self._sum)
        ph = self._sum - self._min_sum
        if self._n >= self.burn_in and ph > self.lambda_:
            self.drift_detected = True
            return True
        return False

    def reset(self) -> None:
        self._sum = 0.0
        self._min_sum = 0.0
        self._n = 0
        self._mean = 0.0
        self.drift_detected = False

    @property
    def ph_statistic(self) -> float:
        return self._sum - self._min_sum


class CUSUMDetector:
    """
    Cumulative Sum (CUSUM) change detection for monitoring accuracy trends.

    Parameters
    ----------
    k : float
        Allowable slack (half the expected shift to detect).
    h : float
        Decision threshold.
    """

    def __init__(self, k: float = 0.5, h: float = 5.0) -> None:
        self.k = k
        self.h = h
        self._c_plus: float = 0.0
        self._c_minus: float = 0.0
        self._n: int = 0
        self._mean_ref: Optional[float] = None
        self.drift_detected: bool = False

    def set_reference(self, values: np.ndarray) -> None:
        """Set the reference mean from a baseline window."""
        self._mean_ref = float(values.mean())
        log.debug(f"CUSUM reference mean: {self._mean_ref:.4f}")

    def update(self, value: float) -> bool:
        if self._mean_ref is None:
            return False
        self._n += 1
        z = value - self._mean_ref
        self._c_plus  = max(0.0, self._c_plus  + z - self.k)
        self._c_minus = max(0.0, self._c_minus - z - self.k)
        if self._c_plus > self.h or self._c_minus > self.h:
            self.drift_detected = True
            return True
        return False

    def reset(self) -> None:
        self._c_plus = 0.0
        self._c_minus = 0.0
        self._n = 0
        self.drift_detected = False

    @property
    def statistic(self) -> float:
        return max(self._c_plus, self._c_minus)


class ConceptDriftDetector:
    """
    High-level concept drift detector wrapping Page-Hinkley + CUSUM.

    Usage
    -----
    >>> detector = ConceptDriftDetector()
    >>> detector.set_baseline(baseline_confidences)
    >>> events = detector.update_batch(current_confidences, model_id="yolov11s")
    >>> if events:
    ...     trigger_retraining()
    """

    def __init__(
        self,
        ph_delta: float = 0.005,
        ph_lambda: float = 50.0,
        ph_burn_in: int = 30,
        cusum_k: float = 0.5,
        cusum_h: float = 5.0,
        report_dir: Path = Path("mlops/drift"),
    ) -> None:
        self._ph = PageHinkleyDetector(ph_delta, ph_lambda, ph_burn_in)
        self._cusum = CUSUMDetector(cusum_k, cusum_h)
        self.report_dir = Path(report_dir)
        self.report_dir.mkdir(parents=True, exist_ok=True)
        self._events: List[ConceptDriftEvent] = []

    def set_baseline(self, baseline_values: np.ndarray) -> None:
        """Initialise detectors with a baseline window."""
        self._ph.reset()
        self._cusum.set_reference(baseline_values)
        log.info(f"Concept drift baseline set (n={len(baseline_values)})")

    def update(self, value: float) -> List[ConceptDriftEvent]:
        """Update with a single observation. Returns any new drift events."""
        events: List[ConceptDriftEvent] = []
        now = time.strftime("%Y-%m-%dT%H:%M:%SZ")

        if self._ph.update(value):
            ev = ConceptDriftEvent(
                detected_at=now,
                detector="page_hinkley",
                statistic=round(self._ph.ph_statistic, 4),
                threshold=self._ph.lambda_,
                n_samples=self._ph._n,
                details=f"PH statistic {self._ph.ph_statistic:.2f} exceeded λ={self._ph.lambda_}",
            )
            events.append(ev)
            self._events.append(ev)
            log.warning(f"Concept drift (Page-Hinkley): {ev.details}")
            self._ph.reset()

        if self._cusum.update(value):
            ev = ConceptDriftEvent(
                detected_at=now,
                detector="cusum",
                statistic=round(self._cusum.statistic, 4),
                threshold=self._cusum.h,
                n_samples=self._cusum._n,
                details=f"CUSUM statistic {self._cusum.statistic:.2f} exceeded h={self._cusum.h}",
            )
            events.append(ev)
            self._events.append(ev)
            log.warning(f"Concept drift (CUSUM): {ev.details}")
            self._cusum.reset()

        return events

    def update_batch(
        self,
        values: np.ndarray,
        model_id: str = "unknown",
    ) -> List[ConceptDriftEvent]:
        """Update with a batch of values."""
        all_events: List[ConceptDriftEvent] = []
        for v in values:
            all_events.extend(self.update(float(v)))
        if all_events:
            self._save_events(model_id, all_events)
        return all_events

    def detect_entropy_drift(
        self,
        baseline_probs: np.ndarray,
        current_probs: np.ndarray,
        threshold: float = 0.2,
    ) -> Tuple[bool, float]:
        """
        Entropy-based drift detection.
        High entropy = more uncertain = potential concept drift.

        Parameters
        ----------
        baseline_probs : [N_baseline, num_classes] softmax probabilities
        current_probs  : [N_current, num_classes]
        threshold      : mean entropy increase that triggers detection

        Returns
        -------
        (drifted: bool, entropy_delta: float)
        """
        def _entropy(probs: np.ndarray) -> float:
            eps = 1e-12
            p = np.clip(probs, eps, 1.0)
            return float(-(p * np.log(p)).sum(axis=1).mean())

        b_ent = _entropy(baseline_probs)
        c_ent = _entropy(current_probs)
        delta = c_ent - b_ent
        drifted = delta > threshold
        if drifted:
            log.warning(
                f"Entropy drift detected: baseline={b_ent:.4f} current={c_ent:.4f} "
                f"delta={delta:.4f} (threshold={threshold})"
            )
        return drifted, round(delta, 4)

    def export_report(self, model_id: str, output_path: Optional[Path] = None) -> Path:
        output_path = output_path or Path("mlops/reports/drift_report.json")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        report = {
            "model_id": model_id,
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "total_events": len(self._events),
            "events": [e.to_dict() for e in self._events],
        }
        output_path.write_text(json.dumps(report, indent=2))
        log.info(f"Drift report → {output_path}")
        return output_path

    def _save_events(self, model_id: str, events: List[ConceptDriftEvent]) -> None:
        out = self.report_dir / f"concept_drift_{model_id}_{int(time.time())}.json"
        out.write_text(json.dumps([e.to_dict() for e in events], indent=2))
