"""
SPIRO ML — DriftDetector
Statistical drift detection for the SPIRO inference pipeline.

Detects
-------
- Input drift (pixel statistics via PSI / KS test)
- Prediction drift (class distribution shift)
- Confidence drift (mean confidence trend)
- Embedding drift (feature-space shift — placeholder, requires embeddings)
- Data quality drift (blur/brightness trends)

Uses Population Stability Index (PSI) as the primary drift metric.
PSI < 0.1 → no drift, 0.1–0.25 → moderate, > 0.25 → significant.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy import stats

from lib.ml.core.logger import get_logger

log = get_logger(__name__)

PSI_LOW = 0.1
PSI_HIGH = 0.25


def _psi(baseline: np.ndarray, current: np.ndarray, n_bins: int = 10) -> float:
    """Population Stability Index between two distributions."""
    eps = 1e-6
    # Build shared bins from baseline
    min_v, max_v = baseline.min(), baseline.max()
    if max_v == min_v:
        return 0.0
    bins = np.linspace(min_v, max_v, n_bins + 1)
    b_counts, _ = np.histogram(baseline, bins=bins)
    c_counts, _ = np.histogram(current, bins=bins)
    b_pct = b_counts / (b_counts.sum() + eps) + eps
    c_pct = c_counts / (c_counts.sum() + eps) + eps
    return float(np.sum((c_pct - b_pct) * np.log(c_pct / b_pct)))


def _ks_drift(baseline: np.ndarray, current: np.ndarray) -> Tuple[float, float, bool]:
    """Two-sample KS test. Returns (statistic, p_value, drifted)."""
    stat, pval = stats.ks_2samp(baseline, current)
    return float(stat), float(pval), bool(pval < 0.05)


@dataclass
class DriftResult:
    """Result of a single drift check."""
    drift_type: str
    psi: Optional[float] = None
    ks_statistic: Optional[float] = None
    ks_pvalue: Optional[float] = None
    mean_baseline: Optional[float] = None
    mean_current: Optional[float] = None
    drift_score: float = 0.0
    severity: str = "none"    # none | low | medium | high
    details: str = ""
    drifted: bool = False
    checked_at: str = field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ"))

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DriftReport:
    """Full drift report for one model over one evaluation window."""
    model_id: str
    version: str
    window_start: str
    window_end: str
    generated_at: str = field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ"))
    results: List[DriftResult] = field(default_factory=list)
    overall_drift_score: float = 0.0
    overall_severity: str = "none"
    any_drifted: bool = False
    recommendation: str = ""

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["results"] = [r.to_dict() for r in self.results]
        return d

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)


class DriftDetector:
    """
    Statistical drift detection for SPIRO inference streams.

    Parameters
    ----------
    baseline_path : Path, optional
        Saved baseline statistics (JSON). If None, must call set_baseline().
    psi_threshold_low : float
    psi_threshold_high : float
    report_dir : Path

    Example
    -------
    >>> detector = DriftDetector()
    >>> detector.set_baseline(confidences=baseline_confs,
    ...                       class_ids=baseline_cls,
    ...                       blur_scores=baseline_blur)
    >>> report = detector.detect(
    ...     confidences=current_confs,
    ...     class_ids=current_cls,
    ...     blur_scores=current_blur,
    ...     model_id="yolov11s", version="v1.0.0",
    ... )
    """

    def __init__(
        self,
        baseline_path: Optional[Path] = None,
        psi_threshold_low: float = PSI_LOW,
        psi_threshold_high: float = PSI_HIGH,
        report_dir: Path = Path("mlops/drift"),
        num_classes: int = 109,
    ) -> None:
        self.psi_low = psi_threshold_low
        self.psi_high = psi_threshold_high
        self.report_dir = Path(report_dir)
        self.report_dir.mkdir(parents=True, exist_ok=True)
        self.num_classes = num_classes
        self._baseline: Optional[Dict[str, Any]] = None
        if baseline_path and Path(baseline_path).exists():
            self.load_baseline(baseline_path)

    # ------------------------------------------------------------------
    # Baseline management
    # ------------------------------------------------------------------

    def set_baseline(
        self,
        confidences: np.ndarray,
        class_ids: Optional[np.ndarray] = None,
        blur_scores: Optional[np.ndarray] = None,
        brightness_scores: Optional[np.ndarray] = None,
    ) -> None:
        """Set the reference distribution from baseline inference data."""
        self._baseline = {
            "confidences": confidences.tolist(),
            "class_distribution": self._class_dist(class_ids) if class_ids is not None else {},
            "blur_scores": blur_scores.tolist() if blur_scores is not None else [],
            "brightness_scores": brightness_scores.tolist() if brightness_scores is not None else [],
            "n_samples": len(confidences),
            "mean_confidence": float(confidences.mean()),
            "std_confidence": float(confidences.std()),
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
        log.info(f"Baseline set: n={len(confidences)}, mean_conf={confidences.mean():.3f}")

    def save_baseline(self, path: Path) -> None:
        if self._baseline is None:
            raise RuntimeError("No baseline set")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(self._baseline, indent=2))
        log.info(f"Baseline saved → {path}")

    def load_baseline(self, path: Path) -> None:
        self._baseline = json.loads(Path(path).read_text())
        log.info(f"Baseline loaded from {path}")

    # ------------------------------------------------------------------
    # Detection
    # ------------------------------------------------------------------

    def detect(
        self,
        confidences: np.ndarray,
        class_ids: Optional[np.ndarray] = None,
        blur_scores: Optional[np.ndarray] = None,
        brightness_scores: Optional[np.ndarray] = None,
        model_id: str = "unknown",
        version: str = "",
        window_start: Optional[str] = None,
        window_end: Optional[str] = None,
    ) -> DriftReport:
        """
        Run all drift checks against the stored baseline.

        Returns
        -------
        DriftReport
        """
        if self._baseline is None:
            raise RuntimeError("No baseline set. Call set_baseline() first.")

        now = time.strftime("%Y-%m-%dT%H:%M:%SZ")
        report = DriftReport(
            model_id=model_id,
            version=version,
            window_start=window_start or now,
            window_end=window_end or now,
        )

        results: List[DriftResult] = []

        # 1. Confidence drift
        baseline_conf = np.array(self._baseline["confidences"])
        results.append(self._check_confidence_drift(baseline_conf, confidences))

        # 2. Prediction / class distribution drift
        if class_ids is not None and self._baseline.get("class_distribution"):
            results.append(self._check_prediction_drift(
                self._baseline["class_distribution"], class_ids
            ))

        # 3. Input quality drift (blur)
        if blur_scores is not None and self._baseline.get("blur_scores"):
            baseline_blur = np.array(self._baseline["blur_scores"])
            results.append(self._check_distribution_drift(
                baseline_blur, blur_scores, "input_blur_drift"
            ))

        # 4. Brightness drift
        if brightness_scores is not None and self._baseline.get("brightness_scores"):
            baseline_bright = np.array(self._baseline["brightness_scores"])
            results.append(self._check_distribution_drift(
                baseline_bright, brightness_scores, "brightness_drift"
            ))

        report.results = results
        scores = [r.drift_score for r in results]
        report.overall_drift_score = round(float(np.mean(scores)) if scores else 0.0, 4)
        report.any_drifted = any(r.drifted for r in results)
        report.overall_severity = self._severity(report.overall_drift_score)
        report.recommendation = self._recommend(report)

        self._save_report(report)
        log.info(
            f"Drift check [{model_id}]: score={report.overall_drift_score:.3f} "
            f"severity={report.overall_severity} "
            f"drifted={report.any_drifted}"
        )
        return report

    # ------------------------------------------------------------------
    # Individual checks
    # ------------------------------------------------------------------

    def _check_confidence_drift(
        self, baseline: np.ndarray, current: np.ndarray
    ) -> DriftResult:
        psi = _psi(baseline, current)
        ks_stat, ks_pval, ks_drifted = _ks_drift(baseline, current)
        drifted = psi > self.psi_low or ks_drifted
        return DriftResult(
            drift_type="confidence_drift",
            psi=round(psi, 4),
            ks_statistic=round(ks_stat, 4),
            ks_pvalue=round(ks_pval, 4),
            mean_baseline=round(float(baseline.mean()), 4),
            mean_current=round(float(current.mean()), 4),
            drift_score=min(1.0, psi / self.psi_high),
            severity=self._severity(psi),
            drifted=drifted,
            details=(
                f"Baseline mean={baseline.mean():.3f}, "
                f"Current mean={current.mean():.3f}, PSI={psi:.4f}"
            ),
        )

    def _check_prediction_drift(
        self,
        baseline_dist: Dict[str, int],
        current_class_ids: np.ndarray,
    ) -> DriftResult:
        """Chi-squared test on class distribution."""
        nc = self.num_classes
        baseline_counts = np.zeros(nc)
        for cls_str, cnt in baseline_dist.items():
            idx = int(cls_str)
            if idx < nc:
                baseline_counts[idx] = cnt

        current_counts = np.zeros(nc)
        for cid in current_class_ids:
            if 0 <= int(cid) < nc:
                current_counts[int(cid)] += 1

        # Normalise
        b_total = baseline_counts.sum()
        c_total = current_counts.sum()
        if b_total == 0 or c_total == 0:
            return DriftResult(drift_type="prediction_drift", drifted=False,
                               details="Insufficient data")
        b_freq = baseline_counts / b_total
        c_freq = current_counts / c_total
        # PSI over class distribution
        psi = _psi(np.repeat(np.arange(nc), np.round(b_freq * 1000).astype(int).clip(0)),
                   np.repeat(np.arange(nc), np.round(c_freq * 1000).astype(int).clip(0)))

        drifted = psi > self.psi_low
        return DriftResult(
            drift_type="prediction_drift",
            psi=round(psi, 4),
            drift_score=min(1.0, psi / self.psi_high),
            severity=self._severity(psi),
            drifted=drifted,
            details=f"Class distribution PSI={psi:.4f}",
        )

    def _check_distribution_drift(
        self, baseline: np.ndarray, current: np.ndarray, drift_type: str
    ) -> DriftResult:
        psi = _psi(baseline, current)
        ks_stat, ks_pval, ks_drifted = _ks_drift(baseline, current)
        drifted = psi > self.psi_low or ks_drifted
        return DriftResult(
            drift_type=drift_type,
            psi=round(psi, 4),
            ks_statistic=round(ks_stat, 4),
            ks_pvalue=round(ks_pval, 4),
            mean_baseline=round(float(baseline.mean()), 4),
            mean_current=round(float(current.mean()), 4),
            drift_score=min(1.0, psi / self.psi_high),
            severity=self._severity(psi),
            drifted=drifted,
            details=f"PSI={psi:.4f}, KS={ks_stat:.4f} (p={ks_pval:.4f})",
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _severity(self, score: float) -> str:
        if score < self.psi_low:   return "none"
        if score < self.psi_high:  return "medium"
        return "high"

    def _class_dist(self, class_ids: np.ndarray) -> Dict[str, int]:
        dist: Dict[str, int] = {}
        for cid in class_ids:
            k = str(int(cid))
            dist[k] = dist.get(k, 0) + 1
        return dist

    def _recommend(self, report: DriftReport) -> str:
        if not report.any_drifted:
            return "No action required. Distributions are stable."
        if report.overall_severity == "high":
            return (
                "SIGNIFICANT DRIFT DETECTED. "
                "Trigger immediate model evaluation and consider retraining."
            )
        return (
            "Moderate drift detected. "
            "Monitor closely and schedule retraining within the week."
        )

    def _save_report(self, report: DriftReport) -> Path:
        fname = f"drift_{report.model_id}_{int(time.time())}.json"
        out = self.report_dir / fname
        out.write_text(report.to_json())
        return out
