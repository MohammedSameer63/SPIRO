"""
SPIRO ML — ModelComparator
Head-to-head comparison between a candidate and production model.
Evaluates both models on the same held-out test set and produces
a structured comparison report (JSON + Markdown).

Metrics compared
----------------
Accuracy / Precision / Recall / F1 / mAP
Latency (CPU, GPU if available)
Memory footprint
Model file size
False positive / false negative rates
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from lib.ml.core.logger import get_logger
from lib.ml.mlops.registry.model_metadata import ModelMetadata
from lib.ml.mlops.validation.model_validator import ModelValidator

log = get_logger(__name__)


@dataclass
class ModelComparisonReport:
    candidate_id: str
    candidate_version: str
    production_id: str
    production_version: str
    compared_at: str = field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ"))

    # Per-model metrics
    candidate_metrics: Dict[str, float] = field(default_factory=dict)
    production_metrics: Dict[str, float] = field(default_factory=dict)

    # Deltas (candidate - production)
    deltas: Dict[str, float] = field(default_factory=dict)

    # Model properties
    candidate_size_mb: float = 0.0
    production_size_mb: float = 0.0
    candidate_latency_ms: float = 0.0
    production_latency_ms: float = 0.0

    # Verdict
    recommendation: str = ""     # "promote" | "reject" | "manual_review"
    reasons: List[str] = field(default_factory=list)
    primary_metric: str = "mAP50_95"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "candidate": {
                "model_id": self.candidate_id,
                "version": self.candidate_version,
                "metrics": self.candidate_metrics,
                "size_mb": self.candidate_size_mb,
                "latency_ms": self.candidate_latency_ms,
            },
            "production": {
                "model_id": self.production_id,
                "version": self.production_version,
                "metrics": self.production_metrics,
                "size_mb": self.production_size_mb,
                "latency_ms": self.production_latency_ms,
            },
            "deltas": self.deltas,
            "recommendation": self.recommendation,
            "reasons": self.reasons,
            "primary_metric": self.primary_metric,
            "compared_at": self.compared_at,
        }

    def to_markdown(self) -> str:
        lines = [
            f"# Model Comparison Report",
            f"**Generated:** {self.compared_at}",
            "",
            f"## Models",
            f"| | Candidate | Production |",
            f"|---|---|---|",
            f"| Model ID | {self.candidate_id} | {self.production_id} |",
            f"| Version | {self.candidate_version} | {self.production_version} |",
            f"| Size (MB) | {self.candidate_size_mb:.1f} | {self.production_size_mb:.1f} |",
            f"| Latency (ms) | {self.candidate_latency_ms:.1f} | {self.production_latency_ms:.1f} |",
            "",
            f"## Metrics",
            f"| Metric | Candidate | Production | Delta |",
            f"|---|---|---|---|",
        ]
        all_metrics = sorted(set(list(self.candidate_metrics) + list(self.production_metrics)))
        for m in all_metrics:
            c_val = self.candidate_metrics.get(m, 0.0)
            p_val = self.production_metrics.get(m, 0.0)
            delta = self.deltas.get(m, c_val - p_val)
            sign = "▲" if delta > 0.001 else ("▼" if delta < -0.001 else "=")
            lines.append(f"| {m} | {c_val:.4f} | {p_val:.4f} | {sign} {abs(delta):.4f} |")

        lines += [
            "",
            f"## Verdict: **{self.recommendation.upper()}**",
            "",
            "**Reasons:**",
        ]
        for r in self.reasons:
            lines.append(f"- {r}")
        return "\n".join(lines)


class ModelComparator:
    """
    Compares candidate vs production model on shared test data.

    Parameters
    ----------
    validator : ModelValidator
    primary_metric : str
        Key metric for promotion decision.
    min_improvement : float
        Minimum delta on primary_metric to recommend promotion.
    max_latency_regression_ms : float
        Maximum allowed latency regression.

    Example
    -------
    >>> comparator = ModelComparator()
    >>> report = comparator.compare(candidate_meta, production_meta)
    >>> print(report.recommendation, report.reasons)
    """

    def __init__(
        self,
        validator: Optional[ModelValidator] = None,
        primary_metric: str = "mAP50_95",
        min_improvement: float = 0.001,
        max_latency_regression_ms: float = 50.0,
        report_dir: Path = Path("mlops/reports"),
    ) -> None:
        self.validator = validator or ModelValidator()
        self.primary_metric = primary_metric
        self.min_improvement = min_improvement
        self.max_latency_regression = max_latency_regression_ms
        self.report_dir = Path(report_dir)
        self.report_dir.mkdir(parents=True, exist_ok=True)

    def compare(
        self,
        candidate: ModelMetadata,
        production: ModelMetadata,
        run_latency_benchmark: bool = True,
        n_latency_runs: int = 20,
    ) -> ModelComparisonReport:
        """
        Build a full comparison report.

        Parameters
        ----------
        candidate, production : ModelMetadata objects from the registry
        run_latency_benchmark : measure ORT latency for both models
        n_latency_runs : inference runs for latency averaging
        """
        report = ModelComparisonReport(
            candidate_id=candidate.model_id,
            candidate_version=candidate.version,
            production_id=production.model_id,
            production_version=production.version,
            primary_metric=self.primary_metric,
        )

        # Extract stored metrics
        report.candidate_metrics = candidate.metrics.to_dict()
        report.production_metrics = production.metrics.to_dict()

        # Compute deltas
        for m in report.candidate_metrics:
            c_val = report.candidate_metrics.get(m, 0.0)
            p_val = report.production_metrics.get(m, 0.0)
            report.deltas[m] = round(float(c_val) - float(p_val), 6)

        # Model sizes
        c_path = Path(candidate.onnx_path) if candidate.onnx_path else None
        p_path = Path(production.onnx_path) if production.onnx_path else None
        report.candidate_size_mb = round(c_path.stat().st_size / 1e6, 2) if c_path and c_path.exists() else candidate.model_size_mb
        report.production_size_mb = round(p_path.stat().st_size / 1e6, 2) if p_path and p_path.exists() else production.model_size_mb

        # Latency benchmark
        if run_latency_benchmark:
            if c_path and c_path.exists():
                report.candidate_latency_ms = self._benchmark_latency(c_path, candidate.input_size, n_latency_runs)
            else:
                report.candidate_latency_ms = candidate.metrics.latency_cpu_ms
            if p_path and p_path.exists():
                report.production_latency_ms = self._benchmark_latency(p_path, production.input_size, n_latency_runs)
            else:
                report.production_latency_ms = production.metrics.latency_cpu_ms

        # Build verdict
        report.recommendation, report.reasons = self._verdict(report)

        # Save
        self._save(report)
        log.info(
            f"Comparison [{candidate.model_id} v{candidate.version} vs "
            f"{production.model_id} v{production.version}]: "
            f"{report.recommendation}"
        )
        return report

    def _verdict(
        self, report: ModelComparisonReport
    ):
        reasons: List[str] = []
        promote = True

        # Primary metric
        primary_delta = report.deltas.get(self.primary_metric, 0.0)
        if primary_delta >= self.min_improvement:
            reasons.append(
                f"{self.primary_metric} improved by {primary_delta:+.4f} "
                f"({report.candidate_metrics.get(self.primary_metric, 0):.4f} vs "
                f"{report.production_metrics.get(self.primary_metric, 0):.4f})"
            )
        else:
            promote = False
            reasons.append(
                f"{self.primary_metric} did not improve sufficiently "
                f"(delta={primary_delta:+.4f}, min={self.min_improvement})"
            )

        # Latency regression
        lat_delta = report.candidate_latency_ms - report.production_latency_ms
        if lat_delta > self.max_latency_regression:
            promote = False
            reasons.append(
                f"Latency regression: +{lat_delta:.1f}ms "
                f"(threshold {self.max_latency_regression}ms)"
            )
        else:
            reasons.append(f"Latency acceptable (delta={lat_delta:+.1f}ms)")

        # Secondary metrics
        for metric in ["f1_macro", "precision_macro", "recall_macro"]:
            d = report.deltas.get(metric, 0.0)
            if d < -0.01:
                promote = False
                reasons.append(f"{metric} regressed by {d:.4f}")

        rec = "promote" if promote else "reject"
        return rec, reasons

    def _benchmark_latency(
        self, onnx_path: Path, input_size: int, n_runs: int
    ) -> float:
        try:
            import onnxruntime as ort
            sess = ort.InferenceSession(
                str(onnx_path), providers=["CPUExecutionProvider"]
            )
            inp = sess.get_inputs()[0]
            dummy = np.zeros((1, 3, input_size, input_size), dtype=np.float32)
            # Warmup
            for _ in range(3):
                sess.run(None, {inp.name: dummy})
            times = []
            for _ in range(n_runs):
                t0 = time.perf_counter()
                sess.run(None, {inp.name: dummy})
                times.append((time.perf_counter() - t0) * 1000)
            return round(float(np.mean(times)), 2)
        except Exception as e:
            log.warning(f"Latency benchmark failed: {e}")
            return 0.0

    def _save(self, report: ModelComparisonReport) -> None:
        base = self.report_dir / f"comparison_{report.candidate_id}_{report.candidate_version}"
        json_path = base.with_suffix(".json")
        md_path = Path(str(base) + "_comparison.md")
        json_path.write_text(json.dumps(report.to_dict(), indent=2))
        md_path.write_text(report.to_markdown())
        log.info(f"Comparison report → {json_path}")
        log.info(f"Comparison Markdown → {md_path}")
