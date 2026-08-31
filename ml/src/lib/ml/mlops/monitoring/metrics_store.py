"""
SPIRO ML — MetricsStore
Persistent time-series store for model performance metrics.
Enables trend analysis, health scoring, and regression detection.

Storage: mlops/experiments/metrics/<model_id>/<metric>.jsonl
(one JSON line per observation, append-only)
"""
from __future__ import annotations

import json
import threading
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from lib.ml.core.logger import get_logger

log = get_logger(__name__)
_LOCK = threading.Lock()


class MetricsStore:
    """
    Append-only time-series store for model accuracy, latency, and drift scores.

    Example
    -------
    >>> store = MetricsStore()
    >>> store.record("yolov11s", "mAP50_95", 0.72, version="v1.0.0")
    >>> trend = store.trend("yolov11s", "mAP50_95", last_n=10)
    >>> report = store.health_report("yolov11s")
    """

    def __init__(self, root: Path = Path("mlops/experiments/metrics")) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def record(
        self,
        model_id: str,
        metric_name: str,
        value: float,
        version: str = "",
        tags: Optional[Dict[str, str]] = None,
        timestamp: Optional[str] = None,
    ) -> None:
        """Append a single metric observation."""
        entry = {
            "ts": timestamp or time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "model_id": model_id,
            "version": version,
            "metric": metric_name,
            "value": round(float(value), 8),
            "tags": tags or {},
        }
        path = self._metric_path(model_id, metric_name)
        with _LOCK:
            with open(path, "a") as f:
                f.write(json.dumps(entry) + "\n")

    def record_many(
        self,
        model_id: str,
        metrics: Dict[str, float],
        version: str = "",
        tags: Optional[Dict[str, str]] = None,
    ) -> None:
        """Record multiple metrics in one call."""
        ts = time.strftime("%Y-%m-%dT%H:%M:%SZ")
        for name, value in metrics.items():
            self.record(model_id, name, value, version=version, tags=tags, timestamp=ts)

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def read(
        self,
        model_id: str,
        metric_name: str,
        last_n: Optional[int] = None,
        version: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Return observations for a metric, optionally filtered by version."""
        path = self._metric_path(model_id, metric_name)
        if not path.exists():
            return []
        entries = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    if version is None or entry.get("version") == version:
                        entries.append(entry)
                except json.JSONDecodeError:
                    pass
        if last_n:
            entries = entries[-last_n:]
        return entries

    def values(
        self,
        model_id: str,
        metric_name: str,
        last_n: Optional[int] = None,
        version: Optional[str] = None,
    ) -> List[float]:
        return [e["value"] for e in self.read(model_id, metric_name, last_n, version)]

    # ------------------------------------------------------------------
    # Analysis
    # ------------------------------------------------------------------

    def trend(
        self,
        model_id: str,
        metric_name: str,
        last_n: int = 20,
    ) -> Dict[str, Any]:
        """Compute trend statistics over the last N observations."""
        vals = self.values(model_id, metric_name, last_n=last_n)
        if len(vals) < 2:
            return {"n": len(vals), "mean": vals[0] if vals else 0.0,
                    "trend": "insufficient_data"}
        arr = np.array(vals)
        slope = float(np.polyfit(np.arange(len(arr)), arr, 1)[0])
        return {
            "n": len(vals),
            "mean": round(float(arr.mean()), 6),
            "std": round(float(arr.std()), 6),
            "min": round(float(arr.min()), 6),
            "max": round(float(arr.max()), 6),
            "last": round(float(arr[-1]), 6),
            "slope": round(slope, 8),
            "trend": "improving" if slope > 1e-4 else ("degrading" if slope < -1e-4 else "stable"),
        }

    def health_report(
        self,
        model_id: str,
        metrics: Optional[List[str]] = None,
        last_n: int = 30,
    ) -> Dict[str, Any]:
        """
        Generate a health report for a model.

        Returns trend analysis for all tracked metrics plus a
        composite health score (0–1).
        """
        if metrics is None:
            metrics = self._available_metrics(model_id)

        report: Dict[str, Any] = {
            "model_id": model_id,
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "metrics": {},
        }

        health_scores = []
        for m in metrics:
            t = self.trend(model_id, m, last_n)
            report["metrics"][m] = t
            if t["n"] >= 2:
                # Score: improving=1, stable=0.7, degrading=0.3
                score = {"improving": 1.0, "stable": 0.7, "degrading": 0.3}.get(
                    t["trend"], 0.5
                )
                health_scores.append(score)

        report["health_score"] = round(float(np.mean(health_scores)) if health_scores else 0.5, 3)
        report["status"] = (
            "healthy" if report["health_score"] >= 0.7
            else "degrading" if report["health_score"] >= 0.4
            else "critical"
        )
        return report

    def export_performance_report(self, output_path: Optional[Path] = None) -> Path:
        """Export performance metrics for all models."""
        output_path = output_path or Path("mlops/reports/performance_report.json")
        output_path.parent.mkdir(parents=True, exist_ok=True)

        model_dirs = [d for d in self.root.iterdir() if d.is_dir()]
        all_reports = []
        for model_dir in model_dirs:
            model_id = model_dir.name
            metrics = self._available_metrics(model_id)
            if metrics:
                all_reports.append(self.health_report(model_id, metrics))

        with open(output_path, "w") as f:
            json.dump({"reports": all_reports, "total": len(all_reports)}, f, indent=2)
        log.info(f"Performance report → {output_path}")
        return output_path

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _metric_path(self, model_id: str, metric_name: str) -> Path:
        d = self.root / model_id
        d.mkdir(parents=True, exist_ok=True)
        return d / f"{metric_name}.jsonl"

    def _available_metrics(self, model_id: str) -> List[str]:
        d = self.root / model_id
        if not d.exists():
            return []
        return [p.stem for p in d.glob("*.jsonl")]
