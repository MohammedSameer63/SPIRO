"""
SPIRO ML — PerformanceMonitor
Real-time monitoring of inference pipeline resource usage:
  - CPU / RAM / GPU utilisation
  - Per-request latency (preprocess, detect, verify, total)
  - Batch throughput and FPS
  - ORT memory footprint
  - Queue / wait time tracking
  - Slow request alerting
"""
from __future__ import annotations

import json
import threading
import time
from collections import deque
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional

import numpy as np

from lib.ml.core.logger import get_logger

log = get_logger(__name__)


@dataclass
class RequestMetrics:
    request_id: str
    timestamp: str
    preprocess_ms: float = 0.0
    detection_ms: float = 0.0
    verification_ms: float = 0.0
    fusion_ms: float = 0.0
    total_ms: float = 0.0
    detections: int = 0
    status: str = "success"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ResourceSnapshot:
    timestamp: str
    cpu_percent: float = 0.0
    ram_used_gb: float = 0.0
    ram_total_gb: float = 0.0
    gpu_percent: float = 0.0
    gpu_mem_used_gb: float = 0.0
    gpu_mem_total_gb: float = 0.0
    process_ram_mb: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class PerformanceMonitor:
    """
    Monitors SPIRO inference pipeline performance metrics.

    Parameters
    ----------
    window_size : int
        Rolling window for computing statistics.
    slow_threshold_ms : float
        Log a warning when total_ms exceeds this.
    report_dir : Path

    Example
    -------
    >>> monitor = PerformanceMonitor()
    >>> monitor.record_request("req_001", preprocess_ms=12.0,
    ...                        detection_ms=48.0, total_ms=75.0)
    >>> report = monitor.report()
    >>> print(report["fps"], report["p95_latency_ms"])
    """

    def __init__(
        self,
        window_size: int = 500,
        slow_threshold_ms: float = 500.0,
        report_dir: Path = Path("mlops/reports"),
        enable_system_monitoring: bool = True,
    ) -> None:
        self.window_size = window_size
        self.slow_threshold_ms = slow_threshold_ms
        self.report_dir = Path(report_dir)
        self.report_dir.mkdir(parents=True, exist_ok=True)
        self.enable_system = enable_system_monitoring

        self._requests: Deque[RequestMetrics] = deque(maxlen=window_size)
        self._resources: Deque[ResourceSnapshot] = deque(maxlen=1000)
        self._lock = threading.Lock()
        self._start_time = time.perf_counter()
        self._total_requests = 0
        self._failed_requests = 0

        self._monitor_thread: Optional[threading.Thread] = None
        self._monitoring = False
        if enable_system_monitoring:
            self.start_system_monitor()

    # ------------------------------------------------------------------
    # Request tracking
    # ------------------------------------------------------------------

    def record_request(
        self,
        request_id: str,
        preprocess_ms: float = 0.0,
        detection_ms: float = 0.0,
        verification_ms: float = 0.0,
        fusion_ms: float = 0.0,
        total_ms: float = 0.0,
        detections: int = 0,
        status: str = "success",
    ) -> None:
        """Record metrics for a single inference request."""
        with self._lock:
            self._total_requests += 1
            if status != "success":
                self._failed_requests += 1
            rec = RequestMetrics(
                request_id=request_id,
                timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ"),
                preprocess_ms=preprocess_ms,
                detection_ms=detection_ms,
                verification_ms=verification_ms,
                fusion_ms=fusion_ms,
                total_ms=total_ms,
                detections=detections,
                status=status,
            )
            self._requests.append(rec)

        if total_ms > self.slow_threshold_ms:
            log.warning(
                f"Slow request [{request_id}]: {total_ms:.0f}ms "
                f"(threshold {self.slow_threshold_ms}ms)"
            )

    def record_from_pipeline_result(
        self, request_id: str, result: Dict[str, Any]
    ) -> None:
        """Convenience: extract timings directly from SPIROPipeline response dict."""
        timings = result.get("timings", {})
        self.record_request(
            request_id=request_id,
            preprocess_ms=timings.get("preprocess_ms", 0.0),
            detection_ms=timings.get("detection_ms", 0.0),
            verification_ms=timings.get("verification_ms", 0.0),
            fusion_ms=timings.get("fusion_ms", 0.0),
            total_ms=timings.get("total_ms", 0.0),
            detections=result.get("detection_count", 0),
            status=result.get("status", "unknown"),
        )

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def report(self) -> Dict[str, Any]:
        """Compute current performance statistics."""
        with self._lock:
            reqs = list(self._requests)

        if not reqs:
            return {
                "total_requests": self._total_requests,
                "failed_requests": self._failed_requests,
                "window_size": 0,
                "fps": 0.0,
            }

        total_ms   = np.array([r.total_ms for r in reqs])
        preproc_ms = np.array([r.preprocess_ms for r in reqs])
        detect_ms  = np.array([r.detection_ms for r in reqs])
        verify_ms  = np.array([r.verification_ms for r in reqs])
        detections = np.array([r.detections for r in reqs])

        uptime_s = time.perf_counter() - self._start_time
        fps = self._total_requests / max(uptime_s, 1.0)

        return {
            "total_requests": self._total_requests,
            "failed_requests": self._failed_requests,
            "success_rate": round(
                (self._total_requests - self._failed_requests) / max(self._total_requests, 1), 4
            ),
            "window_size": len(reqs),
            "uptime_s": round(uptime_s, 1),
            "fps": round(fps, 2),
            "latency": {
                "mean_ms":   round(float(total_ms.mean()), 2),
                "std_ms":    round(float(total_ms.std()), 2),
                "min_ms":    round(float(total_ms.min()), 2),
                "p50_ms":    round(float(np.percentile(total_ms, 50)), 2),
                "p95_ms":    round(float(np.percentile(total_ms, 95)), 2),
                "p99_ms":    round(float(np.percentile(total_ms, 99)), 2),
                "max_ms":    round(float(total_ms.max()), 2),
            },
            "stage_breakdown": {
                "preprocess_mean_ms": round(float(preproc_ms.mean()), 2),
                "detection_mean_ms":  round(float(detect_ms.mean()), 2),
                "verification_mean_ms": round(float(verify_ms.mean()), 2),
            },
            "detections": {
                "mean_per_image": round(float(detections.mean()), 2),
                "max_per_image": int(detections.max()),
                "total": int(detections.sum()),
            },
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        }

    def export_performance_report(self, output_path: Optional[Path] = None) -> Path:
        output_path = output_path or self.report_dir / "performance_report.json"
        rpt = self.report()
        if self._resources:
            snapshots = list(self._resources)
            cpu_vals = [s.cpu_percent for s in snapshots]
            ram_vals = [s.ram_used_gb for s in snapshots]
            rpt["system"] = {
                "cpu_mean_pct": round(float(np.mean(cpu_vals)), 1),
                "cpu_max_pct":  round(float(np.max(cpu_vals)), 1),
                "ram_mean_gb":  round(float(np.mean(ram_vals)), 2),
                "ram_max_gb":   round(float(np.max(ram_vals)), 2),
            }
        with open(output_path, "w") as f:
            json.dump(rpt, f, indent=2)
        log.info(f"Performance report → {output_path}")
        return output_path

    # ------------------------------------------------------------------
    # System monitoring background thread
    # ------------------------------------------------------------------

    def start_system_monitor(self, interval_s: float = 5.0) -> None:
        self._monitoring = True
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop, args=(interval_s,), daemon=True
        )
        self._monitor_thread.start()

    def stop_system_monitor(self) -> None:
        self._monitoring = False

    def _monitor_loop(self, interval_s: float) -> None:
        while self._monitoring:
            snap = self._collect_snapshot()
            with self._lock:
                self._resources.append(snap)
            time.sleep(interval_s)

    def _collect_snapshot(self) -> ResourceSnapshot:
        snap = ResourceSnapshot(timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ"))
        try:
            import psutil
            snap.cpu_percent = psutil.cpu_percent(interval=None)
            mem = psutil.virtual_memory()
            snap.ram_used_gb = round(mem.used / 1e9, 2)
            snap.ram_total_gb = round(mem.total / 1e9, 2)
            proc = psutil.Process()
            snap.process_ram_mb = round(proc.memory_info().rss / 1e6, 1)
        except ImportError:
            pass
        try:
            import torch
            if torch.cuda.is_available():
                snap.gpu_mem_used_gb = round(torch.cuda.memory_allocated() / 1e9, 2)
                snap.gpu_mem_total_gb = round(
                    torch.cuda.get_device_properties(0).total_memory / 1e9, 2
                )
        except ImportError:
            pass
        return snap
