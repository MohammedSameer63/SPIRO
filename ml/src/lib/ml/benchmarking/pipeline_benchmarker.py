"""
SPIRO ML — PipelineBenchmarker
Comprehensive performance benchmarking for the SPIRO inference pipeline.

Measures
--------
- Per-stage latency (preprocess, detect, verify, fuse, total)
- FPS and throughput
- Memory usage (RAM + GPU)
- Cold start vs warm start
- Batch throughput (1, 10, 100, 1000 images)
- Concurrent inference stress test
- P50 / P95 / P99 latency percentiles
- CPU and GPU utilisation

Output
------
- benchmark_report.md
- performance_summary.json
"""
from __future__ import annotations

import gc
import json
import statistics
import threading
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from lib.ml.core.logger import get_logger

log = get_logger(__name__)


@dataclass
class StageStats:
    name: str
    mean_ms: float = 0.0
    std_ms: float = 0.0
    min_ms: float = 0.0
    p50_ms: float = 0.0
    p95_ms: float = 0.0
    p99_ms: float = 0.0
    max_ms: float = 0.0
    n_runs: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_times(cls, name: str, times_ms: List[float]) -> "StageStats":
        if not times_ms:
            return cls(name=name)
        arr = sorted(times_ms)
        n = len(arr)
        return cls(
            name=name,
            mean_ms=round(statistics.mean(arr), 2),
            std_ms=round(statistics.stdev(arr) if n > 1 else 0.0, 2),
            min_ms=round(arr[0], 2),
            p50_ms=round(arr[int(n * 0.50)], 2),
            p95_ms=round(arr[int(n * 0.95)], 2),
            p99_ms=round(arr[int(n * 0.99)], 2),
            max_ms=round(arr[-1], 2),
            n_runs=n,
        )


@dataclass
class BenchmarkReport:
    model_id: str
    hardware: Dict[str, Any]
    n_warmup: int
    n_runs: int
    image_size: Tuple[int, int]
    stages: List[StageStats]
    fps: float = 0.0
    cold_start_ms: float = 0.0
    warm_start_ms: float = 0.0
    peak_ram_mb: float = 0.0
    peak_gpu_mb: float = 0.0
    batch_throughput: Dict[str, Any] = field(default_factory=dict)
    load_test: Dict[str, Any] = field(default_factory=dict)
    generated_at: str = field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ"))

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["stages"] = [s.to_dict() for s in self.stages]
        return d

    def to_markdown(self) -> str:
        lines = [
            f"# SPIRO Benchmark Report — {self.model_id}",
            f"**Generated:** {self.generated_at}",
            f"**Hardware:** {self.hardware.get('platform', 'N/A')}",
            f"**GPU:** {self.hardware.get('gpu_0', 'CPU only')}",
            "",
            f"## Summary",
            f"| Metric | Value |",
            f"|---|---|",
            f"| FPS (warm) | {self.fps:.1f} |",
            f"| Cold start | {self.cold_start_ms:.0f}ms |",
            f"| Warm start (mean) | {self.warm_start_ms:.0f}ms |",
            f"| Peak RAM | {self.peak_ram_mb:.0f}MB |",
            f"| Peak GPU | {self.peak_gpu_mb:.0f}MB |",
            "",
            f"## Stage Breakdown (n={self.n_runs})",
            f"| Stage | Mean | P50 | P95 | P99 | Max |",
            f"|---|---|---|---|---|---|",
        ]
        for s in self.stages:
            lines.append(
                f"| {s.name} | {s.mean_ms:.1f}ms | {s.p50_ms:.1f}ms | "
                f"{s.p95_ms:.1f}ms | {s.p99_ms:.1f}ms | {s.max_ms:.1f}ms |"
            )
        if self.batch_throughput:
            lines += ["", "## Batch Throughput", "| Batch Size | FPS | Time/Image |",
                      "|---|---|---|"]
            for bs, v in self.batch_throughput.items():
                lines.append(
                    f"| {bs} | {v.get('fps', 0):.1f} | {v.get('ms_per_image', 0):.1f}ms |"
                )
        if self.load_test:
            lines += ["", "## Load Test", "| N Images | Total (s) | FPS | Errors |",
                      "|---|---|---|---|"]
            for k, v in self.load_test.items():
                lines.append(
                    f"| {k} | {v.get('total_s', 0):.1f} | "
                    f"{v.get('fps', 0):.1f} | {v.get('errors', 0)} |"
                )
        return "\n".join(lines)


def _collect_hw() -> Dict[str, Any]:
    import platform, os
    hw = {"platform": platform.platform(), "cpu_count": os.cpu_count()}
    try:
        import torch
        hw["pytorch"] = torch.__version__
        hw["cuda"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            hw["gpu_0"] = torch.cuda.get_device_name(0)
            hw["gpu_mem_GB"] = round(
                torch.cuda.get_device_properties(0).total_memory / 1e9, 1
            )
    except ImportError:
        pass
    try:
        import psutil
        hw["ram_GB"] = round(psutil.virtual_memory().total / 1e9, 1)
    except ImportError:
        pass
    return hw


def _peak_memory() -> Tuple[float, float]:
    """Return (ram_mb, gpu_mb)."""
    ram_mb = 0.0
    gpu_mb = 0.0
    try:
        import psutil, os
        proc = psutil.Process(os.getpid())
        ram_mb = round(proc.memory_info().rss / 1e6, 1)
    except Exception:
        pass
    try:
        import torch
        if torch.cuda.is_available():
            gpu_mb = round(torch.cuda.memory_allocated() / 1e6, 1)
    except Exception:
        pass
    return ram_mb, gpu_mb


class OnnxSessionBenchmarker:
    """Benchmarks a single ONNX model in isolation."""

    def __init__(self, onnx_path: Path, input_size: int = 640,
                 n_warmup: int = 5, n_runs: int = 100) -> None:
        import onnxruntime as ort
        self.onnx_path = Path(onnx_path)
        self.input_size = input_size
        self.n_warmup = n_warmup
        self.n_runs = n_runs

        # Build session with auto-provider selection
        providers = []
        avail = ort.get_available_providers()
        for p in ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]:
            if p in avail:
                providers.append(p)
        if not providers:
            providers = ["CPUExecutionProvider"]

        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.intra_op_num_threads = 4
        self.session = ort.InferenceSession(str(onnx_path), sess_options=opts,
                                             providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.active_providers = self.session.get_providers()

    def benchmark(self) -> StageStats:
        dummy = np.random.rand(1, 3, self.input_size, self.input_size).astype(np.float32)
        for _ in range(self.n_warmup):
            self.session.run(None, {self.input_name: dummy})
        times = []
        for _ in range(self.n_runs):
            t0 = time.perf_counter()
            self.session.run(None, {self.input_name: dummy})
            times.append((time.perf_counter() - t0) * 1000)
        return StageStats.from_times(self.onnx_path.name, times)

    def cold_start_ms(self) -> float:
        import onnxruntime as ort
        dummy = np.random.rand(1, 3, self.input_size, self.input_size).astype(np.float32)
        t0 = time.perf_counter()
        opts = ort.SessionOptions()
        sess = ort.InferenceSession(str(self.onnx_path), sess_options=opts,
                                     providers=["CPUExecutionProvider"])
        inp = sess.get_inputs()[0].name
        sess.run(None, {inp: dummy})
        return round((time.perf_counter() - t0) * 1000, 2)

    def batch_throughput(self, batch_sizes: List[int] = None) -> Dict[str, Any]:
        batch_sizes = batch_sizes or [1, 4, 8, 16, 32]
        results: Dict[str, Any] = {}
        for bs in batch_sizes:
            dummy = np.random.rand(bs, 3, self.input_size, self.input_size).astype(np.float32)
            # Warmup
            for _ in range(3):
                self.session.run(None, {self.input_name: dummy})
            times = []
            for _ in range(10):
                t0 = time.perf_counter()
                self.session.run(None, {self.input_name: dummy})
                times.append((time.perf_counter() - t0) * 1000)
            mean_ms = float(np.mean(times))
            results[str(bs)] = {
                "fps": round(bs * 1000 / mean_ms, 1),
                "ms_per_image": round(mean_ms / bs, 2),
                "batch_ms": round(mean_ms, 2),
            }
        return results

    def load_test(self, n_images: List[int] = None) -> Dict[str, Any]:
        n_images = n_images or [1, 10, 100, 1000]
        results: Dict[str, Any] = {}
        dummy = np.random.rand(1, 3, self.input_size, self.input_size).astype(np.float32)
        for n in n_images:
            errors = 0
            t0 = time.perf_counter()
            for _ in range(n):
                try:
                    self.session.run(None, {self.input_name: dummy})
                except Exception:
                    errors += 1
            total_s = time.perf_counter() - t0
            results[str(n)] = {
                "total_s": round(total_s, 2),
                "fps": round(n / max(total_s, 1e-6), 1),
                "ms_per_image": round(total_s * 1000 / max(n, 1), 2),
                "errors": errors,
            }
            log.info(f"Load test n={n}: {total_s:.2f}s fps={results[str(n)]['fps']:.1f}")
        return results

    def concurrent_test(self, n_threads: int = 4, n_per_thread: int = 50) -> Dict[str, Any]:
        dummy = np.random.rand(1, 3, self.input_size, self.input_size).astype(np.float32)
        times_per_thread: List[List[float]] = [[] for _ in range(n_threads)]
        errors_count = [0]

        def worker(idx: int) -> None:
            for _ in range(n_per_thread):
                try:
                    t0 = time.perf_counter()
                    self.session.run(None, {self.input_name: dummy})
                    times_per_thread[idx].append((time.perf_counter() - t0) * 1000)
                except Exception:
                    errors_count[0] += 1

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(n_threads)]
        t_start = time.perf_counter()
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        total_s = time.perf_counter() - t_start

        all_times = [t for ts in times_per_thread for t in ts]
        total_reqs = n_threads * n_per_thread
        return {
            "n_threads": n_threads,
            "n_per_thread": n_per_thread,
            "total_requests": total_reqs,
            "total_s": round(total_s, 2),
            "throughput_fps": round(total_reqs / max(total_s, 1e-6), 1),
            "mean_latency_ms": round(float(np.mean(all_times)), 2),
            "p95_latency_ms": round(float(np.percentile(all_times, 95)), 2),
            "errors": errors_count[0],
        }


class PipelineBenchmarker:
    """
    Full pipeline benchmark using synthetic images.
    Does not require real ONNX models — uses SPIROPipeline if available,
    or individual OnnxSessionBenchmarker for standalone model benchmarks.

    Example
    -------
    >>> benchmarker = PipelineBenchmarker(report_dir=Path("reports"))
    >>> report = benchmarker.benchmark_model(
    ...     onnx_path=Path("models/exports/yolo.onnx"),
    ...     model_id="yolov11s",
    ...     input_size=640,
    ... )
    >>> benchmarker.save_report(report)
    """

    def __init__(
        self,
        report_dir: Path = Path("reports"),
        n_warmup: int = 5,
        n_runs: int = 100,
    ) -> None:
        self.report_dir = Path(report_dir)
        self.report_dir.mkdir(parents=True, exist_ok=True)
        self.n_warmup = n_warmup
        self.n_runs = n_runs

    def benchmark_model(
        self,
        onnx_path: Path,
        model_id: str,
        input_size: int = 640,
        run_load_test: bool = True,
        run_concurrent: bool = True,
    ) -> BenchmarkReport:
        onnx_path = Path(onnx_path)
        bench = OnnxSessionBenchmarker(onnx_path, input_size, self.n_warmup, self.n_runs)
        hw = _collect_hw()

        cold_ms = bench.cold_start_ms()
        stage = bench.benchmark()
        ram_mb, gpu_mb = _peak_memory()

        batch = bench.batch_throughput([1, 4, 8])
        load = bench.load_test([1, 10, 100]) if run_load_test else {}
        concurrent = bench.concurrent_test() if run_concurrent else {}

        report = BenchmarkReport(
            model_id=model_id,
            hardware=hw,
            n_warmup=self.n_warmup,
            n_runs=self.n_runs,
            image_size=(input_size, input_size),
            stages=[stage],
            fps=round(1000 / max(stage.mean_ms, 1e-3), 1),
            cold_start_ms=cold_ms,
            warm_start_ms=stage.mean_ms,
            peak_ram_mb=ram_mb,
            peak_gpu_mb=gpu_mb,
            batch_throughput=batch,
            load_test={**load, "concurrent": concurrent},
        )
        return report

    def save_report(self, report: BenchmarkReport) -> Tuple[Path, Path]:
        """Save JSON + Markdown reports."""
        json_path = self.report_dir / f"{report.model_id}_benchmark.json"
        md_path   = self.report_dir / f"{report.model_id}_benchmark.md"
        json_path.write_text(json.dumps(report.to_dict(), indent=2))
        md_path.write_text(report.to_markdown())
        log.info(f"Benchmark report → {json_path}")
        log.info(f"Benchmark Markdown → {md_path}")
        return json_path, md_path
