"""
SPIRO ML — ONNXOptimizer
Production ONNX model optimization pipeline:
  1. Graph optimization (constant folding, dead node elimination, operator fusion)
  2. FP16 conversion (CUDA-only)
  3. INT8 dynamic quantization (CPU)
  4. INT8 static quantization with calibration dataset
  5. Pre/post accuracy validation
  6. Optimization report generation

References
----------
- ONNX Runtime graph optimization: ORT_ENABLE_ALL
- onnxruntime.quantization for INT8/FP16
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
class OptimizationResult:
    model_id: str
    original_path: str
    optimized_path: str
    optimization_type: str        # "graph" | "fp16" | "int8_dynamic" | "int8_static"
    original_size_mb: float = 0.0
    optimized_size_mb: float = 0.0
    size_reduction_pct: float = 0.0
    original_latency_ms: float = 0.0
    optimized_latency_ms: float = 0.0
    speedup_x: float = 1.0
    max_output_diff: float = 0.0
    accuracy_delta: float = 0.0
    passed_validation: bool = False
    generated_at: str = field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ"))
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def summary(self) -> str:
        return (
            f"{self.optimization_type}: "
            f"size {self.original_size_mb:.1f}→{self.optimized_size_mb:.1f}MB "
            f"({self.size_reduction_pct:.1f}% reduction), "
            f"latency {self.original_latency_ms:.1f}→{self.optimized_latency_ms:.1f}ms "
            f"({self.speedup_x:.2f}x speedup), "
            f"max_diff={self.max_output_diff:.6f}, "
            f"{'PASSED' if self.passed_validation else 'FAILED'}"
        )


class ONNXOptimizer:
    """
    Multi-strategy ONNX optimization pipeline.

    Parameters
    ----------
    report_dir : Path
    max_diff_threshold : float
        Maximum allowable output difference for validation.
    n_benchmark_runs : int
    n_warmup_runs : int

    Example
    -------
    >>> optimizer = ONNXOptimizer()
    >>> results = optimizer.optimize_all(
    ...     onnx_path=Path("models/exports/yolo_best.onnx"),
    ...     model_id="yolov11s",
    ...     input_size=640,
    ... )
    """

    def __init__(
        self,
        report_dir: Path = Path("reports/optimization"),
        max_diff_threshold: float = 0.01,
        n_benchmark_runs: int = 30,
        n_warmup_runs: int = 5,
    ) -> None:
        self.report_dir = Path(report_dir)
        self.report_dir.mkdir(parents=True, exist_ok=True)
        self.max_diff = max_diff_threshold
        self.n_runs = n_benchmark_runs
        self.n_warmup = n_warmup_runs

    # ------------------------------------------------------------------
    # Full optimization suite
    # ------------------------------------------------------------------

    def optimize_all(
        self,
        onnx_path: Path,
        model_id: str,
        input_size: int = 640,
        batch_size: int = 1,
        calibration_data: Optional[np.ndarray] = None,
        skip_int8: bool = False,
        skip_fp16: bool = False,
    ) -> List[OptimizationResult]:
        """
        Run all applicable optimization strategies.

        Returns
        -------
        List[OptimizationResult] — one per strategy applied
        """
        onnx_path = Path(onnx_path)
        results: List[OptimizationResult] = []

        log.info(f"Starting optimization suite: {onnx_path.name}")

        # 1. Graph optimization
        graph_path = self._graph_optimize(onnx_path, model_id)
        if graph_path:
            r = self._validate_and_benchmark(
                onnx_path, graph_path, model_id, "graph", input_size, batch_size
            )
            results.append(r)
            log.info(f"Graph opt: {r.summary()}")

        # Use graph-optimized as baseline for further optimization
        base = graph_path if (graph_path and graph_path.exists()) else onnx_path

        # 2. FP16 (CUDA only)
        if not skip_fp16:
            fp16_path = self._fp16_convert(base, model_id)
            if fp16_path:
                r = self._validate_and_benchmark(
                    onnx_path, fp16_path, model_id, "fp16", input_size, batch_size
                )
                results.append(r)
                log.info(f"FP16: {r.summary()}")

        # 3. INT8 dynamic quantization
        if not skip_int8:
            int8_dyn_path = self._int8_dynamic(base, model_id)
            if int8_dyn_path:
                r = self._validate_and_benchmark(
                    onnx_path, int8_dyn_path, model_id, "int8_dynamic", input_size, batch_size,
                    tolerance=0.05
                )
                results.append(r)
                log.info(f"INT8-dynamic: {r.summary()}")

            # 4. INT8 static quantization (requires calibration data)
            if calibration_data is not None:
                int8_stat_path = self._int8_static(base, model_id, calibration_data, input_size)
                if int8_stat_path:
                    r = self._validate_and_benchmark(
                        onnx_path, int8_stat_path, model_id, "int8_static", input_size, batch_size,
                        tolerance=0.08
                    )
                    results.append(r)
                    log.info(f"INT8-static: {r.summary()}")

        self._save_report(model_id, results)
        return results

    # ------------------------------------------------------------------
    # Strategy implementations
    # ------------------------------------------------------------------

    def _graph_optimize(self, src: Path, model_id: str) -> Optional[Path]:
        """ORT graph optimization: constant folding + dead node elim + op fusion."""
        try:
            import onnxruntime as ort
            from onnxruntime.transformers.optimizer import optimize_model
        except ImportError:
            log.warning("onnxruntime.transformers not available — using basic optimization")

        try:
            import onnxruntime as ort
            dst = self.report_dir / f"{model_id}_graph_opt.onnx"
            so = ort.SessionOptions()
            so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            so.optimized_model_filepath = str(dst)
            sess = ort.InferenceSession(str(src), sess_options=so,
                                        providers=["CPUExecutionProvider"])
            del sess
            if dst.exists() and dst.stat().st_size > 0:
                log.info(f"Graph optimization → {dst}")
                return dst
        except Exception as e:
            log.warning(f"Graph optimization failed: {e}")
        return None

    def _fp16_convert(self, src: Path, model_id: str) -> Optional[Path]:
        """Convert float32 ONNX to float16."""
        try:
            from onnxconverter_common import float16
            import onnx
            dst = self.report_dir / f"{model_id}_fp16.onnx"
            model = onnx.load(str(src))
            model_fp16 = float16.convert_float_to_float16(model, keep_io_types=True)
            onnx.save(model_fp16, str(dst))
            log.info(f"FP16 conversion → {dst}")
            return dst
        except ImportError:
            log.warning("onnxconverter-common not installed — skipping FP16")
        except Exception as e:
            log.warning(f"FP16 conversion failed: {e}")
        return None

    def _int8_dynamic(self, src: Path, model_id: str) -> Optional[Path]:
        """Dynamic INT8 quantization (no calibration data needed)."""
        try:
            from onnxruntime.quantization import quantize_dynamic, QuantType
            dst = self.report_dir / f"{model_id}_int8_dynamic.onnx"
            quantize_dynamic(
                model_input=str(src),
                model_output=str(dst),
                weight_type=QuantType.QUInt8,
                optimize_model=True,
            )
            log.info(f"INT8-dynamic quantization → {dst}")
            return dst
        except ImportError:
            log.warning("onnxruntime.quantization not available")
        except Exception as e:
            log.warning(f"INT8-dynamic failed: {e}")
        return None

    def _int8_static(
        self,
        src: Path,
        model_id: str,
        calibration_data: np.ndarray,
        input_size: int,
    ) -> Optional[Path]:
        """Static INT8 quantization with calibration dataset."""
        try:
            from onnxruntime.quantization import (
                quantize_static, CalibrationDataReader,
                QuantType, QuantFormat,
            )
            import onnxruntime as ort

            class CalibReader(CalibrationDataReader):
                def __init__(self, data: np.ndarray, input_name: str):
                    self.data = iter(
                        {input_name: data[i:i+1]}
                        for i in range(min(len(data), 100))
                    )
                def get_next(self):
                    return next(self.data, None)

            # Get input name
            sess = ort.InferenceSession(str(src), providers=["CPUExecutionProvider"])
            input_name = sess.get_inputs()[0].name
            del sess

            dst = self.report_dir / f"{model_id}_int8_static.onnx"
            quantize_static(
                model_input=str(src),
                model_output=str(dst),
                calibration_data_reader=CalibReader(calibration_data, input_name),
                quant_format=QuantFormat.QDQ,
                per_channel=True,
                weight_type=QuantType.QUInt8,
            )
            log.info(f"INT8-static quantization → {dst}")
            return dst
        except ImportError:
            log.warning("onnxruntime.quantization not available")
        except Exception as e:
            log.warning(f"INT8-static failed: {e}")
        return None

    # ------------------------------------------------------------------
    # Validation and benchmarking
    # ------------------------------------------------------------------

    def _validate_and_benchmark(
        self,
        original: Path,
        optimized: Path,
        model_id: str,
        opt_type: str,
        input_size: int,
        batch_size: int,
        tolerance: float = None,
    ) -> OptimizationResult:
        tolerance = tolerance or self.max_diff
        result = OptimizationResult(
            model_id=model_id,
            original_path=str(original),
            optimized_path=str(optimized),
            optimization_type=opt_type,
            original_size_mb=round(original.stat().st_size / 1e6, 2),
            optimized_size_mb=round(optimized.stat().st_size / 1e6, 2),
        )
        result.size_reduction_pct = round(
            (1 - result.optimized_size_mb / max(result.original_size_mb, 1e-6)) * 100, 1
        )

        try:
            import onnxruntime as ort

            providers = ["CPUExecutionProvider"]
            orig_sess = ort.InferenceSession(str(original), providers=providers)
            opt_sess  = ort.InferenceSession(str(optimized), providers=providers)
            inp_name  = orig_sess.get_inputs()[0].name

            dtype = np.float16 if opt_type == "fp16" else np.float32
            dummy = np.random.rand(batch_size, 3, input_size, input_size).astype(dtype)
            dummy_f32 = dummy.astype(np.float32)

            # Numerical validation
            orig_out = orig_sess.run(None, {inp_name: dummy_f32})[0]
            opt_dummy = dummy if opt_type == "fp16" else dummy_f32
            opt_out = opt_sess.run(None, {opt_sess.get_inputs()[0].name: opt_dummy})[0]

            if orig_out.shape == opt_out.shape:
                result.max_output_diff = float(np.abs(orig_out.astype(np.float32) - opt_out.astype(np.float32)).max())
            result.passed_validation = result.max_output_diff <= tolerance

            # Latency benchmark
            result.original_latency_ms  = self._bench_latency(orig_sess, inp_name, dummy_f32)
            opt_inp_name = opt_sess.get_inputs()[0].name
            result.optimized_latency_ms = self._bench_latency(opt_sess, opt_inp_name, opt_dummy)
            result.speedup_x = round(
                result.original_latency_ms / max(result.optimized_latency_ms, 0.001), 3
            )

        except Exception as e:
            result.notes = str(e)
            result.passed_validation = False
            log.warning(f"Validation failed for {opt_type}: {e}")

        return result

    def _bench_latency(self, sess, input_name: str, dummy: np.ndarray) -> float:
        for _ in range(self.n_warmup):
            sess.run(None, {input_name: dummy})
        times = []
        for _ in range(self.n_runs):
            t0 = time.perf_counter()
            sess.run(None, {input_name: dummy})
            times.append((time.perf_counter() - t0) * 1000)
        return round(float(np.mean(times)), 2)

    def _save_report(self, model_id: str, results: List[OptimizationResult]) -> Path:
        out = self.report_dir / f"{model_id}_optimization_report.json"
        data = {
            "model_id": model_id,
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "results": [r.to_dict() for r in results],
            "best_size_reduction": max(
                (r.size_reduction_pct for r in results if r.passed_validation), default=0.0
            ),
            "best_speedup": max(
                (r.speedup_x for r in results if r.passed_validation), default=1.0
            ),
        }
        out.write_text(json.dumps(data, indent=2))
        log.info(f"Optimization report → {out}")
        return out
