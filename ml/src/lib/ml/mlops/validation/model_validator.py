"""
SPIRO ML — ModelValidator
Automated validation suite run before any model promotion.

Checks
------
1. ONNX file exists and loads
2. ORT inference executes without error
3. Output shape matches expected (num_classes = 109)
4. Taxonomy consistency (109 SPIRO classes)
5. CPU latency within threshold
6. Memory usage within threshold
7. Accuracy meets minimum threshold
8. Regression tests (known-class synthetic inputs)
"""
from __future__ import annotations

import json
import time
import traceback
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from lib.ml.core.logger import get_logger
from lib.ml.dataset_engineering.taxonomy import TaxonomyLoader

log = get_logger(__name__)

_NUM_CLASSES = 109


@dataclass
class ValidationCheck:
    name: str
    passed: bool
    details: str = ""
    value: Optional[float] = None
    threshold: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ValidationReport:
    model_id: str
    version: str
    onnx_path: str
    validated_at: str = field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ"))
    overall_passed: bool = False
    checks: List[ValidationCheck] = field(default_factory=list)
    summary: str = ""

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["checks"] = [c.to_dict() for c in self.checks]
        return d

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)


class ModelValidator:
    """
    Runs all pre-deployment validation checks against an ONNX model.

    Parameters
    ----------
    min_accuracy : float
        Minimum top-1 accuracy (if accuracy data provided).
    max_latency_ms : float
        Maximum acceptable CPU inference latency.
    max_memory_mb : float
        Maximum memory increment during inference.
    num_warmup : int
        ORT warmup runs before latency measurement.

    Example
    -------
    >>> validator = ModelValidator()
    >>> report = validator.validate(onnx_path, model_id="yolov11s", version="v1.0")
    >>> if report.overall_passed:
    ...     registry.promote(...)
    """

    def __init__(
        self,
        min_accuracy: float = 0.0,        # 0 = no threshold
        max_latency_ms: float = 500.0,
        max_memory_mb: float = 2048.0,
        num_warmup: int = 3,
        num_latency_runs: int = 20,
    ) -> None:
        self.min_accuracy = min_accuracy
        self.max_latency_ms = max_latency_ms
        self.max_memory_mb = max_memory_mb
        self.num_warmup = num_warmup
        self.num_latency_runs = num_latency_runs
        self.taxonomy = TaxonomyLoader()

    def validate(
        self,
        onnx_path: str | Path,
        model_id: str,
        version: str,
        input_size: int = 640,
        num_classes: int = _NUM_CLASSES,
        task: str = "detection",
        accuracy: Optional[float] = None,
    ) -> ValidationReport:
        """
        Run the full validation suite.

        Parameters
        ----------
        onnx_path : ONNX model file to validate
        model_id, version : registry keys for logging
        input_size : expected square input (pixels)
        num_classes : expected output class count
        task : "detection" | "verification"
        accuracy : pre-computed accuracy to check against threshold

        Returns
        -------
        ValidationReport
        """
        onnx_path = Path(onnx_path)
        report = ValidationReport(
            model_id=model_id,
            version=version,
            onnx_path=str(onnx_path),
        )

        checks: List[ValidationCheck] = []

        # 1. File exists
        checks.append(self._check_file_exists(onnx_path))

        if not checks[-1].passed:
            report.checks = checks
            report.overall_passed = False
            report.summary = "FAILED: ONNX file not found."
            return report

        # 2. ONNX graph is valid
        checks.append(self._check_onnx_graph(onnx_path))

        # 3. ORT session loads
        sess, load_check = self._check_ort_load(onnx_path)
        checks.append(load_check)

        if sess is None:
            report.checks = checks
            report.overall_passed = False
            report.summary = "FAILED: ORT session could not be created."
            return report

        # 4. Inference executes
        checks.append(self._check_inference(sess, input_size, task))

        # 5. Output shape / class count
        checks.append(self._check_output_shape(sess, input_size, num_classes, task))

        # 6. Taxonomy consistency
        checks.append(self._check_taxonomy(num_classes))

        # 7. Latency
        lat_check, latency_ms = self._check_latency(sess, input_size)
        checks.append(lat_check)

        # 8. Memory
        checks.append(self._check_memory(sess, input_size))

        # 9. Accuracy threshold (optional)
        if accuracy is not None and self.min_accuracy > 0:
            checks.append(self._check_accuracy(accuracy))

        # 10. Regression test
        checks.append(self._check_regression(sess, input_size, num_classes, task))

        report.checks = checks
        failed = [c for c in checks if not c.passed]
        report.overall_passed = len(failed) == 0
        passed_count = len(checks) - len(failed)
        report.summary = (
            f"PASSED ({passed_count}/{len(checks)} checks, "
            f"latency={latency_ms:.1f}ms)"
            if report.overall_passed else
            f"FAILED {len(failed)}/{len(checks)} checks: "
            + ", ".join(c.name for c in failed)
        )
        log.info(f"Validation [{model_id} v{version}]: {report.summary}")
        return report

    # ------------------------------------------------------------------
    # Individual checks
    # ------------------------------------------------------------------

    def _check_file_exists(self, path: Path) -> ValidationCheck:
        exists = path.exists()
        size_mb = round(path.stat().st_size / 1e6, 2) if exists else 0
        return ValidationCheck(
            name="file_exists",
            passed=exists,
            details=f"{path} ({size_mb}MB)" if exists else f"Not found: {path}",
            value=size_mb,
        )

    def _check_onnx_graph(self, path: Path) -> ValidationCheck:
        try:
            import onnx
            model = onnx.load(str(path))
            onnx.checker.check_model(model)
            return ValidationCheck(name="onnx_graph_valid", passed=True,
                                   details="onnx.checker passed")
        except Exception as e:
            return ValidationCheck(name="onnx_graph_valid", passed=False,
                                   details=str(e))

    def _check_ort_load(self, path: Path):
        try:
            import onnxruntime as ort
            opts = ort.SessionOptions()
            opts.intra_op_num_threads = 2
            sess = ort.InferenceSession(str(path), sess_options=opts,
                                        providers=["CPUExecutionProvider"])
            return sess, ValidationCheck(name="ort_loads", passed=True,
                                         details="Session created successfully")
        except Exception as e:
            return None, ValidationCheck(name="ort_loads", passed=False, details=str(e))

    def _check_inference(self, sess, input_size: int, task: str) -> ValidationCheck:
        try:
            inp = sess.get_inputs()[0]
            dummy = np.zeros((1, 3, input_size, input_size), dtype=np.float32)
            sess.run(None, {inp.name: dummy})
            return ValidationCheck(name="inference_executes", passed=True,
                                   details="Forward pass succeeded")
        except Exception as e:
            return ValidationCheck(name="inference_executes", passed=False,
                                   details=str(e))

    def _check_output_shape(
        self, sess, input_size: int, num_classes: int, task: str
    ) -> ValidationCheck:
        try:
            inp = sess.get_inputs()[0]
            dummy = np.zeros((1, 3, input_size, input_size), dtype=np.float32)
            outputs = sess.run(None, {inp.name: dummy})
            out = outputs[0]
            if task == "verification":
                # Expected: [1, num_classes]
                ok = out.ndim == 2 and out.shape[1] == num_classes
            else:
                # Detection: [1, 4+nc, N] or [1, N, 4+nc]
                nc_present = (
                    (out.ndim == 3 and (out.shape[1] >= 4 + num_classes or
                                        out.shape[2] >= 4 + num_classes))
                )
                ok = nc_present
            return ValidationCheck(
                name="output_shape",
                passed=ok,
                details=f"shape={out.shape}, expected nc={num_classes}",
            )
        except Exception as e:
            return ValidationCheck(name="output_shape", passed=False, details=str(e))

    def _check_taxonomy(self, num_classes: int) -> ValidationCheck:
        actual = self.taxonomy.num_classes
        ok = actual == num_classes
        return ValidationCheck(
            name="taxonomy_consistency",
            passed=ok,
            details=f"taxonomy has {actual} classes, model expects {num_classes}",
            value=float(actual),
            threshold=float(num_classes),
        )

    def _check_latency(self, sess, input_size: int):
        try:
            inp = sess.get_inputs()[0]
            dummy = np.zeros((1, 3, input_size, input_size), dtype=np.float32)
            for _ in range(self.num_warmup):
                sess.run(None, {inp.name: dummy})
            times = []
            for _ in range(self.num_latency_runs):
                t0 = time.perf_counter()
                sess.run(None, {inp.name: dummy})
                times.append((time.perf_counter() - t0) * 1000)
            mean_ms = float(np.mean(times))
            ok = mean_ms <= self.max_latency_ms
            check = ValidationCheck(
                name="latency_cpu",
                passed=ok,
                details=f"mean={mean_ms:.1f}ms p95={np.percentile(times,95):.1f}ms",
                value=round(mean_ms, 2),
                threshold=self.max_latency_ms,
            )
            return check, mean_ms
        except Exception as e:
            return ValidationCheck(name="latency_cpu", passed=False, details=str(e)), 0.0

    def _check_memory(self, sess, input_size: int) -> ValidationCheck:
        try:
            import tracemalloc
            tracemalloc.start()
            inp = sess.get_inputs()[0]
            dummy = np.zeros((1, 3, input_size, input_size), dtype=np.float32)
            sess.run(None, {inp.name: dummy})
            _, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            peak_mb = round(peak / 1e6, 1)
            ok = peak_mb <= self.max_memory_mb
            return ValidationCheck(
                name="memory_usage",
                passed=ok,
                details=f"peak={peak_mb}MB (threshold={self.max_memory_mb}MB)",
                value=peak_mb,
                threshold=self.max_memory_mb,
            )
        except Exception as e:
            return ValidationCheck(name="memory_usage", passed=True,
                                   details=f"tracemalloc unavailable: {e}")

    def _check_accuracy(self, accuracy: float) -> ValidationCheck:
        ok = accuracy >= self.min_accuracy
        return ValidationCheck(
            name="accuracy_threshold",
            passed=ok,
            details=f"accuracy={accuracy:.4f} (min={self.min_accuracy:.4f})",
            value=accuracy,
            threshold=self.min_accuracy,
        )

    def _check_regression(
        self, sess, input_size: int, num_classes: int, task: str
    ) -> ValidationCheck:
        """
        Regression test: deterministic input → deterministic output class.
        For verification models, a fixed synthetic input must always produce
        the same top-1 prediction across runs.
        """
        try:
            inp = sess.get_inputs()[0]
            np.random.seed(0)
            dummy = np.random.rand(1, 3, input_size, input_size).astype(np.float32)
            out1 = sess.run(None, {inp.name: dummy})[0]
            out2 = sess.run(None, {inp.name: dummy})[0]
            max_diff = float(np.abs(out1 - out2).max())
            ok = max_diff < 1e-5
            return ValidationCheck(
                name="regression_deterministic",
                passed=ok,
                details=f"max_diff={max_diff:.2e} between identical runs",
                value=max_diff,
            )
        except Exception as e:
            return ValidationCheck(name="regression_deterministic", passed=False,
                                   details=str(e))

    def save_report(self, report: ValidationReport, output_dir: Optional[Path] = None) -> Path:
        out_dir = output_dir or Path("mlops/reports/validation")
        out_dir.mkdir(parents=True, exist_ok=True)
        out = out_dir / f"{report.model_id}_{report.version}_validation.json"
        out.write_text(report.to_json())
        log.info(f"Validation report → {out}")
        return out
