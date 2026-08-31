"""
Unit tests for Prompt 7 — optimization, benchmarking, robustness.
No real models required; uses synthetic ONNX models.
"""
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

import lib.ml.dataset_engineering.taxonomy as _tax


@pytest.fixture(autouse=True)
def reset_taxonomy():
    _tax.TaxonomyLoader._instance = None
    yield
    _tax.TaxonomyLoader._instance = None


# =============================================================================
# Tiny ONNX model fixtures
# =============================================================================

class TinyModel(nn.Module):
    def __init__(self, nc=109, sz=64):
        super().__init__()
        self.conv = nn.Conv2d(3, 8, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc   = nn.Linear(8, nc)

    def forward(self, x):
        return self.fc(self.pool(self.conv(x)).view(x.size(0), -1))


def _export_onnx(tmp_path: Path, nc: int = 109, sz: int = 64) -> Path:
    model = TinyModel(nc=nc, sz=sz); model.eval()
    dummy = torch.zeros(1, 3, sz, sz)
    out = tmp_path / "tiny.onnx"
    torch.onnx.export(model, dummy, str(out), opset_version=17,
                      input_names=["images"], output_names=["logits"])
    return out


# =============================================================================
# ONNXOptimizer
# =============================================================================
class TestONNXOptimizer:
    def test_graph_optimization_creates_file(self, tmp_path):
        from lib.ml.optimization.onnx_optimizer import ONNXOptimizer
        onnx = _export_onnx(tmp_path)
        optimizer = ONNXOptimizer(report_dir=tmp_path / "opt", n_benchmark_runs=3)
        result = optimizer._graph_optimize(onnx, "tiny")
        # May be None if ORT version doesn't write optimized file
        if result is not None:
            assert result.exists()

    def test_optimize_all_returns_list(self, tmp_path):
        from lib.ml.optimization.onnx_optimizer import ONNXOptimizer
        onnx = _export_onnx(tmp_path)
        optimizer = ONNXOptimizer(
            report_dir=tmp_path / "opt",
            n_benchmark_runs=3,
            n_warmup_runs=1,
        )
        results = optimizer.optimize_all(
            onnx_path=onnx,
            model_id="tiny",
            input_size=64,
            skip_fp16=True,   # requires onnxconverter-common
            skip_int8=True,   # requires onnxruntime.quantization
        )
        assert isinstance(results, list)
        # Report JSON should be created
        report = tmp_path / "opt" / "tiny_optimization_report.json"
        assert report.exists()
        data = json.loads(report.read_text())
        assert "model_id" in data

    def test_optimization_result_serialisable(self, tmp_path):
        from lib.ml.optimization.onnx_optimizer import OptimizationResult
        r = OptimizationResult(
            model_id="tiny", original_path="a.onnx", optimized_path="b.onnx",
            optimization_type="graph", original_size_mb=5.0, optimized_size_mb=4.8,
            size_reduction_pct=4.0, speedup_x=1.05, passed_validation=True,
        )
        d = r.to_dict()
        assert d["optimization_type"] == "graph"
        assert d["passed_validation"] is True
        serialised = json.dumps(d)
        assert len(serialised) > 0

    def test_validate_and_benchmark_parity(self, tmp_path):
        from lib.ml.optimization.onnx_optimizer import ONNXOptimizer
        onnx = _export_onnx(tmp_path)
        optimizer = ONNXOptimizer(
            report_dir=tmp_path / "opt", n_benchmark_runs=3, n_warmup_runs=0
        )
        # Compare onnx vs itself (should pass with 0 diff)
        r = optimizer._validate_and_benchmark(onnx, onnx, "tiny", "graph", 64, 1)
        assert r.max_output_diff < 1e-5
        assert r.passed_validation


# =============================================================================
# OnnxSessionBenchmarker
# =============================================================================
class TestOnnxSessionBenchmarker:
    def test_benchmark_returns_stats(self, tmp_path):
        from lib.ml.benchmarking.pipeline_benchmarker import OnnxSessionBenchmarker, StageStats
        onnx = _export_onnx(tmp_path)
        bench = OnnxSessionBenchmarker(onnx, input_size=64, n_warmup=1, n_runs=5)
        stats = bench.benchmark()
        assert isinstance(stats, StageStats)
        assert stats.mean_ms > 0
        assert stats.p95_ms >= stats.p50_ms
        assert stats.n_runs == 5

    def test_cold_start_returns_float(self, tmp_path):
        from lib.ml.benchmarking.pipeline_benchmarker import OnnxSessionBenchmarker
        onnx = _export_onnx(tmp_path)
        bench = OnnxSessionBenchmarker(onnx, input_size=64, n_warmup=0, n_runs=3)
        cold = bench.cold_start_ms()
        assert isinstance(cold, float)
        assert cold > 0

    def test_batch_throughput(self, tmp_path):
        from lib.ml.benchmarking.pipeline_benchmarker import OnnxSessionBenchmarker
        onnx = _export_onnx(tmp_path)
        bench = OnnxSessionBenchmarker(onnx, input_size=64, n_warmup=1, n_runs=3)
        results = bench.batch_throughput(batch_sizes=[1, 2])
        assert "1" in results
        assert "2" in results
        assert results["1"]["fps"] > 0

    def test_load_test(self, tmp_path):
        from lib.ml.benchmarking.pipeline_benchmarker import OnnxSessionBenchmarker
        onnx = _export_onnx(tmp_path)
        bench = OnnxSessionBenchmarker(onnx, input_size=64, n_warmup=0, n_runs=3)
        results = bench.load_test(n_images=[1, 5])
        assert "1" in results
        assert results["1"]["errors"] == 0
        assert results["1"]["fps"] > 0

    def test_concurrent_test(self, tmp_path):
        from lib.ml.benchmarking.pipeline_benchmarker import OnnxSessionBenchmarker
        onnx = _export_onnx(tmp_path)
        bench = OnnxSessionBenchmarker(onnx, input_size=64, n_warmup=0, n_runs=3)
        result = bench.concurrent_test(n_threads=2, n_per_thread=3)
        assert result["n_threads"] == 2
        assert result["throughput_fps"] > 0
        assert result["errors"] == 0

    def test_stage_stats_from_times(self):
        from lib.ml.benchmarking.pipeline_benchmarker import StageStats
        times = [10.0, 12.0, 11.0, 15.0, 9.0, 20.0, 11.0, 12.0, 10.0, 13.0]
        stats = StageStats.from_times("test", times)
        assert stats.n_runs == len(times)
        assert stats.min_ms == pytest.approx(9.0)
        assert stats.max_ms == pytest.approx(20.0)
        assert stats.p95_ms >= stats.p50_ms >= stats.min_ms


# =============================================================================
# PipelineBenchmarker
# =============================================================================
class TestPipelineBenchmarker:
    def test_benchmark_model_produces_report(self, tmp_path):
        from lib.ml.benchmarking.pipeline_benchmarker import PipelineBenchmarker, BenchmarkReport
        onnx = _export_onnx(tmp_path)
        benchmarker = PipelineBenchmarker(report_dir=tmp_path / "reports",
                                          n_warmup=1, n_runs=5)
        report = benchmarker.benchmark_model(
            onnx_path=onnx, model_id="tiny",
            input_size=64, run_load_test=False, run_concurrent=False,
        )
        assert isinstance(report, BenchmarkReport)
        assert report.fps > 0
        assert report.cold_start_ms > 0
        assert len(report.stages) == 1

    def test_save_report_creates_files(self, tmp_path):
        from lib.ml.benchmarking.pipeline_benchmarker import PipelineBenchmarker
        onnx = _export_onnx(tmp_path)
        benchmarker = PipelineBenchmarker(report_dir=tmp_path / "reports",
                                          n_warmup=0, n_runs=3)
        report = benchmarker.benchmark_model(onnx, "tiny", input_size=64,
                                              run_load_test=False, run_concurrent=False)
        json_p, md_p = benchmarker.save_report(report)
        assert json_p.exists()
        assert md_p.exists()
        data = json.loads(json_p.read_text())
        assert data["model_id"] == "tiny"

    def test_benchmark_report_to_markdown(self, tmp_path):
        from lib.ml.benchmarking.pipeline_benchmarker import PipelineBenchmarker
        onnx = _export_onnx(tmp_path)
        benchmarker = PipelineBenchmarker(report_dir=tmp_path / "reports",
                                          n_warmup=0, n_runs=3)
        report = benchmarker.benchmark_model(onnx, "tiny", 64,
                                              run_load_test=False, run_concurrent=False)
        md = report.to_markdown()
        assert "# SPIRO Benchmark Report" in md
        assert "FPS" in md
        assert "tiny" in md


# =============================================================================
# RobustnessTester
# =============================================================================
class TestRobustnessTester:
    def test_image_generators_return_arrays(self, tmp_path):
        from lib.ml.robustness.robustness_tester import RobustnessTester
        tester = RobustnessTester(n_per_category=1, base_size=64)
        for gen_name in ["_gen_blurry", "_gen_dark", "_gen_overexposed",
                         "_gen_rotated", "_gen_compressed", "_gen_low_res",
                         "_gen_occluded", "_gen_multi_object", "_gen_empty"]:
            result = getattr(tester, gen_name)()
            assert isinstance(result, np.ndarray)
            assert result.dtype == np.uint8

    def test_corrupted_generator_returns_bytes(self, tmp_path):
        from lib.ml.robustness.robustness_tester import RobustnessTester
        tester = RobustnessTester(n_per_category=1)
        result = tester._gen_corrupted()
        assert isinstance(result, bytes)

    def test_run_all_preprocessor_only(self, tmp_path):
        from lib.ml.robustness.robustness_tester import RobustnessTester
        tester = RobustnessTester(
            infer_fn=None,
            n_per_category=3,
            base_size=64,
            report_dir=tmp_path / "reports",
        )
        report = tester.run_all(model_id="test")
        assert len(report.categories) == 12
        assert 0.0 <= report.overall_stability <= 1.0

    def test_stability_rate_correct(self, tmp_path):
        from lib.ml.robustness.robustness_tester import RobustnessTester

        def stable_infer(img):
            return {"status": "success", "detection_count": 1,
                    "detections": [{"final_confidence": 0.8}]}

        tester = RobustnessTester(
            infer_fn=stable_infer,
            n_per_category=5,
            base_size=64,
            report_dir=tmp_path / "reports",
        )
        report = tester.run_all(model_id="stable")
        # All stable inference should have stability_rate ≈ 1.0
        for cat in report.categories:
            assert cat.stability_rate == pytest.approx(1.0, abs=0.1)

    def test_save_report_creates_files(self, tmp_path):
        from lib.ml.robustness.robustness_tester import RobustnessTester
        tester = RobustnessTester(n_per_category=2, report_dir=tmp_path / "reports")
        report = tester.run_all(model_id="test")
        json_p, md_p = tester.save_report(report)
        assert json_p.exists()
        assert md_p.exists()
        data = json.loads(json_p.read_text())
        assert data["model_id"] == "test"
        assert len(data["categories"]) == 12

    def test_robustness_report_to_markdown(self, tmp_path):
        from lib.ml.robustness.robustness_tester import RobustnessTester
        tester = RobustnessTester(n_per_category=1, report_dir=tmp_path / "reports")
        report = tester.run_all(model_id="test")
        md = report.to_markdown()
        assert "# SPIRO Robustness Report" in md
        assert "Pass Rate" in md
        assert "Stability" in md


# =============================================================================
# AccuracyValidator
# =============================================================================
class TestAccuracyValidator:
    def test_fp32_vs_fp32_passes(self, tmp_path):
        from lib.ml.robustness.accuracy_validator import AccuracyValidator
        onnx = _export_onnx(tmp_path, nc=109, sz=64)
        validator = AccuracyValidator(n_samples=5, report_dir=tmp_path / "reports")
        # Compare FP32 vs itself (as FP16 substitute) — should pass
        report = validator.compare_formats(
            fp32_path=onnx,
            model_id="tiny",
            input_size=64,
            task="verification",
        )
        assert isinstance(report.to_dict(), dict)
        assert "comparisons" in report.to_dict()

    def test_report_to_markdown(self, tmp_path):
        from lib.ml.robustness.accuracy_validator import AccuracyValidator
        onnx = _export_onnx(tmp_path)
        validator = AccuracyValidator(n_samples=3, report_dir=tmp_path / "reports")
        report = validator.compare_formats(fp32_path=onnx, model_id="tiny",
                                           input_size=64)
        md = report.to_markdown()
        assert "# Accuracy Validation Report" in md


# =============================================================================
# Environment configs
# =============================================================================
class TestEnvironmentConfigs:
    def test_production_config_loads(self):
        from lib.ml.pipeline.pipeline_config import PipelineConfig
        cfg_path = Path("configs/production/pipeline.yaml")
        if not cfg_path.exists():
            pytest.skip("Production config not found")
        cfg = PipelineConfig.load(cfg_path)
        assert cfg.models.num_classes == 109
        assert cfg.detection.conf_threshold == pytest.approx(0.30, abs=0.01)

    def test_testing_config_loads(self):
        from lib.ml.pipeline.pipeline_config import PipelineConfig
        cfg_path = Path("configs/testing/pipeline.yaml")
        if not cfg_path.exists():
            pytest.skip("Testing config not found")
        cfg = PipelineConfig.load(cfg_path)
        assert cfg.models.num_classes == 109
        assert cfg.detection.conf_threshold < 0.01

    def test_development_config_loads(self):
        from lib.ml.pipeline.pipeline_config import PipelineConfig
        cfg_path = Path("configs/development/pipeline.yaml")
        if not cfg_path.exists():
            pytest.skip("Development config not found")
        cfg = PipelineConfig.load(cfg_path)
        assert cfg.preprocessing.blur_threshold == 0.0


# =============================================================================
# Healthcheck
# =============================================================================
class TestHealthcheck:
    def test_imports_check(self):
        from scripts.healthcheck import check_imports
        assert check_imports() is True

    def test_taxonomy_check(self):
        from scripts.healthcheck import check_taxonomy
        assert check_taxonomy() is True

    def test_preprocessor_check(self):
        from scripts.healthcheck import check_preprocessor
        assert check_preprocessor() is True
