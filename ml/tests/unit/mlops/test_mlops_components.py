"""
Unit tests for the SPIRO MLOps platform.
No real model files or training required.
"""
import json
import sys
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

import lib.ml.dataset_engineering.taxonomy as _tax


@pytest.fixture(autouse=True)
def reset_taxonomy():
    _tax.TaxonomyLoader._instance = None
    yield
    _tax.TaxonomyLoader._instance = None


# =============================================================================
# ModelMetadata
# =============================================================================
class TestModelMetadata:
    def test_serialise_roundtrip(self):
        from lib.ml.mlops.registry.model_metadata import ModelMetadata, ModelMetrics
        meta = ModelMetadata(
            model_id="yolov11s", version="v1.0.0",
            architecture="yolo11s", task="detection",
            dataset_version_id="ds_v1", experiment_id="exp_001",
            metrics=ModelMetrics(mAP50_95=0.72, mAP50=0.85),
        )
        j = meta.to_json()
        restored = ModelMetadata.from_json(j)
        assert restored.model_id == "yolov11s"
        assert abs(restored.metrics.mAP50_95 - 0.72) < 1e-6

    def test_is_better_than(self):
        from lib.ml.mlops.registry.model_metadata import ModelMetadata, ModelMetrics
        better = ModelMetadata(
            model_id="m", version="v2", architecture="a", task="t",
            dataset_version_id="d", experiment_id="e",
            metrics=ModelMetrics(mAP50_95=0.80),
        )
        worse = ModelMetadata(
            model_id="m", version="v1", architecture="a", task="t",
            dataset_version_id="d", experiment_id="e",
            metrics=ModelMetrics(mAP50_95=0.72),
        )
        assert better.is_better_than(worse)
        assert not worse.is_better_than(better)


# =============================================================================
# ModelRegistry
# =============================================================================
class TestModelRegistry:
    def test_register_and_get(self, tmp_path):
        from lib.ml.mlops.registry.model_registry import ModelRegistry
        from lib.ml.mlops.registry.model_metadata import ModelMetadata, ModelMetrics
        registry = ModelRegistry(root=tmp_path / "models")
        meta = ModelMetadata(
            model_id="yolov11s", version="v1.0.0",
            architecture="yolo11s", task="detection",
            dataset_version_id="ds1", experiment_id="exp1",
            metrics=ModelMetrics(mAP50_95=0.72),
        )
        registry.register(meta)
        retrieved = registry.get("yolov11s", "v1.0.0")
        assert retrieved is not None
        assert retrieved.version == "v1.0.0"

    def test_promote_to_production(self, tmp_path):
        from lib.ml.mlops.registry.model_registry import ModelRegistry
        from lib.ml.mlops.registry.model_metadata import ModelMetadata, ModelMetrics
        registry = ModelRegistry(root=tmp_path / "models")
        meta = ModelMetadata(
            model_id="yolov11s", version="v1.0.0",
            architecture="yolo11s", task="detection",
            dataset_version_id="ds1", experiment_id="exp1",
        )
        registry.register(meta)
        registry.promote("yolov11s", "v1.0.0", "production")
        prod = registry.get_production_model("yolov11s")
        assert prod is not None
        assert prod.stage == "production"

    def test_promote_archives_previous(self, tmp_path):
        from lib.ml.mlops.registry.model_registry import ModelRegistry
        from lib.ml.mlops.registry.model_metadata import ModelMetadata, ModelMetrics
        registry = ModelRegistry(root=tmp_path / "models")
        for ver in ["v1.0.0", "v2.0.0"]:
            meta = ModelMetadata(
                model_id="yolov11s", version=ver,
                architecture="yolo11s", task="detection",
                dataset_version_id="ds1", experiment_id="exp1",
            )
            registry.register(meta)
        registry.promote("yolov11s", "v1.0.0", "production")
        registry.promote("yolov11s", "v2.0.0", "production")
        old = registry.get("yolov11s", "v1.0.0")
        assert old.stage == "archived"

    def test_list_models(self, tmp_path):
        from lib.ml.mlops.registry.model_registry import ModelRegistry
        from lib.ml.mlops.registry.model_metadata import ModelMetadata, ModelMetrics
        registry = ModelRegistry(root=tmp_path / "models")
        for mid in ["yolov11s", "effnetv2_s"]:
            meta = ModelMetadata(
                model_id=mid, version="v1.0.0",
                architecture=mid, task="detection",
                dataset_version_id="ds1", experiment_id="e1",
            )
            registry.register(meta)
        models = registry.list_models()
        assert "yolov11s" in models
        assert "effnetv2_s" in models

    def test_export_registry_json(self, tmp_path):
        from lib.ml.mlops.registry.model_registry import ModelRegistry
        from lib.ml.mlops.registry.model_metadata import ModelMetadata
        registry = ModelRegistry(root=tmp_path / "models")
        meta = ModelMetadata(
            model_id="m", version="v1", architecture="a", task="t",
            dataset_version_id="d", experiment_id="e",
        )
        registry.register(meta)
        out = registry.export_registry_json(tmp_path / "registry.json")
        data = json.loads(out.read_text())
        assert data["total"] == 1


# =============================================================================
# DatasetRegistry
# =============================================================================
class TestDatasetRegistry:
    def test_register_and_get(self, tmp_path):
        from lib.ml.mlops.registry.dataset_registry import DatasetRegistry
        from lib.ml.mlops.registry.dataset_version import DatasetVersion
        registry = DatasetRegistry(root=tmp_path / "datasets")
        dv = DatasetVersion(
            dataset_id="spiro_v1", version="v1.0.0",
            image_count=1000, annotation_count=3500,
        )
        registry.register(dv)
        retrieved = registry.get("spiro_v1", "v1.0.0")
        assert retrieved is not None
        assert retrieved.image_count == 1000

    def test_next_version_increments(self, tmp_path):
        from lib.ml.mlops.registry.dataset_registry import DatasetRegistry
        from lib.ml.mlops.registry.dataset_version import DatasetVersion
        registry = DatasetRegistry(root=tmp_path / "datasets")
        for ver in ["v1.0.0", "v1.0.1"]:
            dv = DatasetVersion(dataset_id="ds", version=ver, image_count=10)
            registry.register(dv)
        next_v = registry.next_version("ds")
        assert next_v == "v1.0.2"

    def test_get_latest(self, tmp_path):
        from lib.ml.mlops.registry.dataset_registry import DatasetRegistry
        from lib.ml.mlops.registry.dataset_version import DatasetVersion
        registry = DatasetRegistry(root=tmp_path / "datasets")
        for ver in ["v1.0.0", "v1.0.1", "v1.0.2"]:
            dv = DatasetVersion(dataset_id="ds", version=ver, image_count=10)
            registry.register(dv)
        latest = registry.get_latest("ds")
        assert latest.version == "v1.0.2"

    def test_checksum_computed(self, tmp_path):
        from lib.ml.mlops.registry.dataset_version import DatasetVersion
        dv = DatasetVersion(dataset_id="ds", version="v1", image_count=100,
                            annotation_count=300)
        checksum = dv.compute_checksum()
        assert len(checksum) == 64  # sha256 hex
        assert dv.checksum == checksum


# =============================================================================
# ExperimentTracker
# =============================================================================
class TestExperimentTracker:
    def test_start_and_end_run(self, tmp_path):
        from lib.ml.mlops.registry.experiment_tracker import ExperimentTracker
        tracker = ExperimentTracker(experiments_dir=tmp_path / "experiments")
        exp_id = tracker.start_run(
            "test_exp", "yolov11s", "v1.0.0", "ds_v1.0.0",
            hyperparameters={"lr": 0.01},
        )
        assert exp_id.startswith("exp_")
        tracker.log_metrics(exp_id, {"mAP50": 0.72, "val_loss": 0.43})
        tracker.end_run(exp_id, status="completed")
        rec = tracker.get(exp_id)
        assert rec is not None
        assert rec.status == "completed"
        assert abs(rec.metrics.get("mAP50", 0) - 0.72) < 1e-6

    def test_list_experiments_by_model(self, tmp_path):
        from lib.ml.mlops.registry.experiment_tracker import ExperimentTracker
        tracker = ExperimentTracker(experiments_dir=tmp_path / "experiments")
        for i in range(3):
            eid = tracker.start_run(f"exp_{i}", "yolov11s", f"v{i}", "ds1")
            tracker.end_run(eid)
        records = tracker.list_experiments(model_id="yolov11s")
        assert len(records) == 3

    def test_export_history(self, tmp_path):
        from lib.ml.mlops.registry.experiment_tracker import ExperimentTracker
        tracker = ExperimentTracker(experiments_dir=tmp_path / "experiments")
        eid = tracker.start_run("exp", "m", "v1", "d1")
        tracker.end_run(eid)
        out = tracker.export_history_json(tmp_path / "hist.json")
        data = json.loads(out.read_text())
        assert data["total"] >= 1


# =============================================================================
# MetricsStore
# =============================================================================
class TestMetricsStore:
    def test_record_and_read(self, tmp_path):
        from lib.ml.mlops.monitoring.metrics_store import MetricsStore
        store = MetricsStore(root=tmp_path / "metrics")
        for i in range(5):
            store.record("yolov11s", "mAP50_95", 0.70 + i * 0.01, version="v1.0.0")
        vals = store.values("yolov11s", "mAP50_95")
        assert len(vals) == 5
        assert abs(vals[-1] - 0.74) < 1e-5

    def test_trend_improving(self, tmp_path):
        from lib.ml.mlops.monitoring.metrics_store import MetricsStore
        store = MetricsStore(root=tmp_path / "metrics")
        for i in range(10):
            store.record("m", "acc", 0.5 + i * 0.02)
        t = store.trend("m", "acc", last_n=10)
        assert t["trend"] == "improving"
        assert t["n"] == 10

    def test_health_report(self, tmp_path):
        from lib.ml.mlops.monitoring.metrics_store import MetricsStore
        store = MetricsStore(root=tmp_path / "metrics")
        for v in [0.70, 0.71, 0.72, 0.73]:
            store.record("m", "mAP50_95", v)
        report = store.health_report("m", metrics=["mAP50_95"])
        assert "health_score" in report
        assert report["status"] in {"healthy", "degrading", "critical"}


# =============================================================================
# DriftDetector
# =============================================================================
class TestDriftDetector:
    def test_no_drift_on_same_distribution(self, tmp_path):
        from lib.ml.mlops.drift.drift_detector import DriftDetector
        detector = DriftDetector(report_dir=tmp_path / "drift")
        np.random.seed(42)
        baseline = np.random.uniform(0.6, 0.9, 200)
        current  = np.random.uniform(0.6, 0.9, 200)
        detector.set_baseline(confidences=baseline)
        report = detector.detect(confidences=current, model_id="yolov11s")
        # PSI on same distribution should be < 0.1
        conf_result = next(r for r in report.results if r.drift_type == "confidence_drift")
        assert conf_result.psi < 0.15   # small tolerance

    def test_drift_detected_on_shifted_distribution(self, tmp_path):
        from lib.ml.mlops.drift.drift_detector import DriftDetector
        detector = DriftDetector(report_dir=tmp_path / "drift")
        np.random.seed(0)
        baseline = np.random.uniform(0.7, 0.95, 500)
        current  = np.random.uniform(0.1, 0.4, 500)  # big shift
        detector.set_baseline(confidences=baseline)
        report = detector.detect(confidences=current, model_id="yolov11s")
        assert report.any_drifted

    def test_report_structure(self, tmp_path):
        from lib.ml.mlops.drift.drift_detector import DriftDetector
        detector = DriftDetector(report_dir=tmp_path / "drift")
        baseline = np.random.uniform(0.6, 0.9, 100)
        detector.set_baseline(confidences=baseline)
        report = detector.detect(confidences=np.random.uniform(0.6, 0.9, 50))
        d = report.to_dict()
        for key in ["overall_drift_score", "overall_severity", "any_drifted", "recommendation"]:
            assert key in d

    def test_save_and_load_baseline(self, tmp_path):
        from lib.ml.mlops.drift.drift_detector import DriftDetector
        d1 = DriftDetector(report_dir=tmp_path / "drift")
        baseline = np.random.uniform(0.7, 0.9, 100)
        d1.set_baseline(confidences=baseline)
        bpath = tmp_path / "baseline.json"
        d1.save_baseline(bpath)
        d2 = DriftDetector(baseline_path=bpath, report_dir=tmp_path / "drift")
        assert d2._baseline is not None
        assert abs(d2._baseline["mean_confidence"] - float(baseline.mean())) < 0.01


# =============================================================================
# ConceptDriftDetector
# =============================================================================
class TestConceptDriftDetector:
    def test_no_drift_stable_stream(self):
        from lib.ml.mlops.drift.concept_drift import ConceptDriftDetector
        detector = ConceptDriftDetector(ph_lambda=200.0, cusum_h=50.0)
        np.random.seed(42)
        baseline = np.random.normal(0.8, 0.05, 50)
        detector.set_baseline(baseline)
        current = np.random.normal(0.8, 0.05, 30)
        events = detector.update_batch(current)
        assert len(events) == 0

    def test_drift_detected_on_step_change(self):
        from lib.ml.mlops.drift.concept_drift import ConceptDriftDetector
        detector = ConceptDriftDetector(ph_lambda=5.0, cusum_h=2.0, ph_burn_in=5)
        baseline = np.full(20, 0.85)
        detector.set_baseline(baseline)
        shifted = np.full(30, 0.3)  # dramatic drop
        events = detector.update_batch(shifted)
        assert len(events) > 0

    def test_entropy_drift_detection(self):
        from lib.ml.mlops.drift.concept_drift import ConceptDriftDetector
        detector = ConceptDriftDetector()
        nc = 10
        # Confident baseline
        base_probs = np.zeros((50, nc))
        base_probs[:, 0] = 0.9
        base_probs[:, 1:] = 0.1 / (nc - 1)
        # Uncertain current (high entropy)
        curr_probs = np.ones((50, nc)) / nc
        drifted, delta = detector.detect_entropy_drift(base_probs, curr_probs, threshold=0.1)
        assert drifted
        assert delta > 0


# =============================================================================
# ModelValidator
# =============================================================================
class TestModelValidator:
    def _export_tiny_onnx(self, tmp_path: Path, nc: int = 109, size: int = 64) -> Path:
        """Export a minimal verification model to ONNX."""
        import torch.nn as nn
        class Tiny(nn.Module):
            def __init__(self):
                super().__init__()
                self.pool = nn.AdaptiveAvgPool2d(1)
                self.conv = nn.Conv2d(3, 8, 1)
                self.fc   = nn.Linear(8, nc)
            def forward(self, x):
                return self.fc(self.pool(self.conv(x)).view(x.size(0), -1))
        m = Tiny(); m.eval()
        dummy = torch.zeros(1, 3, size, size)
        out = tmp_path / "tiny.onnx"
        torch.onnx.export(m, dummy, str(out), opset_version=17)
        return out

    def test_validation_passes(self, tmp_path):
        from lib.ml.mlops.validation.model_validator import ModelValidator
        onnx = self._export_tiny_onnx(tmp_path, nc=109, size=64)
        validator = ModelValidator(max_latency_ms=5000.0)
        report = validator.validate(onnx, "test_model", "v1.0.0",
                                    input_size=64, num_classes=109, task="verification")
        # file_exists, onnx_graph, ort_loads, inference, regression should pass
        checks_by_name = {c.name: c for c in report.checks}
        assert checks_by_name["file_exists"].passed
        assert checks_by_name["ort_loads"].passed
        assert checks_by_name["inference_executes"].passed

    def test_missing_file_fails(self, tmp_path):
        from lib.ml.mlops.validation.model_validator import ModelValidator
        validator = ModelValidator()
        report = validator.validate(tmp_path / "nonexistent.onnx",
                                    "m", "v1", input_size=64)
        assert not report.overall_passed
        assert report.checks[0].name == "file_exists"
        assert not report.checks[0].passed

    def test_save_report(self, tmp_path):
        from lib.ml.mlops.validation.model_validator import ModelValidator
        onnx = self._export_tiny_onnx(tmp_path)
        validator = ModelValidator(max_latency_ms=5000.0)
        report = validator.validate(onnx, "m", "v1", input_size=64, task="verification")
        out = validator.save_report(report, tmp_path / "reports")
        assert out.exists()
        data = json.loads(out.read_text())
        assert "overall_passed" in data


# =============================================================================
# RetrainingScheduler
# =============================================================================
class TestRetrainingScheduler:
    def test_schedule_and_list(self, tmp_path):
        from lib.ml.mlops.continuous_learning.retraining_scheduler import RetrainingScheduler
        scheduler = RetrainingScheduler(queue_path=tmp_path / "queue.json")
        job_id = scheduler.schedule("yolov11s", "weekly")
        jobs = scheduler.pending_jobs()
        assert any(j.job_id == job_id for j in jobs)

    def test_cancel_job(self, tmp_path):
        from lib.ml.mlops.continuous_learning.retraining_scheduler import RetrainingScheduler
        scheduler = RetrainingScheduler(queue_path=tmp_path / "queue.json")
        job_id = scheduler.schedule("yolov11s", "manual")
        ok = scheduler.cancel(job_id)
        assert ok
        job = scheduler.get_job(job_id)
        assert job.status == "cancelled"

    def test_drift_triggered_above_threshold(self, tmp_path):
        from lib.ml.mlops.continuous_learning.retraining_scheduler import RetrainingScheduler
        scheduler = RetrainingScheduler(queue_path=tmp_path / "queue.json")
        job_id = scheduler.schedule_drift_triggered("yolov11s", drift_score=0.35)
        assert job_id is not None

    def test_drift_not_triggered_below_threshold(self, tmp_path):
        from lib.ml.mlops.continuous_learning.retraining_scheduler import RetrainingScheduler
        scheduler = RetrainingScheduler(queue_path=tmp_path / "queue.json")
        job_id = scheduler.schedule_drift_triggered("yolov11s", drift_score=0.05)
        assert job_id is None

    def test_run_pending_executes_due_jobs(self, tmp_path):
        from lib.ml.mlops.continuous_learning.retraining_scheduler import RetrainingScheduler
        ran = []
        def callback(job):
            ran.append(job.job_id)
        scheduler = RetrainingScheduler(
            queue_path=tmp_path / "queue.json",
            run_callback=callback,
        )
        job_id = scheduler.schedule("yolov11s", "manual")
        # Force scheduled_at to past
        for j in scheduler._queue:
            if j.job_id == job_id:
                j.scheduled_at = "2020-01-01T00:00:00Z"
        scheduler._save_queue()
        executed = scheduler.run_pending()
        assert job_id in executed
        assert job_id in ran


# =============================================================================
# ContinuousLearningManager
# =============================================================================
class TestContinuousLearningManager:
    def _write_images(self, d: Path, n: int = 5) -> None:
        d.mkdir(parents=True, exist_ok=True)
        import cv2
        for i in range(n):
            img = np.random.randint(30, 200, (128, 128, 3), dtype=np.uint8)
            cv2.imwrite(str(d / f"img_{i:04d}.jpg"), img)

    def test_collect_from_directory(self, tmp_path):
        from lib.ml.mlops.continuous_learning.continuous_learning import ContinuousLearningManager
        source = tmp_path / "source"
        self._write_images(source)
        clm = ContinuousLearningManager(
            staging_dir=tmp_path / "staging",
            approved_dir=tmp_path / "approved",
            dataset_root=tmp_path / "dataset",
            records_path=tmp_path / "cl.json",
        )
        sids = clm.collect_from_directory(source)
        assert len(sids) == 5

    def test_validate_staged_samples(self, tmp_path):
        from lib.ml.mlops.continuous_learning.continuous_learning import ContinuousLearningManager
        source = tmp_path / "source"
        self._write_images(source)
        clm = ContinuousLearningManager(
            staging_dir=tmp_path / "staging",
            approved_dir=tmp_path / "approved",
            dataset_root=tmp_path / "dataset",
            records_path=tmp_path / "cl.json",
            blur_threshold=0.0,  # accept all
        )
        clm.collect_from_directory(source)
        counts = clm.validate_staged_samples()
        assert counts["approved"] + counts["rejected"] == 5

    def test_run_cycle_insufficient_samples(self, tmp_path):
        from lib.ml.mlops.continuous_learning.continuous_learning import ContinuousLearningManager
        clm = ContinuousLearningManager(
            staging_dir=tmp_path / "staging",
            approved_dir=tmp_path / "approved",
            dataset_root=tmp_path / "dataset",
            records_path=tmp_path / "cl.json",
            min_new_samples=100,
        )
        cycle = clm.run_cycle(force=False)
        assert cycle.status == "completed"
        assert cycle.retraining_job_id == ""  # not triggered

    def test_run_cycle_forced(self, tmp_path):
        from lib.ml.mlops.continuous_learning.continuous_learning import ContinuousLearningManager
        from lib.ml.mlops.registry.dataset_registry import DatasetRegistry
        # Create minimal dataset directory
        ds_dir = tmp_path / "dataset"
        (ds_dir / "train" / "images").mkdir(parents=True)
        clm = ContinuousLearningManager(
            staging_dir=tmp_path / "staging",
            approved_dir=tmp_path / "approved",
            dataset_root=ds_dir,
            records_path=tmp_path / "cl.json",
            min_new_samples=100,
            dataset_registry=DatasetRegistry(root=tmp_path / "ds_registry"),
        )
        cycle = clm.run_cycle(force=True)
        assert cycle.status == "completed"
        assert cycle.retraining_job_id != ""
