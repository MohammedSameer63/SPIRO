"""Tests for ContinuousLearningManager and DriftDetector."""
import sys
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from lib.ml.continuous_learning.manager import (
    DriftDetector,
    NewSampleBuffer,
    ContinuousLearningManager,
)
from lib.ml.models.registry import ModelRegistry


@pytest.mark.unit
class TestDriftDetector:
    def test_no_drift_stable_stream(self):
        det = DriftDetector(delta=0.005, lambda_=50.0, window=100)
        drifted = False
        for _ in range(200):
            if det.update(0.85):
                drifted = True
        assert not drifted

    def test_drift_detected_on_drop(self):
        det = DriftDetector(delta=0.001, lambda_=5.0, window=50)
        # Feed high confidence first
        for _ in range(60):
            det.update(0.90)
        # Then sudden drop
        detected = False
        for _ in range(60):
            if det.update(0.20):
                detected = True
        assert detected

    def test_reset_clears_state(self):
        det = DriftDetector(delta=0.001, lambda_=5.0, window=50)
        for _ in range(100):
            det.update(0.2)
        det.reset()
        assert det.n_observations == 0
        assert len(det.window) == 0

    def test_observation_count_increments(self):
        det = DriftDetector()
        for i in range(10):
            det.update(0.8)
        assert det.n_observations == 10


@pytest.mark.unit
class TestNewSampleBuffer:
    def test_add_creates_files(self, tmp_path):
        buf = NewSampleBuffer(tmp_path / "buffer")
        img = np.zeros((64, 64, 3), dtype=np.uint8)
        dets = [{"bbox_xyxy": [5.0, 5.0, 30.0, 30.0], "class_id": 0, "confidence": 0.9}]
        buf.add(img, dets, stem="test_sample")
        assert (tmp_path / "buffer" / "images" / "test_sample.jpg").exists()
        assert (tmp_path / "buffer" / "labels" / "test_sample.txt").exists()

    def test_count_increments(self, tmp_path):
        buf = NewSampleBuffer(tmp_path / "buffer")
        img = np.zeros((64, 64, 3), dtype=np.uint8)
        for i in range(5):
            buf.add(img, [], stem=f"sample_{i}")
        assert buf.count == 5

    def test_label_format_valid(self, tmp_path):
        buf = NewSampleBuffer(tmp_path / "buffer")
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        dets = [{"bbox_xyxy": [10.0, 20.0, 50.0, 60.0], "class_id": 2, "confidence": 0.7}]
        buf.add(img, dets, stem="lbl_test")
        lbl = (tmp_path / "buffer" / "labels" / "lbl_test.txt").read_text().strip()
        parts = lbl.split()
        assert len(parts) == 5
        assert parts[0] == "2"
        for v in parts[1:]:
            assert 0.0 <= float(v) <= 1.0

    def test_clear(self, tmp_path):
        buf = NewSampleBuffer(tmp_path / "buffer")
        img = np.zeros((64, 64, 3), dtype=np.uint8)
        buf.add(img, [], stem="s1")
        buf.add(img, [], stem="s2")
        buf.clear()
        assert buf.count == 0
        assert len(list(buf.images_dir.iterdir())) == 0


@pytest.mark.unit
class TestContinuousLearningManager:
    def _make_clm(self, tmp_path, base_config):
        sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
        from lib.ml.core.config import ConfigManager

        cfg = ConfigManager.load(base_config)
        registry = ModelRegistry(tmp_path / "registry")
        clm = ContinuousLearningManager(cfg, registry)
        # Override buffer to use tmp_path
        from lib.ml.continuous_learning.manager import NewSampleBuffer
        clm.buffer = NewSampleBuffer(tmp_path / "cl_buffer")
        return clm, cfg, registry

    def test_observe_accumulates(self, tmp_path, base_config):
        clm, _, _ = self._make_clm(tmp_path, base_config)
        img = np.zeros((64, 64, 3), dtype=np.uint8)
        dets = []
        for _ in range(3):
            clm.observe(img, dets, confidence=0.8, buffer_sample=True)
        assert clm.buffer.count == 3

    def test_should_retrain_false_initially(self, tmp_path, base_config):
        clm, _, _ = self._make_clm(tmp_path, base_config)
        assert clm.should_retrain() is False

    def test_should_retrain_true_after_drift_and_samples(self, tmp_path, base_config):
        clm, _, _ = self._make_clm(tmp_path, base_config)
        clm._drift_detected = True
        # Fill buffer above min_new_samples (set to 5 in test config)
        img = np.zeros((64, 64, 3), dtype=np.uint8)
        for i in range(6):
            clm.buffer.add(img, [], stem=f"s{i}")
        assert clm.should_retrain() is True

    def test_summary_structure(self, tmp_path, base_config):
        clm, _, _ = self._make_clm(tmp_path, base_config)
        s = clm.summary()
        assert "observations" in s
        assert "buffered_samples" in s
        assert "drift_detected" in s
        assert "retrain_cycles" in s

    def test_no_callback_trigger_returns_none(self, tmp_path, base_config):
        clm, _, _ = self._make_clm(tmp_path, base_config)
        clm.retrain_callback = None
        result = clm.trigger_retrain()
        assert result is None
