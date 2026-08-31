"""Tests for ONNXInferenceEngine (CPU, no real model required for unit tests)."""
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from lib.ml.inference.onnx_engine import _nms, _softmax, ONNXInferenceEngine


@pytest.mark.unit
class TestNMSAndSoftmax:
    def test_softmax_sums_to_one(self):
        x = np.array([1.0, 2.0, 3.0])
        result = _softmax(x)
        assert abs(result.sum() - 1.0) < 1e-6

    def test_softmax_max_wins(self):
        x = np.array([0.0, 10.0, 0.0])
        result = _softmax(x)
        assert result.argmax() == 1

    def test_nms_removes_overlapping(self):
        boxes = np.array([
            [0, 0, 10, 10],
            [1, 1, 11, 11],   # high overlap with box 0
            [50, 50, 60, 60], # no overlap
        ], dtype=float)
        scores = np.array([0.9, 0.8, 0.7])
        keep = _nms(boxes, scores, iou_threshold=0.5)
        assert 0 in keep
        assert 2 in keep
        assert 1 not in keep

    def test_nms_no_overlap_keeps_all(self):
        boxes = np.array([
            [0, 0, 5, 5],
            [10, 10, 15, 15],
            [20, 20, 25, 25],
        ], dtype=float)
        scores = np.array([0.9, 0.8, 0.7])
        keep = _nms(boxes, scores, iou_threshold=0.5)
        assert set(keep) == {0, 1, 2}

    def test_nms_empty(self):
        boxes = np.empty((0, 4), dtype=float)
        scores = np.array([])
        keep = _nms(boxes, scores, iou_threshold=0.5)
        assert keep == []


@pytest.mark.unit
class TestONNXEnginePreprocess:
    """Test preprocessing logic without loading a real ONNX model."""

    def _make_engine(self):
        """Build a partially-initialised engine without loading ONNX."""
        with patch("lib.ml.inference.onnx_engine.ort.InferenceSession"):
            engine = object.__new__(ONNXInferenceEngine)
            engine.onnx_path = Path("dummy.onnx")
            engine.input_size = (640, 640)
            engine.conf_threshold = 0.4
            engine.iou_threshold = 0.45
            engine.class_names = ["cat", "dog", "bird"]
            return engine

    def test_preprocess_output_shape(self):
        engine = self._make_engine()
        img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        blob, scale, padding = engine._preprocess(img)
        assert blob.shape == (1, 3, 640, 640)
        assert blob.dtype == np.float32
        assert blob.min() >= 0.0
        assert blob.max() <= 1.0

    def test_preprocess_normalises(self):
        engine = self._make_engine()
        img = np.full((100, 100, 3), 255, dtype=np.uint8)
        blob, _, _ = engine._preprocess(img)
        assert abs(blob.max() - 1.0) < 1e-5

    def test_preprocess_scale_computed(self):
        engine = self._make_engine()
        img = np.zeros((320, 640, 3), dtype=np.uint8)
        _, scale, _ = engine._preprocess(img)
        # Width already 640, height 320 → scale = min(640/640, 640/320) = 1.0
        assert abs(scale - 1.0) < 1e-5

    def test_postprocess_classification(self):
        engine = self._make_engine()
        # Simulate classifier output [1, 3]
        raw = np.array([[0.1, 0.8, 0.1]], dtype=np.float32)
        result = engine._postprocess([raw], (480, 640), 1.0, (0, 0))
        assert result["type"] == "classification"
        assert result["class_id"] == 1
        assert result["class_name"] == "dog"
        assert abs(result["confidence"] - max(_softmax(raw[0]))) < 1e-5

    def test_load_numpy_passthrough(self):
        engine = self._make_engine()
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        loaded = engine._load(img)
        np.testing.assert_array_equal(loaded, img)

    def test_load_missing_file_raises(self):
        engine = self._make_engine()
        with pytest.raises(FileNotFoundError):
            engine._load("/nonexistent/path/image.jpg")
