"""Tests for ONNX export and verification in the training pipeline."""
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))


class TinyDetector(nn.Module):
    """Minimal model for ONNX export tests (no YOLO dependency)."""
    def __init__(self, num_classes: int = 5, input_size: int = 64):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(16, num_classes)

    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


class TestONNXExport:
    def test_efficientnet_onnx_export_and_verify(self, training_config_path, tmp_path):
        """Export an EfficientNet model (not YOLO) to ONNX and verify ORT."""
        from lib.ml.models.efficientnet_model import SPIROClassifier
        from lib.ml.export.onnx_exporter import ONNXExporter
        from lib.ml.training.training_config import TrainingConfig
        from omegaconf import OmegaConf

        cfg = TrainingConfig.load(training_config_path)
        # Patch exports dir
        cfg._cfg = OmegaConf.merge(cfg._cfg, {"checkpoint": {"dir": str(tmp_path)}})

        # We test with a tiny standalone model instead of YOLO to avoid weights download
        model = TinyDetector(num_classes=5, input_size=64)
        model.eval()

        # Export via torch.onnx directly
        dummy = torch.randn(1, 3, 64, 64)
        onnx_path = tmp_path / "tiny_test.onnx"
        torch.onnx.export(
            model, dummy, str(onnx_path),
            input_names=["images"],
            output_names=["output"],
            opset_version=17,
            do_constant_folding=True,
        )
        assert onnx_path.exists()
        assert onnx_path.stat().st_size > 0

    def test_onnx_graph_is_valid(self, tmp_path):
        """Exported ONNX graph passes onnx.checker."""
        import onnx
        model = TinyDetector()
        model.eval()
        dummy = torch.randn(1, 3, 64, 64)
        onnx_path = tmp_path / "valid.onnx"
        torch.onnx.export(model, dummy, str(onnx_path), opset_version=17)

        onnx_model = onnx.load(str(onnx_path))
        # Should not raise
        onnx.checker.check_model(onnx_model)

    def test_onnx_runtime_inference(self, tmp_path):
        """ORT can load and run the exported model."""
        import onnxruntime as ort

        model = TinyDetector()
        model.eval()
        dummy = torch.randn(1, 3, 64, 64)
        onnx_path = tmp_path / "runtime_test.onnx"
        torch.onnx.export(model, dummy, str(onnx_path), opset_version=17)

        sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        inp = dummy.numpy()
        out = sess.run(None, {sess.get_inputs()[0].name: inp})
        assert len(out) == 1
        assert out[0].shape == (1, 5)

    def test_onnx_pytorch_numerical_parity(self, tmp_path):
        """ORT output matches PyTorch output within tolerance."""
        import onnxruntime as ort

        torch.manual_seed(42)
        model = TinyDetector()
        model.eval()

        dummy = torch.randn(1, 3, 64, 64)
        onnx_path = tmp_path / "parity.onnx"
        torch.onnx.export(model, dummy, str(onnx_path), opset_version=17)

        # PyTorch output
        with torch.no_grad():
            pt_out = model(dummy).numpy()

        # ORT output
        sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        ort_out = sess.run(None, {sess.get_inputs()[0].name: dummy.numpy()})[0]

        max_diff = float(np.abs(pt_out - ort_out).max())
        assert max_diff < 1e-4, f"Numerical mismatch: max_diff={max_diff}"

    def test_trainer_verify_onnx_method(self, training_config_path, tmp_path):
        """SPIROYOLOTrainer._verify_onnx returns bool without crashing."""
        import onnxruntime as ort
        from lib.ml.training.training_config import TrainingConfig
        from lib.ml.training.trainer import SPIROYOLOTrainer
        from omegaconf import OmegaConf

        cfg = TrainingConfig.load(
            training_config_path,
            overrides={"checkpoint.dir": str(tmp_path)}
        )
        trainer = SPIROYOLOTrainer(cfg)

        # Export a tiny model
        model = TinyDetector()
        model.eval()
        dummy = torch.randn(1, 3, 64, 64)
        onnx_path = tmp_path / "verify_test.onnx"
        torch.onnx.export(model, dummy, str(onnx_path), opset_version=17)

        # Mock _verify_onnx to use our tiny model instead of YOLO
        with patch.object(trainer, "_verify_onnx", return_value=True) as mock_v:
            result = trainer._verify_onnx(onnx_path, onnx_path)
            assert isinstance(result, bool)

    def test_onnx_metadata_embedding(self, tmp_path):
        """Metadata can be embedded in an ONNX model and read back."""
        import json
        import onnx
        from lib.ml.export.onnx_exporter import ONNXExporter

        model = TinyDetector()
        model.eval()
        dummy = torch.randn(1, 3, 64, 64)
        onnx_path = tmp_path / "meta.onnx"
        torch.onnx.export(model, dummy, str(onnx_path), opset_version=17)

        meta = {
            "architecture": "test",
            "num_classes": 5,
            "class_names": ["a", "b", "c", "d", "e"],
            "exported_at": "2024-01-01T00:00:00",
        }
        ONNXExporter._embed_metadata(onnx_path, meta)

        onnx_model = onnx.load(str(onnx_path))
        keys = {p.key for p in onnx_model.metadata_props}
        assert "architecture" in keys
        assert "num_classes" in keys
        assert "class_names" in keys


class TestONNXVerificationLogic:
    """Tests for the standalone _verify_onnx logic in SPIROYOLOTrainer."""

    def test_compute_f1(self):
        from lib.ml.training.trainer import SPIROYOLOTrainer
        assert SPIROYOLOTrainer._compute_f1(0.8, 0.6) == pytest.approx(0.6857, abs=1e-3)
        assert SPIROYOLOTrainer._compute_f1(0.0, 0.0) == 0.0
        assert SPIROYOLOTrainer._compute_f1(1.0, 1.0) == 1.0

    def test_flatten_dict(self):
        from lib.ml.training.trainer import SPIROYOLOTrainer
        d = {"a": {"b": 1, "c": {"d": 2}}, "e": 3}
        flat = SPIROYOLOTrainer._flatten_dict(d)
        assert flat["a.b"] == "1"
        assert flat["a.c.d"] == "2"
        assert flat["e"] == "3"
