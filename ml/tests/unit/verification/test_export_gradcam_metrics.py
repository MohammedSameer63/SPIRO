"""Tests for ONNX export, GradCAM, and VerifyMetrics."""
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pytest
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))


# =============================================================================
# ONNX Export Tests
# =============================================================================

class TinyClassifier(nn.Module):
    """Minimal model for export tests."""
    def __init__(self, nc=5, size=64):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv2d(3, 8, 3, padding=1)
        self.fc = nn.Linear(8, nc)
        self.class_names = [f"cls_{i}" for i in range(nc)]
        self.num_classes = nc
        self.input_size = size
        self.device = torch.device("cpu")

    def forward(self, x):
        return self.fc(self.pool(self.conv(x)).view(x.size(0), -1))


class TestVerifyExporter:
    def test_export_creates_onnx(self, verify_config_path, tmp_path):
        from lib.ml.verification.verify_config import VerifyConfig
        from lib.ml.verification.export.verify_export import VerifyExporter
        from omegaconf import OmegaConf

        cfg = VerifyConfig.load(verify_config_path)
        OmegaConf.update(cfg._cfg, "export.output_dir", str(tmp_path))

        model = TinyClassifier(nc=5, size=64)
        model.eval()

        exporter = VerifyExporter(cfg)

        # Patch the model's attribute access
        with patch.object(exporter, "export") as mock_export:
            onnx_path = tmp_path / "test_model.onnx"
            # Manually export
            dummy = torch.zeros(1, 3, 64, 64)
            torch.onnx.export(
                model, dummy, str(onnx_path),
                input_names=["images"], output_names=["logits"],
                opset_version=17,
            )
            assert onnx_path.exists()

    def test_onnx_checker_passes(self, tmp_path):
        import onnx
        model = TinyClassifier()
        model.eval()
        dummy = torch.zeros(1, 3, 64, 64)
        path = tmp_path / "valid.onnx"
        torch.onnx.export(model, dummy, str(path), opset_version=17)
        onnx_model = onnx.load(str(path))
        onnx.checker.check_model(onnx_model)  # should not raise

    def test_ort_inference_matches_pytorch(self, tmp_path):
        import onnxruntime as ort
        torch.manual_seed(42)
        model = TinyClassifier()
        model.eval()
        dummy = torch.randn(1, 3, 64, 64)
        path = tmp_path / "parity.onnx"
        torch.onnx.export(model, dummy, str(path), opset_version=17)

        with torch.no_grad():
            pt_out = model(dummy).numpy()

        sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        ort_out = sess.run(None, {sess.get_inputs()[0].name: dummy.numpy()})[0]

        assert np.abs(pt_out - ort_out).max() < 1e-4

    def test_verify_report_written(self, verify_config_path, tmp_path):
        import onnxruntime as ort
        from lib.ml.verification.verify_config import VerifyConfig
        from lib.ml.verification.export.verify_export import VerifyExporter
        from omegaconf import OmegaConf
        import json

        cfg = VerifyConfig.load(verify_config_path)
        OmegaConf.update(cfg._cfg, "export.output_dir", str(tmp_path))

        model = TinyClassifier(nc=5, size=64)
        model.eval()
        dummy = torch.zeros(1, 3, 64, 64)
        onnx_path = tmp_path / "to_verify.onnx"
        torch.onnx.export(model, dummy, str(onnx_path), opset_version=17)

        exporter = VerifyExporter(cfg)
        report = exporter.verify(onnx_path, pt_model=model)

        assert isinstance(report, dict)
        assert "passed" in report
        assert "ort_output_shape" in report

        report_file = tmp_path / "to_verify_verification_report.json"
        assert report_file.exists()
        data = json.loads(report_file.read_text())
        assert "max_diff" in data


# =============================================================================
# GradCAM Tests
# =============================================================================

class TestGradCAM:
    def _make_model(self, verify_config_path):
        from lib.ml.verification.verify_config import VerifyConfig
        from lib.ml.verification.model.verify_model import VerifierModel
        cfg = VerifyConfig.load(verify_config_path)
        return VerifierModel(cfg), cfg

    def test_gradcam_output_shape(self, verify_config_path, dummy_bgr_image):
        from lib.ml.verification.explainability.gradcam import GradCAM
        model, cfg = self._make_model(verify_config_path)
        cam = GradCAM(model, method="gradcam")
        heatmap, overlay, explanation = cam.explain(dummy_bgr_image)
        assert heatmap.shape == dummy_bgr_image.shape[:2]
        assert overlay.shape == dummy_bgr_image.shape
        cam.remove_hooks()

    def test_explanation_contains_required_keys(self, verify_config_path, dummy_bgr_image):
        from lib.ml.verification.explainability.gradcam import GradCAM
        model, cfg = self._make_model(verify_config_path)
        cam = GradCAM(model, method="gradcam")
        _, _, explanation = cam.explain(dummy_bgr_image)
        for key in ["class_id", "class_name", "confidence", "top5_predictions",
                    "top_contributing_regions", "cam_method"]:
            assert key in explanation, f"Missing key: {key}"
        cam.remove_hooks()

    def test_top_regions_structure(self, verify_config_path, dummy_bgr_image):
        from lib.ml.verification.explainability.gradcam import GradCAM
        model, cfg = self._make_model(verify_config_path)
        cam = GradCAM(model, method="gradcam")
        _, _, explanation = cam.explain(dummy_bgr_image)
        for region in explanation["top_contributing_regions"]:
            assert "center_x" in region
            assert "center_y" in region
            assert "score" in region
            assert "bbox" in region
            assert len(region["bbox"]) == 4
        cam.remove_hooks()

    def test_save_explanation_creates_files(self, verify_config_path, dummy_bgr_image, tmp_path):
        from lib.ml.verification.explainability.gradcam import GradCAM
        model, cfg = self._make_model(verify_config_path)
        cam = GradCAM(model, method="gradcam")
        paths = cam.save_explanation(
            source=dummy_bgr_image,
            output_dir=tmp_path / "gradcam",
            stem="test_crop",
        )
        assert Path(paths["overlay"]).exists()
        assert Path(paths["heatmap"]).exists()
        assert Path(paths["json"]).exists()
        cam.remove_hooks()

    def test_find_last_conv(self, verify_config_path):
        from lib.ml.verification.explainability.gradcam import GradCAM
        model, _ = self._make_model(verify_config_path)
        conv = GradCAM._find_last_conv(model)
        assert isinstance(conv, torch.nn.Conv2d)


# =============================================================================
# VerifyMetrics Tests
# =============================================================================

class TestVerifyMetrics:
    def test_compute_basic_perfect(self):
        from lib.ml.verification.evaluation.verify_metrics import VerifyMetrics
        logits = torch.zeros(5, 10)
        for i in range(5):
            logits[i, i % 10] = 10.0  # high confidence correct
        labels = torch.arange(5) % 10
        result = VerifyMetrics.compute_basic(logits, labels, top_k=5)
        assert result["top1_accuracy"] == 1.0

    def test_compute_basic_shape(self):
        from lib.ml.verification.evaluation.verify_metrics import VerifyMetrics
        logits = torch.randn(16, 109)
        labels = torch.randint(0, 109, (16,))
        result = VerifyMetrics.compute_basic(logits, labels, top_k=5)
        assert "top1_accuracy" in result
        assert "top5_accuracy" in result
        assert 0.0 <= result["top1_accuracy"] <= 1.0
        assert 0.0 <= result["top5_accuracy"] <= 1.0
        # Top-5 ≥ Top-1
        assert result["top5_accuracy"] >= result["top1_accuracy"] - 1e-6

    def test_evaluate_returns_dict(self, tmp_path):
        from lib.ml.verification.evaluation.verify_metrics import VerifyMetrics
        metrics = VerifyMetrics(num_classes=5,
                                class_names=[f"c{i}" for i in range(5)],
                                output_dir=tmp_path)
        logits = torch.randn(20, 5)
        labels = torch.randint(0, 5, (20,))
        result = metrics.evaluate(logits, labels, split="test", save_reports=True)
        assert "top1_accuracy" in result
        assert "top5_accuracy" in result
        assert "f1_macro" in result
        assert "per_class_accuracy" in result

    def test_confusion_matrix_created(self, tmp_path):
        from lib.ml.verification.evaluation.verify_metrics import VerifyMetrics
        metrics = VerifyMetrics(num_classes=5,
                                class_names=[f"c{i}" for i in range(5)],
                                output_dir=tmp_path)
        logits = torch.randn(20, 5)
        labels = torch.randint(0, 5, (20,))
        metrics.evaluate(logits, labels, split="val", save_reports=True)
        assert (tmp_path / "val_confusion_matrix.png").exists()

    def test_classification_report_json(self, tmp_path):
        import json
        from lib.ml.verification.evaluation.verify_metrics import VerifyMetrics
        metrics = VerifyMetrics(num_classes=5,
                                class_names=[f"c{i}" for i in range(5)],
                                output_dir=tmp_path)
        logits = torch.randn(20, 5)
        labels = torch.randint(0, 5, (20,))
        metrics.evaluate(logits, labels, split="test", save_reports=True)
        report_path = tmp_path / "test_classification_report.json"
        assert report_path.exists()
        data = json.loads(report_path.read_text())
        assert "top1_accuracy" in data
        assert "classification_report" in data

    def test_plot_training_curves(self, tmp_path):
        from lib.ml.verification.evaluation.verify_metrics import VerifyMetrics
        metrics = VerifyMetrics(num_classes=5,
                                class_names=[f"c{i}" for i in range(5)],
                                output_dir=tmp_path)
        history = {
            "train_loss": [1.5, 1.2, 0.9],
            "train_acc": [0.3, 0.5, 0.7],
            "val_loss": [1.6, 1.3, 1.0],
            "val_acc": [0.28, 0.45, 0.65],
            "val_top5_acc": [0.5, 0.7, 0.9],
        }
        outputs = metrics.plot_training_curves(history)
        assert len(outputs) >= 2
        for p in outputs:
            assert Path(p).exists()
