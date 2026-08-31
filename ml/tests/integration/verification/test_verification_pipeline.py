"""
Integration tests for the EfficientNetV2 verification pipeline.
No internet — uses untrained (pretrained=False) models and synthetic data.
"""
import json
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import cv2
import numpy as np
import pytest
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

import lib.ml.dataset_engineering.taxonomy as _tax_module


@pytest.fixture(autouse=True)
def reset_taxonomy():
    _tax_module.TaxonomyLoader._instance = None
    yield
    _tax_module.TaxonomyLoader._instance = None


def _make_image(size=64):
    return np.random.randint(30, 200, (size, size, 3), dtype=np.uint8)


def _make_yolo_dataset(root: Path, n_per_split=5, nc=5):
    """Create tiny synthetic YOLO-format dataset."""
    for split in ["train", "val", "test"]:
        (root / split / "images").mkdir(parents=True)
        (root / split / "labels").mkdir(parents=True)
        for i in range(n_per_split):
            img = np.random.randint(30, 200, (64, 64, 3), dtype=np.uint8)
            cv2.imwrite(str(root / split / "images" / f"img_{i:04d}.jpg"), img)
            cls = i % nc
            with open(root / split / "labels" / f"img_{i:04d}.txt", "w") as f:
                f.write(f"{cls} 0.5 0.5 0.3 0.3\n")


@pytest.mark.integration
class TestVerificationPipelineIntegration:
    """Full pipeline integration tests with synthetic data."""

    def test_model_predict_and_fuse(self, verify_config_path, tmp_path):
        """VerifierModel predict → ConfidenceFusion → FusionResult."""
        from lib.ml.verification.verify_config import VerifyConfig
        from lib.ml.verification.model.verify_model import VerifierModel
        from lib.ml.verification.fusion.confidence_fusion import ConfidenceFusion

        cfg = VerifyConfig.load(verify_config_path)
        model = VerifierModel(cfg)
        img = _make_image(cfg.input_size)

        result = model.predict(img)
        assert result.confidence > 0

        # Simulate a YOLO detection with matching class
        nc = cfg.model.num_classes
        yolo_probs = np.zeros(nc)
        yolo_probs[result.class_id] = 0.75
        yolo_det = {
            "class_id": result.class_id,
            "confidence": 0.75,
            "class_name": result.class_name,
            "probabilities": yolo_probs.tolist(),
        }
        effnet_dict = {
            "class_id": result.class_id,
            "confidence": result.confidence,
            "probabilities": result.probabilities,
        }

        for method in ["weighted_average", "geometric_mean", "harmonic_mean", "bayesian", "temperature"]:
            fusion = ConfidenceFusion(method=method, num_classes=nc, min_confidence=0.0)
            fr = fusion.fuse(yolo_det, effnet_dict)
            assert 0.0 <= fr.fused_confidence <= 1.0

    def test_save_load_and_predict(self, verify_config_path, tmp_path):
        """Save checkpoint, reload, predict — full round-trip."""
        from lib.ml.verification.verify_config import VerifyConfig
        from lib.ml.verification.model.verify_model import VerifierModel

        cfg = VerifyConfig.load(verify_config_path)
        model = VerifierModel(cfg)

        path = tmp_path / "round_trip.pt"
        model.save(path)

        loaded = VerifierModel.load(cfg, path)
        img = _make_image(cfg.input_size)
        r = loaded.predict(img)
        assert r.class_id >= 0

    def test_onnx_export_and_inference(self, verify_config_path, tmp_path):
        """Export to ONNX → VerifyInferenceEngine → result dict."""
        from lib.ml.verification.verify_config import VerifyConfig
        from lib.ml.verification.model.verify_model import VerifierModel
        from lib.ml.verification.inference.verify_inference import VerifyInferenceEngine
        from omegaconf import OmegaConf

        cfg = VerifyConfig.load(verify_config_path)
        OmegaConf.update(cfg._cfg, "export.output_dir", str(tmp_path))

        model = VerifierModel(cfg)
        model.eval()

        # Export directly with torch.onnx
        dummy = torch.zeros(1, 3, cfg.input_size, cfg.input_size)
        onnx_path = tmp_path / "test_verify.onnx"
        torch.onnx.export(
            model, dummy, str(onnx_path),
            input_names=["images"], output_names=["logits"],
            opset_version=17,
        )

        # Run inference
        engine = VerifyInferenceEngine(
            onnx_path=onnx_path,
            input_size=cfg.input_size,
            warmup_runs=0,
        )
        img = _make_image(cfg.input_size)
        result = engine.verify(img)
        assert "class_id" in result
        assert "confidence" in result
        assert "top_k" in result
        assert 0.0 <= result["confidence"] <= 1.0

    def test_batch_inference(self, verify_config_path, tmp_path):
        """Batch inference on multiple crops."""
        from lib.ml.verification.verify_config import VerifyConfig
        from lib.ml.verification.model.verify_model import VerifierModel
        from lib.ml.verification.inference.verify_inference import VerifyInferenceEngine

        cfg = VerifyConfig.load(verify_config_path)
        model = VerifierModel(cfg)
        model.eval()

        dummy = torch.zeros(1, 3, cfg.input_size, cfg.input_size)
        onnx_path = tmp_path / "batch_test.onnx"
        torch.onnx.export(model, dummy, str(onnx_path), opset_version=17)

        engine = VerifyInferenceEngine(onnx_path=onnx_path,
                                       input_size=cfg.input_size, warmup_runs=0)
        imgs = [_make_image(cfg.input_size) for _ in range(4)]
        results = engine.verify_batch(imgs)
        assert len(results) == 4
        for r in results:
            assert "class_name" in r

    def test_verify_detections_from_yolo(self, verify_config_path, tmp_path):
        """verify_detections enriches YOLO detection dicts."""
        from lib.ml.verification.verify_config import VerifyConfig
        from lib.ml.verification.model.verify_model import VerifierModel
        from lib.ml.verification.inference.verify_inference import VerifyInferenceEngine

        cfg = VerifyConfig.load(verify_config_path)
        model = VerifierModel(cfg)
        model.eval()

        dummy = torch.zeros(1, 3, cfg.input_size, cfg.input_size)
        onnx_path = tmp_path / "det_test.onnx"
        torch.onnx.export(model, dummy, str(onnx_path), opset_version=17)

        engine = VerifyInferenceEngine(onnx_path=onnx_path,
                                       input_size=cfg.input_size, warmup_runs=0)

        img = _make_image(200)
        detections = [
            {"bbox_xyxy": [10.0, 10.0, 100.0, 100.0], "class_id": 0,
             "class_name": "plastic_bottle", "confidence": 0.85},
        ]
        enriched = engine.verify_detections(img, detections)
        assert len(enriched) == 1
        assert "verification" in enriched[0]
        if enriched[0]["verification"] is not None:
            assert "class_name" in enriched[0]["verification"]

    def test_gradcam_pipeline(self, verify_config_path, tmp_path):
        """GradCAM generates overlay + JSON explanation."""
        from lib.ml.verification.verify_config import VerifyConfig
        from lib.ml.verification.model.verify_model import VerifierModel
        from lib.ml.verification.explainability.gradcam import GradCAM

        cfg = VerifyConfig.load(verify_config_path)
        model = VerifierModel(cfg)
        img = _make_image(cfg.input_size)

        cam = GradCAM(model, method="gradcam")
        paths = cam.save_explanation(
            source=img,
            output_dir=tmp_path / "gradcam_out",
            stem="integration_test",
        )
        cam.remove_hooks()

        assert Path(paths["overlay"]).exists()
        assert Path(paths["heatmap"]).exists()
        assert Path(paths["json"]).exists()

        data = json.loads(Path(paths["json"]).read_text())
        assert "class_name" in data
        assert "confidence" in data
        assert "top_contributing_regions" in data

    def test_metrics_full_evaluation(self, tmp_path):
        """VerifyMetrics.evaluate writes all reports."""
        from lib.ml.verification.evaluation.verify_metrics import VerifyMetrics

        nc = 5
        names = [f"class_{i}" for i in range(nc)]
        metrics = VerifyMetrics(num_classes=nc, class_names=names, output_dir=tmp_path)

        n = 50
        logits = torch.randn(n, nc)
        labels = torch.randint(0, nc, (n,))

        result = metrics.evaluate(logits, labels, split="test", save_reports=True)

        assert 0.0 <= result["top1_accuracy"] <= 1.0
        assert 0.0 <= result["top5_accuracy"] <= 1.0
        assert result["top5_accuracy"] >= result["top1_accuracy"] - 1e-6
        assert (tmp_path / "test_confusion_matrix.png").exists()
        assert (tmp_path / "test_classification_report.json").exists()

    def test_crop_extractor(self, tmp_path):
        """CropExtractor produces crops in class subdirectories."""
        from lib.ml.verification.training.verify_dataset import CropExtractor

        imgs_dir = tmp_path / "images"
        lbls_dir = tmp_path / "labels"
        imgs_dir.mkdir()
        lbls_dir.mkdir()

        for i in range(5):
            img = np.random.randint(30, 200, (128, 128, 3), dtype=np.uint8)
            cv2.imwrite(str(imgs_dir / f"img_{i}.jpg"), img)
            with open(lbls_dir / f"img_{i}.txt", "w") as f:
                f.write(f"0 0.5 0.5 0.4 0.4\n")  # plastic_bottle

        extractor = CropExtractor(
            output_dir=tmp_path / "crops",
            min_crop_size=8,
        )
        counts = extractor.extract_from_split(imgs_dir, lbls_dir, split="train")
        total = sum(counts.values())
        assert total == 5
        assert "plastic_bottle" in counts
