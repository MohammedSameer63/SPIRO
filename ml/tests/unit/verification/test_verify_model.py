"""Tests for VerifierModel (EfficientNetV2)."""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))


class TestVerifierModel:
    def test_forward_shape(self, verify_config_path):
        from lib.ml.verification.verify_config import VerifyConfig
        from lib.ml.verification.model.verify_model import VerifierModel
        cfg = VerifyConfig.load(verify_config_path)
        model = VerifierModel(cfg)
        x = torch.randn(2, 3, cfg.input_size, cfg.input_size)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (2, cfg.model.num_classes)

    def test_predict_returns_result(self, verify_config_path, dummy_bgr_image):
        from lib.ml.verification.verify_config import VerifyConfig
        from lib.ml.verification.model.verify_model import VerifierModel, VerificationResult
        cfg = VerifyConfig.load(verify_config_path)
        model = VerifierModel(cfg)
        result = model.predict(dummy_bgr_image)
        assert isinstance(result, VerificationResult)
        assert 0 <= result.class_id < cfg.model.num_classes
        assert 0.0 <= result.confidence <= 1.0
        assert len(result.probabilities) == cfg.model.num_classes
        assert abs(sum(result.probabilities) - 1.0) < 1e-4

    def test_predict_top5(self, verify_config_path, dummy_bgr_image):
        from lib.ml.verification.verify_config import VerifyConfig
        from lib.ml.verification.model.verify_model import VerifierModel
        cfg = VerifyConfig.load(verify_config_path)
        model = VerifierModel(cfg)
        result = model.predict(dummy_bgr_image, top_k=5)
        assert len(result.top5_classes) == min(5, cfg.model.num_classes)
        assert len(result.top5_names) == min(5, cfg.model.num_classes)
        assert len(result.top5_confidences) == min(5, cfg.model.num_classes)
        # Best class should be first
        assert result.class_id == result.top5_classes[0]

    def test_predict_batch(self, verify_config_path, dummy_bgr_image):
        from lib.ml.verification.verify_config import VerifyConfig
        from lib.ml.verification.model.verify_model import VerifierModel
        cfg = VerifyConfig.load(verify_config_path)
        model = VerifierModel(cfg)
        results = model.predict_batch([dummy_bgr_image] * 3)
        assert len(results) == 3
        for r in results:
            assert 0 <= r.class_id < cfg.model.num_classes

    def test_freeze_backbone(self, verify_config_path):
        from lib.ml.verification.verify_config import VerifyConfig
        from lib.ml.verification.model.verify_model import VerifierModel
        cfg = VerifyConfig.load(verify_config_path)
        model = VerifierModel(cfg, freeze_backbone=True)
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        assert trainable < total

    def test_unfreeze_all(self, verify_config_path):
        from lib.ml.verification.verify_config import VerifyConfig
        from lib.ml.verification.model.verify_model import VerifierModel
        cfg = VerifyConfig.load(verify_config_path)
        model = VerifierModel(cfg, freeze_backbone=True)
        model.unfreeze_all()
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        assert trainable == total

    def test_save_and_load(self, verify_config_path, tmp_path):
        from lib.ml.verification.verify_config import VerifyConfig
        from lib.ml.verification.model.verify_model import VerifierModel
        cfg = VerifyConfig.load(verify_config_path)
        model = VerifierModel(cfg)
        path = tmp_path / "test_effnet.pt"
        model.save(path)
        assert path.exists()
        loaded = VerifierModel.load(cfg, path)
        assert loaded.num_classes == model.num_classes

    def test_parameter_count(self, verify_config_path):
        from lib.ml.verification.verify_config import VerifyConfig
        from lib.ml.verification.model.verify_model import VerifierModel
        cfg = VerifyConfig.load(verify_config_path)
        model = VerifierModel(cfg)
        counts = model.parameter_count()
        assert counts["total"] > 0
        assert counts["trainable"] > 0

    def test_result_to_dict(self, verify_config_path, dummy_bgr_image):
        from lib.ml.verification.verify_config import VerifyConfig
        from lib.ml.verification.model.verify_model import VerifierModel
        cfg = VerifyConfig.load(verify_config_path)
        model = VerifierModel(cfg)
        result = model.predict(dummy_bgr_image)
        d = result.to_dict()
        assert "class_id" in d
        assert "class_name" in d
        assert "confidence" in d
        assert "top5" in d

    def test_temperature_scaling(self, verify_config_path, dummy_bgr_image):
        from lib.ml.verification.verify_config import VerifyConfig
        from lib.ml.verification.model.verify_model import VerifierModel
        cfg = VerifyConfig.load(verify_config_path)
        model = VerifierModel(cfg)
        r1 = model.predict(dummy_bgr_image, temperature=1.0)
        r2 = model.predict(dummy_bgr_image, temperature=2.0)
        # Higher temperature → softer distribution → lower max confidence
        assert r2.confidence <= r1.confidence + 1e-4
