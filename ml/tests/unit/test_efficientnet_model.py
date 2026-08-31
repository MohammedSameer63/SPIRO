"""Tests for SPIROClassifier (CPU, pretrained=False for speed)."""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))


@pytest.mark.unit
class TestSPIROClassifier:
    def test_forward_pass_shape(self, base_config):
        from lib.ml.core.config import ConfigManager
        from lib.ml.models.efficientnet_model import SPIROClassifier

        cfg = ConfigManager.load(base_config)
        model = SPIROClassifier(cfg)
        x = torch.randn(2, 3, 128, 128)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (2, cfg.efficientnetv2.num_classes)

    def test_predict_returns_result(self, base_config):
        from lib.ml.core.config import ConfigManager
        from lib.ml.models.efficientnet_model import SPIROClassifier, ClassificationResult

        cfg = ConfigManager.load(base_config)
        model = SPIROClassifier(cfg)
        img = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
        result = model.predict(img)
        assert isinstance(result, ClassificationResult)
        assert 0 <= result.class_id < cfg.efficientnetv2.num_classes
        assert 0.0 <= result.confidence <= 1.0
        assert len(result.probabilities) == cfg.efficientnetv2.num_classes
        assert abs(sum(result.probabilities) - 1.0) < 1e-5

    def test_predict_batch(self, base_config):
        from lib.ml.core.config import ConfigManager
        from lib.ml.models.efficientnet_model import SPIROClassifier

        cfg = ConfigManager.load(base_config)
        model = SPIROClassifier(cfg)
        imgs = [np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8) for _ in range(3)]
        results = model.predict_batch(imgs)
        assert len(results) == 3

    def test_freeze_backbone(self, base_config):
        from lib.ml.core.config import ConfigManager
        from lib.ml.models.efficientnet_model import SPIROClassifier

        cfg = ConfigManager.load(base_config)
        model = SPIROClassifier(cfg, freeze_backbone=True)
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        assert trainable < total

    def test_unfreeze_all(self, base_config):
        from lib.ml.core.config import ConfigManager
        from lib.ml.models.efficientnet_model import SPIROClassifier

        cfg = ConfigManager.load(base_config)
        model = SPIROClassifier(cfg, freeze_backbone=True)
        model.unfreeze_all()
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        assert trainable == total

    def test_save_and_load(self, base_config, tmp_path):
        from lib.ml.core.config import ConfigManager
        from lib.ml.models.efficientnet_model import SPIROClassifier

        cfg = ConfigManager.load(base_config)
        model = SPIROClassifier(cfg)
        path = tmp_path / "clf.pt"
        model.save(path)
        assert path.exists()

        loaded = SPIROClassifier.load(cfg, path)
        assert loaded.class_names == model.class_names
        assert loaded.num_classes == model.num_classes

    def test_parameter_count(self, base_config):
        from lib.ml.core.config import ConfigManager
        from lib.ml.models.efficientnet_model import SPIROClassifier

        cfg = ConfigManager.load(base_config)
        model = SPIROClassifier(cfg)
        counts = model.parameter_count()
        assert counts["total"] > 0
        assert counts["trainable"] > 0
