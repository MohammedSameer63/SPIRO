"""Tests for loss functions."""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))


class TestLabelSmoothingCrossEntropy:
    def test_forward_shape(self):
        from lib.ml.verification.training.losses import LabelSmoothingCrossEntropy
        loss_fn = LabelSmoothingCrossEntropy(smoothing=0.1, num_classes=10)
        logits = torch.randn(4, 10)
        labels = torch.randint(0, 10, (4,))
        loss = loss_fn(logits, labels)
        assert loss.dim() == 0  # scalar
        assert loss.item() > 0

    def test_zero_smoothing_equals_ce(self):
        from lib.ml.verification.training.losses import LabelSmoothingCrossEntropy
        import torch.nn.functional as F
        torch.manual_seed(0)
        logits = torch.randn(8, 5)
        labels = torch.randint(0, 5, (8,))
        ls_loss = LabelSmoothingCrossEntropy(smoothing=0.0, num_classes=5)(logits, labels)
        ce_loss = F.cross_entropy(logits, labels)
        assert abs(ls_loss.item() - ce_loss.item()) < 1e-5

    def test_smoothing_lowers_confidence(self):
        from lib.ml.verification.training.losses import LabelSmoothingCrossEntropy
        logits = torch.zeros(4, 5)
        logits[:, 0] = 10.0  # very confident
        labels = torch.zeros(4, dtype=torch.long)
        loss_low = LabelSmoothingCrossEntropy(smoothing=0.0)(logits, labels)
        loss_high = LabelSmoothingCrossEntropy(smoothing=0.5)(logits, labels)
        assert loss_high.item() > loss_low.item()


class TestFocalLoss:
    def test_forward_scalar(self):
        from lib.ml.verification.training.losses import FocalLoss
        loss_fn = FocalLoss(gamma=2.0)
        logits = torch.randn(4, 10)
        labels = torch.randint(0, 10, (4,))
        loss = loss_fn(logits, labels)
        assert loss.dim() == 0
        assert loss.item() >= 0

    def test_gamma_zero_equals_ce(self):
        from lib.ml.verification.training.losses import FocalLoss
        import torch.nn.functional as F
        torch.manual_seed(1)
        logits = torch.randn(8, 5)
        labels = torch.randint(0, 5, (8,))
        fl_loss = FocalLoss(gamma=0.0)(logits, labels)
        ce_loss = F.cross_entropy(logits, labels)
        assert abs(fl_loss.item() - ce_loss.item()) < 1e-5

    def test_with_alpha_weights(self):
        from lib.ml.verification.training.losses import FocalLoss
        alpha = [1.0, 2.0, 0.5, 1.0, 1.5]
        loss_fn = FocalLoss(gamma=2.0, alpha=alpha)
        logits = torch.randn(4, 5)
        labels = torch.randint(0, 5, (4,))
        loss = loss_fn(logits, labels)
        assert loss.item() >= 0


class TestWeightedCrossEntropy:
    def test_forward(self):
        from lib.ml.verification.training.losses import WeightedCrossEntropyLoss
        weights = torch.ones(5)
        loss_fn = WeightedCrossEntropyLoss(weights)
        logits = torch.randn(4, 5)
        labels = torch.randint(0, 5, (4,))
        loss = loss_fn(logits, labels)
        assert loss.dim() == 0
        assert loss.item() >= 0


class TestBuildLoss:
    def test_cross_entropy(self, verify_config_path):
        from lib.ml.verification.verify_config import VerifyConfig
        from lib.ml.verification.training.losses import build_loss
        import torch.nn as nn
        cfg = VerifyConfig.load(verify_config_path)
        # Override loss name
        from omegaconf import OmegaConf
        OmegaConf.update(cfg._cfg, "loss.name", "cross_entropy")
        loss_fn = build_loss(cfg)
        assert isinstance(loss_fn, nn.CrossEntropyLoss)

    def test_label_smoothing(self, verify_config_path):
        from lib.ml.verification.verify_config import VerifyConfig
        from lib.ml.verification.training.losses import build_loss, LabelSmoothingCrossEntropy
        cfg = VerifyConfig.load(verify_config_path)
        loss_fn = build_loss(cfg)
        assert isinstance(loss_fn, LabelSmoothingCrossEntropy)

    def test_focal(self, verify_config_path):
        from lib.ml.verification.verify_config import VerifyConfig
        from lib.ml.verification.training.losses import build_loss, FocalLoss
        from omegaconf import OmegaConf
        cfg = VerifyConfig.load(verify_config_path)
        OmegaConf.update(cfg._cfg, "loss.name", "focal")
        loss_fn = build_loss(cfg)
        assert isinstance(loss_fn, FocalLoss)

    def test_invalid_loss_raises(self, verify_config_path):
        from lib.ml.verification.verify_config import VerifyConfig
        from lib.ml.verification.training.losses import build_loss
        from omegaconf import OmegaConf
        cfg = VerifyConfig.load(verify_config_path)
        OmegaConf.update(cfg._cfg, "loss.name", "invalid_xyz")
        with pytest.raises(ValueError, match="Unknown loss"):
            build_loss(cfg)
