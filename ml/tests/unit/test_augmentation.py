"""Tests for Augmentation pipelines."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from lib.ml.core.config import ConfigManager
from lib.ml.data.augmentation import build_train_transform, build_val_transform


@pytest.mark.unit
class TestAugmentation:
    def test_train_transform_returns_image(self, base_config):
        cfg = ConfigManager.load(base_config)
        transform = build_train_transform(cfg)
        img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        bboxes = [(0.5, 0.5, 0.2, 0.2)]
        class_labels = [0]
        result = transform(image=img, bboxes=bboxes, class_labels=class_labels)
        assert "image" in result
        assert result["image"].shape[2] == 3

    def test_train_transform_clips_bboxes(self, base_config):
        cfg = ConfigManager.load(base_config)
        transform = build_train_transform(cfg)
        img = np.zeros((256, 256, 3), dtype=np.uint8)
        bboxes = [(0.5, 0.5, 0.3, 0.3)]
        result = transform(image=img, bboxes=bboxes, class_labels=[1])
        for box in result["bboxes"]:
            for val in box:
                assert 0.0 <= val <= 1.0

    def test_val_transform_no_randomness(self, base_config):
        cfg = ConfigManager.load(base_config)
        transform = build_val_transform(cfg)
        img = np.random.randint(0, 255, (300, 400, 3), dtype=np.uint8)
        r1 = transform(image=img.copy(), bboxes=[], class_labels=[])
        r2 = transform(image=img.copy(), bboxes=[], class_labels=[])
        # Val transform should be deterministic
        np.testing.assert_array_equal(r1["image"], r2["image"])

    def test_empty_bboxes(self, base_config):
        cfg = ConfigManager.load(base_config)
        transform = build_train_transform(cfg)
        img = np.zeros((256, 256, 3), dtype=np.uint8)
        result = transform(image=img, bboxes=[], class_labels=[])
        assert result["bboxes"] == []
        assert result["class_labels"] == []
