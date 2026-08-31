"""Tests for DatasetManager."""
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from lib.ml.core.config import ConfigManager
from lib.ml.data.dataset_manager import DatasetManager


@pytest.mark.unit
class TestDatasetManager:
    def test_validate_raw_passes(self, base_config, tmp_dataset):
        cfg = ConfigManager.load(base_config)
        dm = DatasetManager(cfg)
        assert dm.validate_raw() is True

    def test_validate_bad_class_id(self, base_config, tmp_dataset, tmp_path):
        cfg = ConfigManager.load(base_config)
        dm = DatasetManager(cfg)

        # Inject a bad label file
        bad_lbl = Path(cfg.paths.raw_data) / "labels" / "bad_img.txt"
        bad_img = Path(cfg.paths.raw_data) / "images" / "bad_img.jpg"
        img = np.zeros((128, 128, 3), dtype=np.uint8)
        cv2.imwrite(str(bad_img), img)
        bad_lbl.write_text("99 0.5 0.5 0.1 0.1\n")  # cls_id=99 is out of range

        with pytest.raises(ValueError, match="validation failed"):
            dm.validate_raw()

        # Cleanup
        bad_lbl.unlink()
        bad_img.unlink()

    def test_preprocess_creates_images(self, base_config, tmp_dataset):
        cfg = ConfigManager.load(base_config)
        dm = DatasetManager(cfg)
        dm.validate_raw()
        dm.preprocess()
        proc_images = Path(cfg.paths.processed_data) / "images"
        assert proc_images.exists()
        assert len(list(proc_images.iterdir())) > 0

    def test_create_splits(self, base_config, tmp_dataset):
        cfg = ConfigManager.load(base_config)
        dm = DatasetManager(cfg)
        dm.validate_raw()
        dm.preprocess()
        splits = dm.create_splits()
        assert "train" in splits
        assert "val" in splits
        assert "test" in splits
        total = sum(len(v) for v in splits.values())
        assert total == 20  # 20 raw images

    def test_split_no_overlap(self, base_config, tmp_dataset):
        cfg = ConfigManager.load(base_config)
        dm = DatasetManager(cfg)
        dm.validate_raw()
        dm.preprocess()
        splits = dm.create_splits()
        train_set = set(splits["train"])
        val_set = set(splits["val"])
        test_set = set(splits["test"])
        assert len(train_set & val_set) == 0
        assert len(train_set & test_set) == 0
        assert len(val_set & test_set) == 0

    def test_analyze(self, base_config, tmp_dataset):
        cfg = ConfigManager.load(base_config)
        dm = DatasetManager(cfg)
        dm.validate_raw()
        dm.preprocess()
        dm.create_splits()
        stats = dm.analyze()
        assert "train" in stats
        for split in ["train", "val", "test"]:
            if split in stats:
                assert "num_images" in stats[split]
                assert "class_distribution" in stats[split]

    def test_letterbox_shape(self):
        img = np.zeros((100, 200, 3), dtype=np.uint8)
        result = DatasetManager._letterbox(img, (128, 128))
        assert result.shape == (128, 128, 3)

    def test_export_stats_json(self, base_config, tmp_dataset, tmp_path):
        cfg = ConfigManager.load(base_config)
        dm = DatasetManager(cfg)
        dm.validate_raw()
        dm.preprocess()
        dm.create_splits()
        dm.analyze()
        out = tmp_path / "stats.json"
        dm.export_stats(out)
        assert out.exists()
        import json
        data = json.loads(out.read_text())
        assert isinstance(data, dict)
