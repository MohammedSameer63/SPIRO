"""
Integration test: raw dataset → validate → preprocess → split → augment → analyze.
Runs end-to-end on the session-scoped tmp_dataset fixture.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))


@pytest.mark.integration
class TestDataPipeline:
    def test_full_pipeline(self, base_config, tmp_dataset):
        from lib.ml.core.config import ConfigManager
        from lib.ml.data.dataset_manager import DatasetManager
        from lib.ml.data.augmentation import OfflineAugmentor

        cfg = ConfigManager.load(base_config)
        dm = DatasetManager(cfg)

        # Step 1: Validate
        ok = dm.validate_raw()
        assert ok is True

        # Step 2: Preprocess
        dm.preprocess()
        proc_img = Path(cfg.paths.processed_data) / "images"
        assert len(list(proc_img.iterdir())) == 20

        # Step 3: Split
        splits = dm.create_splits()
        total = sum(len(v) for v in splits.values())
        assert total == 20

        # All split dirs should exist
        for split in ["train", "val", "test"]:
            split_dir = Path(cfg.paths.processed_data) / split / "images"
            assert split_dir.exists(), f"{split} images dir missing"

        # Step 4: Augment training split (factor=1 for speed)
        aug = OfflineAugmentor(cfg)
        train_dir = Path(cfg.paths.processed_data) / "train"
        orig_count = len(list((train_dir / "images").iterdir()))
        written = aug.augment_split(train_dir, factor=1)
        new_count = len(list((train_dir / "images").iterdir()))
        assert new_count == orig_count + written

        # Step 5: Analyze
        stats = dm.analyze()
        assert "train" in stats
        assert stats["train"]["num_images"] > 0

    def test_stats_export_and_yaml_update(self, base_config, tmp_dataset, tmp_path):
        from lib.ml.core.config import ConfigManager
        from lib.ml.data.dataset_manager import DatasetManager
        import json, yaml

        cfg = ConfigManager.load(base_config)
        dm = DatasetManager(cfg)
        dm.validate_raw()
        dm.preprocess()
        dm.create_splits()
        dm.analyze()

        out = tmp_path / "stats.json"
        dm.export_stats(out)
        data = json.loads(out.read_text())
        assert isinstance(data, dict)

        # Update dataset.yaml (write to tmp)
        import shutil
        yaml_src = Path("configs/dataset.yaml")
        if yaml_src.exists():
            yaml_dst = tmp_path / "dataset.yaml"
            shutil.copy2(yaml_src, yaml_dst)
            dm.update_dataset_yaml(yaml_dst)
            with open(yaml_dst) as f:
                cfg_yaml = yaml.safe_load(f)
            assert cfg_yaml["stats"]["total_images"] > 0
