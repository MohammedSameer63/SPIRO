"""Tests for ConfigManager."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from lib.ml.core.config import ConfigManager


@pytest.mark.unit
class TestConfigManager:
    def test_load_base_config(self, base_config):
        cfg = ConfigManager.load(base_config)
        assert cfg.project.name == "test"
        assert cfg.dataset.num_classes == 3
        assert isinstance(cfg.dataset.class_names, list)

    def test_singleton_get(self, base_config):
        cfg1 = ConfigManager.load(base_config)
        cfg2 = ConfigManager.get()
        assert cfg1._cfg is cfg2._cfg

    def test_path_attribute(self, base_config):
        cfg = ConfigManager.load(base_config)
        assert hasattr(cfg, "paths")
        assert hasattr(cfg.paths, "checkpoints_dir")

    def test_as_dict(self, base_config):
        cfg = ConfigManager.load(base_config)
        d = cfg.as_dict()
        assert isinstance(d, dict)
        assert "project" in d
        assert "training" in d

    def test_overrides(self, base_config):
        cfg = ConfigManager.load(base_config, overrides={"training.epochs": 999})
        assert cfg.training.epochs == 999

    def test_to_yaml(self, base_config):
        cfg = ConfigManager.load(base_config)
        yaml_str = cfg.to_yaml()
        assert "project" in yaml_str

    def test_missing_get_raises(self):
        ConfigManager._instance = None
        with pytest.raises(RuntimeError, match="ConfigManager not initialised"):
            ConfigManager.get()

    def test_invalid_key_raises(self, base_config):
        cfg = ConfigManager.load(base_config)
        with pytest.raises(AttributeError):
            _ = cfg.nonexistent_key_xyz
