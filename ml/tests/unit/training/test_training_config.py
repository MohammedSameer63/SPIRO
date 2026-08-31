"""Tests for TrainingConfig."""
import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))


class TestTrainingConfig:
    def test_load_minimal_config(self, training_config_path):
        from lib.ml.training.training_config import TrainingConfig
        cfg = TrainingConfig.load(training_config_path)
        assert cfg.model.weights == "yolo11n.pt"
        assert cfg.model.num_classes == 5
        assert cfg.training.epochs == 2

    def test_override_epochs(self, training_config_path):
        from lib.ml.training.training_config import TrainingConfig
        cfg = TrainingConfig.load(training_config_path, overrides={"training.epochs": 99})
        assert cfg.training.epochs == 99

    def test_override_lr(self, training_config_path):
        from lib.ml.training.training_config import TrainingConfig
        cfg = TrainingConfig.load(training_config_path, overrides={"optimizer.lr0": 0.005})
        assert abs(cfg.optimizer.lr0 - 0.005) < 1e-9

    def test_override_device(self, training_config_path):
        from lib.ml.training.training_config import TrainingConfig
        cfg = TrainingConfig.load(training_config_path, overrides={"training.device": "cpu"})
        assert cfg.training.device == "cpu"

    def test_as_dict_returns_dict(self, training_config_path):
        from lib.ml.training.training_config import TrainingConfig
        cfg = TrainingConfig.load(training_config_path)
        d = cfg.as_dict()
        assert isinstance(d, dict)
        assert "training" in d
        assert "model" in d

    def test_missing_required_field_raises(self, tmp_path):
        from lib.ml.training.training_config import TrainingConfig
        # Write config missing 'dataset.yaml'
        bad_cfg = tmp_path / "bad.yaml"
        with open(bad_cfg, "w") as f:
            yaml.dump({
                "experiment": {"name": "x", "project": "p", "tags": [], "notes": "", "seed": 0},
                "model": {"weights": "yolo11n.pt", "num_classes": 5,
                          "input_size": 64, "pretrained": True, "task": "detect"},
                "training": {"epochs": 1, "batch_size": 2, "device": "cpu",
                             "amp": False, "gradient_accumulation": 1,
                             "patience": 0, "save_period": -1, "val_period": 1,
                             "resume": False, "resume_checkpoint": None,
                             "deterministic": True, "benchmark": False},
            }, f)
        with pytest.raises(ValueError, match="missing required fields"):
            TrainingConfig.load(bad_cfg)

    def test_config_not_found_raises(self):
        from lib.ml.training.training_config import TrainingConfig
        with pytest.raises(FileNotFoundError):
            TrainingConfig.load("nonexistent/path/config.yaml")

    def test_build_ultralytics_args_contains_required_keys(self, training_config_path):
        from lib.ml.training.training_config import TrainingConfig
        cfg = TrainingConfig.load(training_config_path)
        args = cfg.build_ultralytics_args()
        required = ["data", "epochs", "batch", "device", "optimizer", "lr0",
                    "mosaic", "fliplr", "box", "cls", "dfl"]
        for k in required:
            assert k in args, f"Missing key: {k}"

    def test_augmentation_disabled_sets_mosaic_zero(self, training_config_path):
        from lib.ml.training.training_config import TrainingConfig
        cfg = TrainingConfig.load(
            training_config_path,
            overrides={"augmentation.enabled": False, "augmentation.mosaic": 1.0}
        )
        args = cfg.build_ultralytics_args()
        assert args["mosaic"] == 0.0

    def test_resolve_device_auto_returns_string(self, training_config_path):
        from lib.ml.training.training_config import TrainingConfig
        result = TrainingConfig._resolve_device("auto")
        assert result in [0, "cpu", "mps"]

    def test_resolve_device_explicit(self, training_config_path):
        from lib.ml.training.training_config import TrainingConfig
        assert TrainingConfig._resolve_device("cpu") == "cpu"
        assert TrainingConfig._resolve_device("0") == "0"

    def test_base_inheritance(self, tmp_path):
        """Test _base_ config inheritance."""
        from lib.ml.training.training_config import TrainingConfig

        # Write base config
        base = tmp_path / "base.yaml"
        with open(base, "w") as f:
            yaml.dump({
                "experiment": {"name": "base", "project": "p", "tags": [], "notes": "", "seed": 0},
                "model": {"weights": "yolo11n.pt", "num_classes": 5,
                          "input_size": 64, "pretrained": True, "task": "detect"},
                "dataset": {"yaml": "configs/dataset.yaml", "workers": 0, "cache": False,
                            "rect": False, "single_cls": False},
                "training": {"epochs": 100, "batch_size": 16, "device": "cpu",
                             "amp": False, "gradient_accumulation": 1,
                             "patience": 50, "save_period": -1, "val_period": 1,
                             "resume": False, "resume_checkpoint": None,
                             "deterministic": True, "benchmark": False},
                "optimizer": {"name": "SGD", "lr0": 0.01, "lrf": 0.01,
                              "momentum": 0.937, "weight_decay": 0.0005,
                              "warmup_epochs": 0.0, "warmup_momentum": 0.8,
                              "warmup_bias_lr": 0.1, "nbs": 64},
                "scheduler": {"name": "cosine", "pct_start": 0.1,
                              "div_factor": 25.0, "final_div_factor": 1e4},
                "ema": {"enabled": True, "decay": 0.9999, "tau": 2000},
                "loss": {"box": 7.5, "cls": 0.5, "dfl": 1.5},
                "augmentation": {"enabled": True, "hsv_h": 0.015, "hsv_s": 0.7,
                                 "hsv_v": 0.4, "degrees": 0.0, "translate": 0.1,
                                 "scale": 0.5, "shear": 0.0, "perspective": 0.0,
                                 "flipud": 0.0, "fliplr": 0.5, "mosaic": 1.0,
                                 "mixup": 0.0, "copy_paste": 0.0,
                                 "label_smoothing": 0.0, "erasing": 0.4,
                                 "crop_fraction": 1.0, "auto_augment": ""},
                "validation": {"conf_threshold": 0.001, "iou_threshold": 0.6,
                               "max_det": 300, "save_json": False, "save_txt": False,
                               "plots": False},
                "checkpoint": {"dir": str(tmp_path / "ckpt"), "save_best": True,
                               "save_last": True, "metric": "mAP50-95"},
                "export": {"auto_export_onnx": False, "opset": 17, "dynamic": True,
                           "simplify": True, "half": False, "verify": True},
                "logging": {"tensorboard": False, "csv": False, "console": True,
                            "log_dir": str(tmp_path / "logs"), "tensorboard_dir": "logs/tb",
                            "csv_path": str(tmp_path / "hist.csv"),
                            "log_images_every_n_epochs": 5, "verbose": False},
            }, f)

        # Write child config that overrides epochs
        child = tmp_path / "child.yaml"
        with open(child, "w") as f:
            yaml.dump({
                "_base_": "./base.yaml",
                "training": {"epochs": 42},
                "experiment": {"name": "child_run"},
            }, f)

        cfg = TrainingConfig.load(child)
        # Child overrides
        assert cfg.training.epochs == 42
        assert cfg.experiment.name == "child_run"
        # Base values inherited
        assert cfg.optimizer.lr0 == 0.01
        assert cfg.model.weights == "yolo11n.pt"
