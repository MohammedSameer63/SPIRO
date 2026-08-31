"""Tests for VerifyConfig."""
import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))


class TestVerifyConfig:
    def test_load_config(self, verify_config_path):
        from lib.ml.verification.verify_config import VerifyConfig
        cfg = VerifyConfig.load(verify_config_path)
        assert cfg.model.variant == "efficientnetv2_b0"
        assert cfg.model.num_classes == 5

    def test_timm_name_auto_resolved(self, verify_config_path):
        from lib.ml.verification.verify_config import VerifyConfig
        cfg = VerifyConfig.load(verify_config_path)
        assert cfg.timm_name == "tf_efficientnetv2_b0"
        assert cfg.model.timm_name == "tf_efficientnetv2_b0"

    def test_input_size_auto_resolved(self, verify_config_path):
        from lib.ml.verification.verify_config import VerifyConfig
        cfg = VerifyConfig.load(verify_config_path)
        assert cfg.input_size == 64  # from fixture (explicit)

    def test_override_variant(self, verify_config_path):
        from lib.ml.verification.verify_config import VerifyConfig
        cfg = VerifyConfig.load(
            verify_config_path,
            overrides={"model.variant": "efficientnetv2_s"},
        )
        assert cfg.model.variant == "efficientnetv2_s"
        assert "efficientnetv2_s" in cfg.timm_name

    def test_invalid_variant_raises(self, verify_config_path):
        from lib.ml.verification.verify_config import VerifyConfig
        with pytest.raises(ValueError, match="Unknown variant"):
            VerifyConfig.load(
                verify_config_path,
                overrides={"model.variant": "invalid_xyz"},
            )

    def test_missing_config_raises(self):
        from lib.ml.verification.verify_config import VerifyConfig
        with pytest.raises(FileNotFoundError):
            VerifyConfig.load("nonexistent/config.yaml")

    def test_as_dict(self, verify_config_path):
        from lib.ml.verification.verify_config import VerifyConfig
        cfg = VerifyConfig.load(verify_config_path)
        d = cfg.as_dict()
        assert isinstance(d, dict)
        assert "model" in d
        assert "training" in d
        assert "fusion" in d

    def test_to_yaml(self, verify_config_path):
        from lib.ml.verification.verify_config import VerifyConfig
        cfg = VerifyConfig.load(verify_config_path)
        yml = cfg.to_yaml()
        assert "model" in yml
        assert "efficientnetv2" in yml

    def test_base_inheritance(self, tmp_path):
        from lib.ml.verification.verify_config import VerifyConfig

        base = tmp_path / "base.yaml"
        base_data = {
            "experiment": {"name": "base", "project": "p", "tags": [], "notes": "", "seed": 0},
            "model": {
                "variant": "efficientnetv2_s", "timm_name": None,
                "num_classes": 109, "pretrained": True, "drop_rate": 0.2,
                "drop_path_rate": 0.2, "global_pool": "avg", "input_size": None,
            },
            "dataset": {
                "root": ".", "crop_dir": ".", "workers": 0, "pin_memory": False,
                "class_names_file": None, "use_weighted_sampler": False,
                "oversample_rare_classes": False, "min_samples_per_class": 1,
            },
            "training": {
                "epochs": 50, "batch_size": 32, "device": "cpu", "amp": False,
                "gradient_accumulation": 1, "patience": 15, "save_period": 5,
                "val_period": 1, "resume": False, "resume_checkpoint": None,
                "deterministic": True, "num_workers": 0,
                "phase1_epochs": 10, "phase2_epochs": 40,
            },
            "optimizer": {
                "name": "AdamW", "lr0": 0.001, "lr_head": 0.005,
                "lrf": 0.01, "weight_decay": 0.0001, "momentum": 0.9,
                "betas": [0.9, 0.999],
            },
            "scheduler": {"name": "cosine", "warmup_epochs": 3, "step_size": 10, "step_gamma": 0.1},
            "loss": {"name": "label_smoothing", "label_smoothing": 0.1,
                     "focal_gamma": 2.0, "focal_alpha": None, "class_weights": None},
            "augmentation": {
                "enabled": True, "random_crop": True, "crop_scale": [0.7, 1.0],
                "crop_ratio": [0.75, 1.33], "horizontal_flip": 0.5, "vertical_flip": 0.0,
                "rotation_degrees": 20, "perspective_distortion": 0.2,
                "brightness": 0.3, "contrast": 0.3, "saturation": 0.3, "hue": 0.1,
                "color_jitter_prob": 0.5, "grayscale_prob": 0.05, "blur_prob": 0.1,
                "blur_kernel": [3, 7], "clahe_prob": 0.2, "clahe_clip_limit": 4.0,
                "noise_prob": 0.1, "noise_var": [10, 50], "cutout_prob": 0.3,
                "cutout_holes": 8, "cutout_max_size": 32, "random_erasing_prob": 0.2,
                "random_erasing_scale": [0.02, 0.2], "tta_enabled": False, "tta_n": 5,
            },
            "evaluation": {
                "conf_threshold": 0.3, "top_k": 5, "save_confusion_matrix": True,
                "save_classification_report": True, "save_roc_curves": True,
                "save_per_class_accuracy": True, "misclassified_samples": 20,
            },
            "checkpoint": {"dir": ".", "best_metric": "top1_accuracy",
                           "save_best": True, "save_last": True},
            "export": {"auto_export_onnx": True, "opset": 17, "dynamic_axes": True,
                       "simplify": True, "half": False, "verify": True, "output_dir": "."},
            "fusion": {"method": "weighted_average", "yolo_weight": 0.4, "effnet_weight": 0.6,
                       "temperature": 1.5, "bayesian_prior": None, "min_confidence": 0.2},
            "explainability": {
                "enabled": True, "method": "gradcam", "target_layer": None,
                "save_heatmaps": True, "overlay_alpha": 0.5, "colormap": "jet",
                "output_dir": ".", "explain_on_inference": False,
            },
            "logging": {"tensorboard": False, "csv": False, "console": True,
                        "log_dir": ".", "tensorboard_dir": ".", "csv_path": "./hist.csv",
                        "verbose": False},
        }
        with open(base, "w") as f:
            yaml.dump(base_data, f)

        child = tmp_path / "child.yaml"
        with open(child, "w") as f:
            yaml.dump({
                "_base_": "./base.yaml",
                "experiment": {"name": "child"},
                "training": {"epochs": 77},
            }, f)

        cfg = VerifyConfig.load(child)
        assert cfg.training.epochs == 77
        assert cfg.experiment.name == "child"
        assert cfg.optimizer.name == "AdamW"  # inherited
