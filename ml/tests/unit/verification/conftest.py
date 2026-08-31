"""
Shared fixtures for verification pipeline tests.
No internet required — uses tiny synthetic data and untrained models.
"""
import sys
from pathlib import Path
from unittest.mock import MagicMock

import cv2
import numpy as np
import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

NUM_CLASSES = 5
INPUT_SIZE = 64


@pytest.fixture(autouse=True)
def reset_taxonomy():
    import lib.ml.dataset_engineering.taxonomy as _tax
    _tax.TaxonomyLoader._instance = None
    yield
    _tax.TaxonomyLoader._instance = None


@pytest.fixture(scope="session")
def verify_config_path(tmp_path_factory) -> Path:
    """Write a minimal VerifyConfig pointing to tiny synthetic dataset."""
    cfg_dir = tmp_path_factory.mktemp("verify_cfg")
    cfg_path = cfg_dir / "test_verify.yaml"

    data = {
        "experiment": {
            "name": "test_verify_run",
            "project": "spiro_test",
            "tags": ["test"],
            "notes": "",
            "seed": 0,
        },
        "model": {
            "variant": "efficientnetv2_b0",
            "timm_name": "tf_efficientnetv2_b0",
            "num_classes": NUM_CLASSES,
            "pretrained": False,
            "drop_rate": 0.0,
            "drop_path_rate": 0.0,
            "global_pool": "avg",
            "input_size": INPUT_SIZE,
        },
        "dataset": {
            "root": str(tmp_path_factory.mktemp("verify_ds")),
            "crop_dir": str(tmp_path_factory.mktemp("crops")),
            "workers": 0,
            "pin_memory": False,
            "class_names_file": None,
            "use_weighted_sampler": False,
            "oversample_rare_classes": False,
            "min_samples_per_class": 1,
        },
        "training": {
            "epochs": 2,
            "batch_size": 2,
            "device": "cpu",
            "amp": False,
            "gradient_accumulation": 1,
            "patience": 0,
            "save_period": -1,
            "val_period": 1,
            "resume": False,
            "resume_checkpoint": None,
            "deterministic": True,
            "num_workers": 0,
            "phase1_epochs": 1,
            "phase2_epochs": 1,
        },
        "optimizer": {
            "name": "AdamW",
            "lr0": 0.001,
            "lr_head": 0.005,
            "lrf": 0.01,
            "weight_decay": 0.0001,
            "momentum": 0.9,
            "betas": [0.9, 0.999],
        },
        "scheduler": {
            "name": "cosine",
            "warmup_epochs": 0,
            "step_size": 10,
            "step_gamma": 0.1,
        },
        "loss": {
            "name": "label_smoothing",
            "label_smoothing": 0.1,
            "focal_gamma": 2.0,
            "focal_alpha": None,
            "class_weights": None,
        },
        "augmentation": {
            "enabled": False,
            "random_crop": False,
            "crop_scale": [0.8, 1.0],
            "crop_ratio": [0.9, 1.1],
            "horizontal_flip": 0.0,
            "vertical_flip": 0.0,
            "rotation_degrees": 0,
            "perspective_distortion": 0.0,
            "brightness": 0.0,
            "contrast": 0.0,
            "saturation": 0.0,
            "hue": 0.0,
            "color_jitter_prob": 0.0,
            "grayscale_prob": 0.0,
            "blur_prob": 0.0,
            "blur_kernel": [3, 3],
            "clahe_prob": 0.0,
            "clahe_clip_limit": 4.0,
            "noise_prob": 0.0,
            "noise_var": [0, 1],
            "cutout_prob": 0.0,
            "cutout_holes": 0,
            "cutout_max_size": 1,
            "random_erasing_prob": 0.0,
            "random_erasing_scale": [0.02, 0.05],
            "tta_enabled": False,
            "tta_n": 1,
        },
        "evaluation": {
            "conf_threshold": 0.3,
            "top_k": 5,
            "save_confusion_matrix": False,
            "save_classification_report": False,
            "save_roc_curves": False,
            "save_per_class_accuracy": False,
            "misclassified_samples": 0,
        },
        "checkpoint": {
            "dir": str(tmp_path_factory.mktemp("verify_ckpt")),
            "best_metric": "top1_accuracy",
            "save_best": True,
            "save_last": True,
        },
        "export": {
            "auto_export_onnx": False,
            "opset": 17,
            "dynamic_axes": False,
            "simplify": False,
            "half": False,
            "verify": False,
            "output_dir": str(tmp_path_factory.mktemp("verify_exports")),
        },
        "fusion": {
            "method": "weighted_average",
            "yolo_weight": 0.4,
            "effnet_weight": 0.6,
            "temperature": 1.0,
            "bayesian_prior": None,
            "min_confidence": 0.1,
        },
        "explainability": {
            "enabled": False,
            "method": "gradcam",
            "target_layer": None,
            "save_heatmaps": False,
            "overlay_alpha": 0.5,
            "colormap": "jet",
            "output_dir": str(tmp_path_factory.mktemp("gradcam")),
            "explain_on_inference": False,
        },
        "logging": {
            "tensorboard": False,
            "csv": False,
            "console": True,
            "log_dir": str(tmp_path_factory.mktemp("verify_logs")),
            "tensorboard_dir": "logs/verification/tensorboard",
            "csv_path": str(tmp_path_factory.mktemp("verify_logs") / "history.csv"),
            "verbose": False,
        },
    }
    with open(cfg_path, "w") as f:
        yaml.dump(data, f, default_flow_style=False)
    return cfg_path


@pytest.fixture()
def dummy_bgr_image() -> np.ndarray:
    """Random 128×128 BGR image."""
    return np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)


@pytest.fixture()
def dummy_detections():
    """Fake YOLOv11 detections with probability vectors."""
    nc = 5
    probs = np.zeros(109)
    probs[0] = 0.7
    return [
        {
            "bbox_xyxy": [10.0, 20.0, 80.0, 100.0],
            "confidence": 0.82,
            "class_id": 0,
            "class_name": "plastic_bottle",
            "probabilities": probs.tolist(),
        }
    ]


@pytest.fixture()
def effnet_result():
    """Fake EfficientNetV2 verification result."""
    probs = np.zeros(109)
    probs[0] = 0.75
    probs[1] = 0.15
    return {
        "class_id": 0,
        "class_name": "plastic_bottle",
        "confidence": 0.75,
        "probabilities": probs.tolist(),
    }
