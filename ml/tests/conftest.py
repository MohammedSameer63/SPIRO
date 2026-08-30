"""
SPIRO ML — Shared test fixtures.
Loaded automatically by pytest via conftest.py discovery.
"""
import json
import shutil
import tempfile
from pathlib import Path
from typing import Generator

import cv2
import numpy as np
import pytest
import yaml

# ---------------------------------------------------------------------------
# Tiny synthetic dataset fixture
# ---------------------------------------------------------------------------

NUM_CLASSES = 3
CLASS_NAMES = ["cat", "dog", "bird"]
IMG_W, IMG_H = 128, 128


def _make_image(path: Path) -> None:
    """Write a solid-colour random BGR image."""
    colour = np.random.randint(0, 255, 3).tolist()
    img = np.full((IMG_H, IMG_W, 3), colour, dtype=np.uint8)
    cv2.imwrite(str(path), img)


def _make_label(path: Path, num_boxes: int = 2) -> None:
    """Write random YOLO-format labels."""
    with open(path, "w") as f:
        for _ in range(num_boxes):
            cls = np.random.randint(0, NUM_CLASSES)
            cx, cy = np.random.uniform(0.2, 0.8, 2)
            w, h = np.random.uniform(0.1, 0.3, 2)
            f.write(f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")


@pytest.fixture(scope="session")
def tmp_dataset(tmp_path_factory) -> Generator[Path, None, None]:
    """
    Session-scoped fixture: creates a minimal raw YOLO dataset.
    Structure:
        <tmp>/
          raw/
            images/  (20 .jpg files)
            labels/  (20 .txt files)
    """
    base = tmp_path_factory.mktemp("spiro_test_dataset")
    raw_images = base / "raw" / "images"
    raw_labels = base / "raw" / "labels"
    raw_images.mkdir(parents=True)
    raw_labels.mkdir(parents=True)

    for i in range(20):
        _make_image(raw_images / f"img_{i:04d}.jpg")
        _make_label(raw_labels / f"img_{i:04d}.txt", num_boxes=np.random.randint(1, 4))

    yield base


@pytest.fixture(scope="session")
def base_config(tmp_dataset, tmp_path_factory) -> Path:
    """
    Write a minimal config YAML pointing to tmp_dataset.
    Returns the path to the config file.
    """
    cfg_dir = tmp_path_factory.mktemp("spiro_test_cfg")
    cfg_path = cfg_dir / "test_config.yaml"

    cfg = {
        "project": {"name": "test", "version": "0.0.1", "seed": 0},
        "paths": {
            "data_root": str(tmp_dataset),
            "raw_data": str(tmp_dataset / "raw"),
            "processed_data": str(tmp_dataset / "processed"),
            "splits_dir": str(tmp_dataset / "splits"),
            "augmented_data": str(tmp_dataset / "augmented"),
            "models_dir": str(tmp_dataset / "models"),
            "checkpoints_dir": str(tmp_dataset / "models" / "checkpoints"),
            "exports_dir": str(tmp_dataset / "models" / "exports"),
            "registry_dir": str(tmp_dataset / "models" / "registry"),
            "logs_dir": str(tmp_dataset / "logs"),
            "tensorboard_dir": str(tmp_dataset / "logs" / "tensorboard"),
            "mlflow_dir": str(tmp_dataset / "logs" / "mlflow"),
            "reports_dir": str(tmp_dataset / "reports"),
        },
        "dataset": {
            "name": "test_dataset",
            "format": "yolo",
            "num_classes": NUM_CLASSES,
            "class_names": CLASS_NAMES,
            "image_size": [IMG_W, IMG_H],
            "channels": 3,
            "split_ratios": {"train": 0.7, "val": 0.15, "test": 0.15},
            "augmentation_factor": 1,
            "cache_images": False,
            "workers": 0,
        },
        "model": {
            "architecture": "yolov11",
            "variant": "yolo11n.pt",
            "pretrained": True,
            "pretrained_weights": "yolo11n.pt",
            "input_size": [IMG_W, IMG_H],
            "num_classes": NUM_CLASSES,
            "task": "detect",
        },
        "efficientnetv2": {
            "variant": "tf_efficientnetv2_s",
            "pretrained": False,
            "drop_rate": 0.0,
            "drop_path_rate": 0.0,
            "num_classes": NUM_CLASSES,
            "global_pool": "avg",
        },
        "training": {
            "epochs": 1,
            "batch_size": 2,
            "learning_rate": 0.01,
            "lr_scheduler": "cosine",
            "warmup_epochs": 0,
            "warmup_momentum": 0.8,
            "weight_decay": 0.0005,
            "momentum": 0.937,
            "optimizer": "SGD",
            "amp": False,
            "gradient_clip": 10.0,
            "accumulate_grad_batches": 1,
            "patience": 50,
            "save_period": 1,
            "val_period": 1,
            "device": "cpu",
            "workers": 0,
            "pin_memory": False,
            "resume": False,
            "resume_checkpoint": None,
        },
        "loss": {"box": 7.5, "cls": 0.5, "dfl": 1.5},
        "augmentation": {
            "horizontal_flip": 0.5, "vertical_flip": 0.0,
            "rotate_limit": 10, "rotate_prob": 0.2,
            "scale_limit": 0.1, "scale_prob": 0.2,
            "shift_limit": 0.05, "shift_prob": 0.2,
            "perspective_distortion": 0.0, "perspective_prob": 0.0,
            "brightness_limit": 0.1, "contrast_limit": 0.1,
            "color_prob": 0.2, "hue_shift": 10, "sat_shift": 10, "val_shift": 10,
            "hsv_prob": 0.2, "blur_prob": 0.0, "gaussian_noise_prob": 0.0,
            "cutout_prob": 0.0, "cutout_num_holes": 2,
            "cutout_max_h": 10, "cutout_max_w": 10,
            "mosaic": 0.0, "mixup": 0.0, "copy_paste": 0.0,
        },
        "evaluation": {
            "conf_threshold": 0.25, "iou_threshold": 0.45,
            "max_detections": 100,
            "metrics": ["mAP50"], "save_plots": False,
            "save_confusion_matrix": False, "save_pr_curve": False,
        },
        "export": {
            "format": "onnx", "opset_version": 17,
            "dynamic_axes": True, "simplify": False,
            "half_precision": False,
            "input_names": ["images"], "output_names": ["output0"],
            "verify_export": True,
        },
        "inference": {
            "conf_threshold": 0.4, "iou_threshold": 0.45,
            "max_detections": 100, "device": "cpu",
            "providers": ["CPUExecutionProvider"],
            "batch_size": 1, "warmup_runs": 0,
        },
        "continuous_learning": {
            "enabled": True, "trigger_threshold": 0.05,
            "min_new_samples": 5, "retrain_schedule": "weekly",
            "drift_detection": True, "drift_window": 20,
            "registry_max_versions": 3,
        },
        "logging": {
            "level": "DEBUG", "tensorboard": False,
            "mlflow": False, "console": True, "file": False,
            "log_images_every_n_epochs": 1,
        },
    }

    with open(cfg_path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)

    return cfg_path


@pytest.fixture()
def dummy_image() -> np.ndarray:
    """Return a random 480×640 BGR image."""
    return np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)


@pytest.fixture()
def dummy_detections():
    """Return a list of fake detection dicts."""
    return [
        {
            "bbox_xyxy": [10.0, 20.0, 100.0, 150.0],
            "confidence": 0.85,
            "class_id": 0,
            "class_name": "cat",
        },
        {
            "bbox_xyxy": [200.0, 50.0, 350.0, 200.0],
            "confidence": 0.72,
            "class_id": 1,
            "class_name": "dog",
        },
    ]
