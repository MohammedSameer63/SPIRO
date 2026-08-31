"""
Shared fixtures for training pipeline tests.
No real training is performed in unit tests — all Ultralytics calls are mocked.
"""
import json
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest
import yaml


# ---------------------------------------------------------------------------
# Minimal YOLO dataset fixture
# ---------------------------------------------------------------------------

NUM_CLASSES = 5
IMG_SIZE = 64


@pytest.fixture(scope="session")
def tiny_yolo_dataset(tmp_path_factory) -> Path:
    """Create a minimal YOLO dataset (5 classes, 10 images per split)."""
    base = tmp_path_factory.mktemp("tiny_yolo")
    for split in ["train", "val", "test"]:
        (base / split / "images").mkdir(parents=True)
        (base / split / "labels").mkdir(parents=True)
        for i in range(10):
            img = np.random.randint(30, 200, (IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
            cv2.imwrite(str(base / split / "images" / f"img_{i:04d}.jpg"), img)
            cls = i % NUM_CLASSES
            with open(base / split / "labels" / f"img_{i:04d}.txt", "w") as f:
                f.write(f"{cls} 0.5 0.5 0.3 0.3\n")
    # Write dataset.yaml
    ds_yaml = base / "dataset.yaml"
    with open(ds_yaml, "w") as f:
        yaml.dump({
            "path": str(base),
            "train": "train/images",
            "val": "val/images",
            "test": "test/images",
            "nc": NUM_CLASSES,
            "names": {i: f"class_{i}" for i in range(NUM_CLASSES)},
        }, f)
    return base


@pytest.fixture(scope="session")
def training_config_path(tiny_yolo_dataset, tmp_path_factory) -> Path:
    """Write a minimal training config pointing to tiny_yolo_dataset."""
    cfg_dir = tmp_path_factory.mktemp("training_cfg")
    cfg_path = cfg_dir / "test_training.yaml"
    data = {
        "experiment": {
            "name": "test_run",
            "project": "spiro_test",
            "tags": ["test"],
            "notes": "",
            "seed": 0,
        },
        "model": {
            "weights": "yolo11n.pt",
            "num_classes": NUM_CLASSES,
            "input_size": IMG_SIZE,
            "pretrained": True,
            "task": "detect",
        },
        "dataset": {
            "yaml": str(tiny_yolo_dataset / "dataset.yaml"),
            "workers": 0,
            "cache": False,
            "rect": False,
            "single_cls": False,
            "class_names": [f"class_{i}" for i in range(NUM_CLASSES)],
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
            "benchmark": False,
        },
        "optimizer": {
            "name": "SGD",
            "lr0": 0.01,
            "lrf": 0.01,
            "momentum": 0.937,
            "weight_decay": 0.0005,
            "warmup_epochs": 0.0,
            "warmup_momentum": 0.8,
            "warmup_bias_lr": 0.1,
            "nbs": 64,
        },
        "scheduler": {
            "name": "cosine",
            "pct_start": 0.1,
            "div_factor": 25.0,
            "final_div_factor": 1e4,
        },
        "ema": {"enabled": True, "decay": 0.9999, "tau": 2000},
        "loss": {"box": 7.5, "cls": 0.5, "dfl": 1.5},
        "augmentation": {
            "enabled": False,
            "hsv_h": 0.0, "hsv_s": 0.0, "hsv_v": 0.0,
            "degrees": 0.0, "translate": 0.0, "scale": 0.0,
            "shear": 0.0, "perspective": 0.0, "flipud": 0.0, "fliplr": 0.0,
            "mosaic": 0.0, "mixup": 0.0, "copy_paste": 0.0,
            "label_smoothing": 0.0, "erasing": 0.0, "crop_fraction": 1.0,
            "auto_augment": "",
        },
        "validation": {
            "conf_threshold": 0.001,
            "iou_threshold": 0.6,
            "max_det": 300,
            "save_json": False,
            "save_txt": False,
            "plots": False,
        },
        "checkpoint": {
            "dir": str(tmp_path_factory.mktemp("checkpoints")),
            "save_best": True,
            "save_last": True,
            "metric": "mAP50-95",
        },
        "export": {
            "auto_export_onnx": False,
            "opset": 17,
            "dynamic": False,
            "simplify": False,
            "half": False,
            "verify": False,
        },
        "logging": {
            "tensorboard": False,
            "csv": False,
            "console": True,
            "log_dir": str(tmp_path_factory.mktemp("logs")),
            "tensorboard_dir": "logs/tensorboard",
            "csv_path": str(tmp_path_factory.mktemp("logs") / "training_history.csv"),
            "log_images_every_n_epochs": 1,
            "verbose": False,
        },
    }
    with open(cfg_path, "w") as f:
        yaml.dump(data, f, default_flow_style=False)
    return cfg_path


@pytest.fixture()
def mock_yolo_results() -> MagicMock:
    """Mock Ultralytics training results object."""
    m = MagicMock()
    m.results_dict = {
        "metrics/mAP50(B)": 0.55,
        "metrics/mAP50-95(B)": 0.35,
        "metrics/precision(B)": 0.70,
        "metrics/recall(B)": 0.65,
    }
    m.box = MagicMock()
    m.box.map50 = 0.55
    m.box.map = 0.35
    m.box.mp = 0.70
    m.box.mr = 0.65
    m.box.ap_class_index = [0, 1, 2]
    m.box.ap50 = [0.60, 0.50, 0.55]
    return m
