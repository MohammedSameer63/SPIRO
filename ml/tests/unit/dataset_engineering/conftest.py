"""
Shared fixtures for dataset engineering tests.
"""
import json
import shutil
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml


NUM_CLASSES = 109
IMG_W, IMG_H = 128, 128


def _make_image(path: Path, colour=None) -> None:
    colour = colour or np.random.randint(30, 220, 3).tolist()
    img = np.full((IMG_H, IMG_W, 3), colour, dtype=np.uint8)
    cv2.imwrite(str(path), img)


def _make_label(path: Path, num_boxes: int = 2, num_classes: int = 5) -> None:
    with open(path, "w") as f:
        for _ in range(num_boxes):
            cls = np.random.randint(0, num_classes)
            cx = np.random.uniform(0.2, 0.8)
            cy = np.random.uniform(0.2, 0.8)
            w = np.random.uniform(0.05, 0.3)
            h = np.random.uniform(0.05, 0.3)
            f.write(f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")


@pytest.fixture(scope="session")
def taxonomy_path(tmp_path_factory) -> Path:
    """Write a minimal 5-class taxonomy YAML for fast tests."""
    p = tmp_path_factory.mktemp("taxonomy") / "spiro_taxonomy.yaml"
    data = {
        "taxonomy": {
            "version": "1.0.0",
            "total_classes": 5,
            "description": "Test taxonomy",
            "classes": {0: "plastic_bottle", 1: "glass_bottle", 2: "cardboard_box",
                        3: "metal_can", 4: "unknown_litter"},
            "groups": {
                "plastic": [0],
                "glass": [1],
                "paper": [2],
                "metal": [3],
                "misc": [4],
            },
        }
    }
    with open(p, "w") as f:
        yaml.dump(data, f)
    return p


@pytest.fixture(scope="session")
def yolo_dataset(tmp_path_factory) -> Path:
    """Session-scoped YOLO-format dataset (30 images, 5 classes)."""
    root = tmp_path_factory.mktemp("yolo_ds")
    (root / "images").mkdir()
    (root / "labels").mkdir()
    np.random.seed(0)
    for i in range(30):
        _make_image(root / "images" / f"img_{i:04d}.jpg")
        _make_label(root / "labels" / f"img_{i:04d}.txt", num_boxes=2, num_classes=5)
    return root


@pytest.fixture(scope="session")
def classification_dataset(tmp_path_factory) -> Path:
    """Session-scoped image-folder classification dataset (3 classes, 5 each)."""
    root = tmp_path_factory.mktemp("cls_ds")
    for cls in ["plastic_bottle", "glass_bottle", "cardboard_box"]:
        (root / cls).mkdir()
        for i in range(5):
            _make_image(root / cls / f"{cls}_{i:03d}.jpg")
    return root


@pytest.fixture(scope="session")
def coco_dataset(tmp_path_factory) -> Path:
    """Session-scoped minimal COCO-format dataset."""
    root = tmp_path_factory.mktemp("coco_ds")
    (root / "images").mkdir()
    np.random.seed(1)

    images_meta = []
    annotations = []
    ann_id = 1
    categories = [
        {"id": 1, "name": "plastic bottle"},
        {"id": 2, "name": "glass bottle"},
    ]

    for i in range(10):
        img_name = f"img_{i:04d}.jpg"
        _make_image(root / "images" / img_name)
        images_meta.append({"id": i + 1, "file_name": img_name, "width": IMG_W, "height": IMG_H})
        # 2 annotations per image
        for _ in range(2):
            x = np.random.randint(5, 80)
            y = np.random.randint(5, 80)
            w = np.random.randint(10, 40)
            h = np.random.randint(10, 40)
            annotations.append({
                "id": ann_id, "image_id": i + 1,
                "category_id": np.random.choice([1, 2]),
                "bbox": [float(x), float(y), float(w), float(h)],
                "area": float(w * h),
                "iscrowd": 0,
            })
            ann_id += 1

    coco = {"images": images_meta, "annotations": annotations, "categories": categories}
    with open(root / "annotations.json", "w") as f:
        json.dump(coco, f)

    return root
