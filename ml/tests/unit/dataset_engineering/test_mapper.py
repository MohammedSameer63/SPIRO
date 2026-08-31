"""Tests for dataset mappers."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "src"))

import lib.ml.dataset_engineering.taxonomy as _tax_module


@pytest.fixture(autouse=True)
def reset_taxonomy():
    _tax_module.TaxonomyLoader._instance = None
    yield
    _tax_module.TaxonomyLoader._instance = None


class TestCOCOMapper:
    def test_coco_mapper_basic(self, coco_dataset, tmp_path):
        from lib.ml.dataset_engineering.mappers.mapper import COCOMapper

        mapper = COCOMapper(
            dataset_name="TestCOCO",
            mapping_file="taco_to_spiro.yaml",
            raw_dir=coco_dataset,
            annotations_json=coco_dataset / "annotations.json",
            images_dir=coco_dataset / "images",
            output_dir=tmp_path / "out",
            use_segmentation=False,
        )
        # Patch _mappings directly for the test
        mapper._mappings = {
            "plastic bottle": [0],
            "glass bottle": [22],
        }
        mapper._excluded = set()

        imgs, anns = mapper.map()
        assert imgs > 0
        assert anns > 0

    def test_coco_mapper_creates_yolo_labels(self, coco_dataset, tmp_path):
        from lib.ml.dataset_engineering.mappers.mapper import COCOMapper

        mapper = COCOMapper(
            dataset_name="TestCOCO",
            mapping_file="taco_to_spiro.yaml",
            raw_dir=coco_dataset,
            annotations_json=coco_dataset / "annotations.json",
            images_dir=coco_dataset / "images",
            output_dir=tmp_path / "out",
        )
        mapper._mappings = {"plastic bottle": [0], "glass bottle": [22]}
        mapper._excluded = set()
        mapper.map()

        label_files = list((tmp_path / "out" / "labels").iterdir())
        assert len(label_files) > 0
        for lf in label_files:
            for line in lf.read_text().splitlines():
                if not line.strip():
                    continue
                parts = line.split()
                assert len(parts) == 5
                cls_id = int(parts[0])
                assert 0 <= cls_id <= 108
                for v in map(float, parts[1:]):
                    assert 0.0 <= v <= 1.0

    def test_coco_bbox_to_yolo(self):
        from lib.ml.dataset_engineering.mappers.mapper import BaseMapper
        cx, cy, w, h = BaseMapper._coco_bbox_to_yolo(10, 20, 50, 40, 100, 100)
        assert abs(cx - 0.35) < 1e-5  # (10 + 50/2) / 100
        assert abs(cy - 0.40) < 1e-5  # (20 + 40/2) / 100
        assert abs(w - 0.50) < 1e-5
        assert abs(h - 0.40) < 1e-5

    def test_coco_bbox_clips_to_01(self):
        from lib.ml.dataset_engineering.mappers.mapper import BaseMapper
        cx, cy, w, h = BaseMapper._coco_bbox_to_yolo(-10, -10, 200, 200, 100, 100)
        assert 0.0 <= cx <= 1.0
        assert 0.0 <= cy <= 1.0
        assert 0.0 <= w <= 1.0
        assert 0.0 <= h <= 1.0


class TestClassificationFolderMapper:
    def test_maps_folder_to_yolo(self, classification_dataset, tmp_path):
        import yaml
        import shutil

        # Write a minimal mapping file
        mapping_path = tmp_path / "test_mapping.yaml"
        mapping_data = {
            "dataset": {"name": "test"},
            "mappings": {
                "plastic_bottle": {"spiro_ids": [0], "notes": ""},
                "glass_bottle": {"spiro_ids": [22], "notes": ""},
                "cardboard_box": {"spiro_ids": [36], "notes": ""},
            },
            "excluded": [],
            "pseudo_bbox": {"centre_crop_fraction": 0.85},
        }
        with open(mapping_path, "w") as f:
            yaml.dump(mapping_data, f)

        # Temporarily patch mappings dir
        from lib.ml.dataset_engineering.mappers.mapper import ClassificationFolderMapper
        mapper = ClassificationFolderMapper.__new__(ClassificationFolderMapper)
        mapper.dataset_name = "TestCls"
        mapper.mapping_file = mapping_path.name

        from lib.ml.dataset_engineering.taxonomy import TaxonomyLoader
        mapper.taxonomy = TaxonomyLoader(Path("configs/taxonomy/spiro_taxonomy.yaml"))
        mapper._mapping_cfg = mapping_data
        mapper._mappings = {"plastic_bottle": [0], "glass_bottle": [22], "cardboard_box": [36]}
        mapper._excluded = set()
        mapper._report = {
            "dataset": "TestCls", "images_converted": 0,
            "annotations_converted": 0, "annotations_skipped": 0,
            "class_distribution": {}, "unmapped_classes": [],
        }
        mapper.raw_dir = classification_dataset
        mapper.output_dir = tmp_path / "out"
        (tmp_path / "out" / "images").mkdir(parents=True)
        (tmp_path / "out" / "labels").mkdir(parents=True)
        mapper._root = classification_dataset
        mapper._crop_frac = 0.85

        imgs, anns = mapper.map()
        assert imgs == 15  # 3 classes × 5 images
        assert anns == 15  # 1 pseudo-bbox per image

    def test_pseudo_bbox_is_valid(self, classification_dataset, tmp_path):
        from lib.ml.dataset_engineering.mappers.mapper import ClassificationFolderMapper

        mapper = ClassificationFolderMapper.__new__(ClassificationFolderMapper)
        mapper.dataset_name = "TestCls"
        from lib.ml.dataset_engineering.taxonomy import TaxonomyLoader
        mapper.taxonomy = TaxonomyLoader(Path("configs/taxonomy/spiro_taxonomy.yaml"))
        mapper._mappings = {"plastic_bottle": [0]}
        mapper._excluded = set()
        mapper._mapping_cfg = {"pseudo_bbox": {"centre_crop_fraction": 0.85}}
        mapper._report = {
            "dataset": "TestCls", "images_converted": 0, "annotations_converted": 0,
            "annotations_skipped": 0, "class_distribution": {}, "unmapped_classes": [],
        }
        mapper.raw_dir = classification_dataset
        mapper.output_dir = tmp_path / "cls_out"
        (tmp_path / "cls_out" / "images").mkdir(parents=True)
        (tmp_path / "cls_out" / "labels").mkdir(parents=True)
        mapper._root = classification_dataset
        mapper._crop_frac = 0.85

        mapper.map()

        # Check one label file
        label_files = list((tmp_path / "cls_out" / "labels").iterdir())
        assert len(label_files) > 0
        for lf in label_files:
            content = lf.read_text().strip()
            if content:
                parts = content.split()
                assert len(parts) == 5
                # centre crop bbox should be centred
                cx, cy = float(parts[1]), float(parts[2])
                assert abs(cx - 0.5) < 0.01
                assert abs(cy - 0.5) < 0.01
