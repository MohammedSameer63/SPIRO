"""Tests for AugmentationPipeline (dataset engineering)."""
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "src"))


class TestAugmentationPipeline:
    def _make_split(self, tmp_path: Path, n: int = 5) -> Path:
        (tmp_path / "images").mkdir()
        (tmp_path / "labels").mkdir()
        for i in range(n):
            img = np.random.randint(50, 200, (128, 128, 3), dtype=np.uint8)
            cv2.imwrite(str(tmp_path / "images" / f"img_{i:04d}.jpg"), img)
            with open(tmp_path / "labels" / f"img_{i:04d}.txt", "w") as f:
                f.write("0 0.5 0.5 0.3 0.3\n")
        return tmp_path

    def test_augment_creates_new_images(self, tmp_path):
        from lib.ml.dataset_engineering.augmentation.pipeline import AugmentationPipeline
        split = self._make_split(tmp_path / "train", n=4)
        aug = AugmentationPipeline(image_size=(128, 128), factor=2, seed=0)
        written = aug.augment_split(split)
        assert written > 0
        total = len(list((split / "images").iterdir()))
        assert total == 4 + written

    def test_augmented_labels_valid(self, tmp_path):
        from lib.ml.dataset_engineering.augmentation.pipeline import AugmentationPipeline
        split = self._make_split(tmp_path / "train", n=3)
        aug = AugmentationPipeline(image_size=(128, 128), factor=1, seed=1)
        aug.augment_split(split)

        for lbl in (split / "labels").iterdir():
            if "_aug" not in lbl.stem:
                continue
            for line in lbl.read_text().splitlines():
                if not line.strip():
                    continue
                parts = line.split()
                assert len(parts) == 5
                for v in map(float, parts[1:]):
                    assert 0.0 <= v <= 1.0

    def test_augmented_images_readable(self, tmp_path):
        from lib.ml.dataset_engineering.augmentation.pipeline import AugmentationPipeline
        split = self._make_split(tmp_path / "train", n=3)
        aug = AugmentationPipeline(image_size=(128, 128), factor=1, seed=2)
        aug.augment_split(split)

        for img_path in (split / "images").iterdir():
            if "_aug" not in img_path.stem:
                continue
            img = cv2.imread(str(img_path))
            assert img is not None
            assert img.shape[0] > 0

    def test_heavy_transform_output_shape(self):
        from lib.ml.dataset_engineering.augmentation.pipeline import build_heavy_augmentation
        transform = build_heavy_augmentation((128, 128))
        img = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
        bboxes = [(0.5, 0.5, 0.3, 0.3)]
        cls = [0]
        result = transform(image=img, bboxes=bboxes, class_labels=cls)
        assert result["image"].ndim == 3
        assert result["image"].shape[2] == 3
        # Bboxes should still be valid if retained
        for box in result["bboxes"]:
            for v in box:
                assert 0.0 <= v <= 1.0

    def test_empty_bboxes_handled(self):
        from lib.ml.dataset_engineering.augmentation.pipeline import build_heavy_augmentation
        transform = build_heavy_augmentation((128, 128))
        img = np.zeros((128, 128, 3), dtype=np.uint8)
        result = transform(image=img, bboxes=[], class_labels=[])
        assert result["bboxes"] == []
