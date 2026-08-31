"""Tests for DataCleaner."""
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "src"))

import lib.ml.dataset_engineering.taxonomy as _tax_module


@pytest.fixture(autouse=True)
def reset_taxonomy():
    _tax_module.TaxonomyLoader._instance = None
    yield
    _tax_module.TaxonomyLoader._instance = None


class TestDataCleaner:
    def _make_tiny_dataset(self, tmp_path: Path, n_images: int = 10) -> Path:
        """Create a small clean YOLO dataset."""
        (tmp_path / "images").mkdir()
        (tmp_path / "labels").mkdir()
        for i in range(n_images):
            img = np.random.randint(50, 200, (128, 128, 3), dtype=np.uint8)
            cv2.imwrite(str(tmp_path / "images" / f"img_{i:04d}.jpg"), img)
            with open(tmp_path / "labels" / f"img_{i:04d}.txt", "w") as f:
                f.write(f"0 0.5 0.5 0.3 0.3\n")
        return tmp_path

    def test_clean_clean_dataset(self, tmp_path):
        from lib.ml.dataset_engineering.cleaners.cleaner import DataCleaner
        ds = self._make_tiny_dataset(tmp_path / "ds", n_images=10)
        cleaner = DataCleaner(ds, num_classes=109)
        report = cleaner.clean(remove=False)
        assert report["total_images"] == 10
        assert len(report["corrupt_images"]) == 0
        assert len(report["exact_duplicates"]) == 0

    def test_detects_corrupt_image(self, tmp_path):
        from lib.ml.dataset_engineering.cleaners.cleaner import DataCleaner
        ds = self._make_tiny_dataset(tmp_path / "ds", n_images=5)
        # Write a corrupt file
        (ds / "images" / "corrupt.jpg").write_bytes(b"\x00" * 100)
        (ds / "labels" / "corrupt.txt").write_text("0 0.5 0.5 0.3 0.3\n")

        cleaner = DataCleaner(ds, num_classes=109)
        report = cleaner.clean(remove=False)
        assert len(report["corrupt_images"]) >= 1

    def test_detects_exact_duplicates(self, tmp_path):
        from lib.ml.dataset_engineering.cleaners.cleaner import DataCleaner
        ds = self._make_tiny_dataset(tmp_path / "ds", n_images=5)
        # Copy image to create exact duplicate
        import shutil
        src = ds / "images" / "img_0000.jpg"
        shutil.copy2(src, ds / "images" / "img_dup.jpg")
        (ds / "labels" / "img_dup.txt").write_text("0 0.5 0.5 0.3 0.3\n")

        cleaner = DataCleaner(ds, num_classes=109)
        report = cleaner.clean(remove=False)
        assert len(report["exact_duplicates"]) >= 1

    def test_detects_missing_labels(self, tmp_path):
        from lib.ml.dataset_engineering.cleaners.cleaner import DataCleaner
        ds = self._make_tiny_dataset(tmp_path / "ds", n_images=5)
        # Remove one label file
        (ds / "labels" / "img_0000.txt").unlink()

        cleaner = DataCleaner(ds, num_classes=109)
        report = cleaner.clean(remove=False)
        assert len(report["missing_labels"]) == 1

    def test_detects_orphan_labels(self, tmp_path):
        from lib.ml.dataset_engineering.cleaners.cleaner import DataCleaner
        ds = self._make_tiny_dataset(tmp_path / "ds", n_images=5)
        # Create orphan label
        (ds / "labels" / "orphan_file.txt").write_text("0 0.5 0.5 0.1 0.1\n")

        cleaner = DataCleaner(ds, num_classes=109)
        report = cleaner.clean(remove=False)
        assert len(report["orphan_labels"]) >= 1

    def test_detects_low_resolution(self, tmp_path):
        from lib.ml.dataset_engineering.cleaners.cleaner import DataCleaner
        ds = self._make_tiny_dataset(tmp_path / "ds", n_images=5)
        # Write tiny image
        tiny = np.zeros((10, 10, 3), dtype=np.uint8)
        cv2.imwrite(str(ds / "images" / "tiny.jpg"), tiny)
        (ds / "labels" / "tiny.txt").write_text("0 0.5 0.5 0.5 0.5\n")

        cleaner = DataCleaner(ds, min_resolution=(64, 64), num_classes=109)
        report = cleaner.clean(remove=False)
        assert len(report["low_resolution"]) >= 1

    def test_removes_corrupt_when_remove_true(self, tmp_path):
        from lib.ml.dataset_engineering.cleaners.cleaner import DataCleaner
        ds = self._make_tiny_dataset(tmp_path / "ds", n_images=5)
        (ds / "images" / "corrupt.jpg").write_bytes(b"\x00" * 50)
        (ds / "labels" / "corrupt.txt").touch()

        cleaner = DataCleaner(ds, num_classes=109)
        report = cleaner.clean(remove=True)
        # corrupt should be gone
        assert not (ds / "images" / "corrupt.jpg").exists()

    def test_invalid_bbox_detected(self, tmp_path):
        from lib.ml.dataset_engineering.cleaners.cleaner import DataCleaner
        ds = self._make_tiny_dataset(tmp_path / "ds", n_images=3)
        # Inject out-of-range bbox
        (ds / "labels" / "img_0000.txt").write_text("0 1.5 0.5 0.3 0.3\n")

        cleaner = DataCleaner(ds, num_classes=109)
        report = cleaner.clean(remove=False)
        assert len(report["invalid_bbox_images"]) >= 1

    def test_dhash_returns_int(self, tmp_path):
        from lib.ml.dataset_engineering.cleaners.cleaner import DataCleaner
        img = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
        p = tmp_path / "test.jpg"
        cv2.imwrite(str(p), img)
        result = DataCleaner._dhash(p)
        assert isinstance(result, int)

    def test_dhash_corrupt_returns_none(self, tmp_path):
        from lib.ml.dataset_engineering.cleaners.cleaner import DataCleaner
        p = tmp_path / "bad.jpg"
        p.write_bytes(b"\x00" * 10)
        result = DataCleaner._dhash(p)
        assert result is None
