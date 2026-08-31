"""Tests for DatasetStatistics and DatasetVersioning."""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "src"))

import lib.ml.dataset_engineering.taxonomy as _tax_module


@pytest.fixture(autouse=True)
def reset_taxonomy():
    _tax_module.TaxonomyLoader._instance = None
    yield
    _tax_module.TaxonomyLoader._instance = None


class TestDatasetStatistics:
    def test_compute_returns_dict(self, yolo_dataset):
        from lib.ml.dataset_engineering.stats.statistics import DatasetStatistics
        stats = DatasetStatistics(yolo_dataset, name="test")
        report = stats.compute()
        assert isinstance(report, dict)
        assert report["total_images"] == 30
        assert report["total_annotations"] > 0

    def test_class_distribution_populated(self, yolo_dataset):
        from lib.ml.dataset_engineering.stats.statistics import DatasetStatistics
        stats = DatasetStatistics(yolo_dataset, name="test")
        report = stats.compute()
        assert isinstance(report["class_distribution"], dict)
        # yolo_dataset uses 5 classes (0–4); all should appear with the taxonomy names
        assert len(report["class_distribution"]) > 0

    def test_missing_classes_identified(self, yolo_dataset):
        from lib.ml.dataset_engineering.stats.statistics import DatasetStatistics
        stats = DatasetStatistics(yolo_dataset, name="test")
        report = stats.compute()
        # yolo_dataset uses only 5 of 109 classes
        assert len(report["missing_classes"]) > 0

    def test_group_coverage_present(self, yolo_dataset):
        from lib.ml.dataset_engineering.stats.statistics import DatasetStatistics
        stats = DatasetStatistics(yolo_dataset, name="test")
        report = stats.compute()
        assert "group_coverage" in report
        assert "plastic" in report["group_coverage"]

    def test_bbox_stats_present(self, yolo_dataset):
        from lib.ml.dataset_engineering.stats.statistics import DatasetStatistics
        stats = DatasetStatistics(yolo_dataset, name="test")
        report = stats.compute()
        assert "bbox_width_stats" in report
        assert "mean" in report["bbox_width_stats"]

    def test_save_json(self, yolo_dataset, tmp_path):
        from lib.ml.dataset_engineering.stats.statistics import DatasetStatistics
        stats = DatasetStatistics(yolo_dataset, name="test")
        stats.compute()
        out = tmp_path / "stats.json"
        stats.save_json(out)
        assert out.exists()
        loaded = json.loads(out.read_text())
        assert loaded["name"] == "test"

    def test_markdown_report(self, yolo_dataset, tmp_path):
        from lib.ml.dataset_engineering.stats.statistics import DatasetStatistics
        stats = DatasetStatistics(yolo_dataset, name="test")
        stats.compute()
        out = tmp_path / "report.md"
        stats.generate_markdown_report(out)
        assert out.exists()
        content = out.read_text()
        assert "Class Distribution" in content
        assert "Group Coverage" in content


class TestDatasetVersioning:
    def _make_processed(self, tmp_path: Path, n: int = 5) -> Path:
        import cv2
        import numpy as np
        proc = tmp_path / "processed"
        for split in ["train", "val", "test"]:
            (proc / split / "images").mkdir(parents=True)
            (proc / split / "labels").mkdir(parents=True)
            for i in range(n):
                img = np.zeros((64, 64, 3), dtype=np.uint8)
                cv2.imwrite(str(proc / split / "images" / f"img_{i}.jpg"), img)
                (proc / split / "labels" / f"img_{i}.txt").write_text("0 0.5 0.5 0.3 0.3\n")
        return proc

    def test_create_version(self, tmp_path):
        from lib.ml.dataset_engineering.versioning.versioning import DatasetVersioning
        proc = self._make_processed(tmp_path)
        dv = DatasetVersioning(tmp_path / "versions")
        vid = dv.create_version(proc, sources=["taco", "trashnet"])
        assert vid.startswith("v")
        assert len(vid) > 10

    def test_list_versions(self, tmp_path):
        from lib.ml.dataset_engineering.versioning.versioning import DatasetVersioning
        proc = self._make_processed(tmp_path)
        dv = DatasetVersioning(tmp_path / "versions")
        dv.create_version(proc, sources=["taco"])
        dv.create_version(proc, sources=["trashnet"])
        versions = dv.list_versions()
        assert len(versions) == 2

    def test_get_version(self, tmp_path):
        from lib.ml.dataset_engineering.versioning.versioning import DatasetVersioning
        proc = self._make_processed(tmp_path)
        dv = DatasetVersioning(tmp_path / "versions")
        vid = dv.create_version(proc, sources=["taco"])
        entry = dv.get_version(vid)
        assert entry["version_id"] == vid
        assert entry["sources"] == ["taco"]

    def test_latest(self, tmp_path):
        from lib.ml.dataset_engineering.versioning.versioning import DatasetVersioning
        proc = self._make_processed(tmp_path)
        dv = DatasetVersioning(tmp_path / "versions")
        v1 = dv.create_version(proc, sources=["a"])
        v2 = dv.create_version(proc, sources=["b"])
        assert dv.latest()["version_id"] == v2

    def test_delete_version(self, tmp_path):
        from lib.ml.dataset_engineering.versioning.versioning import DatasetVersioning
        proc = self._make_processed(tmp_path)
        dv = DatasetVersioning(tmp_path / "versions")
        vid = dv.create_version(proc, sources=["taco"])
        dv.delete_version(vid)
        with pytest.raises(KeyError):
            dv.get_version(vid)

    def test_hash_is_deterministic(self, tmp_path):
        from lib.ml.dataset_engineering.versioning.versioning import DatasetVersioning
        proc = self._make_processed(tmp_path)
        h1 = DatasetVersioning._hash_labels(proc)
        h2 = DatasetVersioning._hash_labels(proc)
        assert h1 == h2

    def test_different_labels_different_hash(self, tmp_path):
        from lib.ml.dataset_engineering.versioning.versioning import DatasetVersioning
        proc1 = self._make_processed(tmp_path / "p1")
        proc2 = self._make_processed(tmp_path / "p2")
        # Modify one label
        (proc2 / "train" / "labels" / "img_0.txt").write_text("1 0.4 0.4 0.2 0.2\n")
        h1 = DatasetVersioning._hash_labels(proc1)
        h2 = DatasetVersioning._hash_labels(proc2)
        assert h1 != h2
