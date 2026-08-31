"""Tests for DatasetMerger."""
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


def _make_source(tmp_path: Path, name: str, n: int, cls_id: int = 0) -> Path:
    src = tmp_path / name
    (src / "images").mkdir(parents=True)
    (src / "labels").mkdir(parents=True)
    for i in range(n):
        img = np.random.randint(30, 220, (64, 64, 3), dtype=np.uint8)
        cv2.imwrite(str(src / "images" / f"{name}_{i:03d}.jpg"), img)
        (src / "labels" / f"{name}_{i:03d}.txt").write_text(
            f"{cls_id} 0.5 0.5 0.3 0.3\n"
        )
    return src


class TestDatasetMerger:
    def test_merge_combines_sources(self, tmp_path):
        from lib.ml.dataset_engineering.cleaners.merger import DatasetMerger

        s1 = _make_source(tmp_path / "src", "A", 6)
        s2 = _make_source(tmp_path / "src", "B", 4)
        out = tmp_path / "merged"

        merger = DatasetMerger(source_dirs=[s1, s2], output_dir=out, dedup_across_sources=False)
        report = merger.merge()

        assert report["total_images"] == 10
        assert len(list((out / "images").iterdir())) == 10
        assert len(list((out / "labels").iterdir())) == 10

    def test_dedup_removes_cross_source_duplicates(self, tmp_path):
        from lib.ml.dataset_engineering.cleaners.merger import DatasetMerger
        import shutil

        s1 = _make_source(tmp_path / "src", "S1", 5)
        s2 = _make_source(tmp_path / "src", "S2", 5)
        # Copy first image of S1 into S2 to create a cross-source duplicate
        shutil.copy2(
            s1 / "images" / "S1_000.jpg",
            s2 / "images" / "S2_dup.jpg",
        )
        (s2 / "labels" / "S2_dup.txt").write_text("0 0.5 0.5 0.3 0.3\n")

        out = tmp_path / "merged"
        merger = DatasetMerger(source_dirs=[s1, s2], output_dir=out, dedup_across_sources=True)
        report = merger.merge()

        # Duplicate should be detected (original is 10 + 1 dup = 11 images, dedup removes 1)
        assert report["cross_source_duplicates_removed"] >= 1
        assert report["total_images"] == 10  # original 10 without the dup

    def test_output_stems_unique(self, tmp_path):
        from lib.ml.dataset_engineering.cleaners.merger import DatasetMerger

        s1 = _make_source(tmp_path / "src", "A", 5)
        s2 = _make_source(tmp_path / "src", "B", 5)
        out = tmp_path / "merged"
        DatasetMerger([s1, s2], out, dedup_across_sources=False).merge()

        stems = [p.stem for p in (out / "images").iterdir()]
        assert len(stems) == len(set(stems)), "Duplicate stems in merged output"

    def test_per_source_report(self, tmp_path):
        from lib.ml.dataset_engineering.cleaners.merger import DatasetMerger

        s1 = _make_source(tmp_path / "src", "DS1", 3)
        s2 = _make_source(tmp_path / "src", "DS2", 4)
        out = tmp_path / "merged"
        merger = DatasetMerger([s1, s2], out, dedup_across_sources=False)
        report = merger.merge()

        assert "DS1" in report["per_source"]
        assert "DS2" in report["per_source"]
        assert report["per_source"]["DS1"]["images"] == 3
        assert report["per_source"]["DS2"]["images"] == 4

    def test_class_distribution_tracked(self, tmp_path):
        from lib.ml.dataset_engineering.cleaners.merger import DatasetMerger

        s1 = _make_source(tmp_path / "src", "X", 5, cls_id=0)  # plastic_bottle
        out = tmp_path / "merged"
        merger = DatasetMerger([s1], out, dedup_across_sources=False)
        report = merger.merge()

        # plastic_bottle (id=0) should appear in class_distribution
        assert report["class_distribution"].get("plastic_bottle", 0) == 5

    def test_empty_source_skipped(self, tmp_path):
        from lib.ml.dataset_engineering.cleaners.merger import DatasetMerger

        s1 = _make_source(tmp_path / "src", "Full", 5)
        empty = tmp_path / "empty_ds"
        (empty / "images").mkdir(parents=True)
        (empty / "labels").mkdir(parents=True)

        out = tmp_path / "merged"
        merger = DatasetMerger([s1, empty], out, dedup_across_sources=False)
        report = merger.merge()
        assert report["total_images"] == 5
