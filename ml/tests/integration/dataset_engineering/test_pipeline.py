"""
Integration test: full dataset engineering pipeline using synthetic data.
No internet required — downloads are skipped, data is generated locally.
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

import lib.ml.dataset_engineering.taxonomy as _tax_module


@pytest.fixture(autouse=True)
def reset_taxonomy():
    _tax_module.TaxonomyLoader._instance = None
    yield
    _tax_module.TaxonomyLoader._instance = None


def _make_mapped_dataset(root: Path, name: str, n: int = 20) -> Path:
    """Create a synthetic SPIRO YOLO-format dataset simulating mapper output."""
    ds = root / name
    (ds / "images").mkdir(parents=True)
    (ds / "labels").mkdir(parents=True)
    np.random.seed(hash(name) % 2**31)
    for i in range(n):
        img = np.random.randint(30, 200, (128, 128, 3), dtype=np.uint8)
        cv2.imwrite(str(ds / "images" / f"{name}_{i:04d}.jpg"), img)
        cls_id = np.random.randint(0, 5)
        with open(ds / "labels" / f"{name}_{i:04d}.txt", "w") as f:
            cx, cy = np.random.uniform(0.2, 0.8, 2)
            w, h = np.random.uniform(0.1, 0.4, 2)
            f.write(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")
    return ds


@pytest.mark.integration
class TestDatasetEngineeringPipeline:
    """End-to-end integration tests for dataset engineering pipeline."""

    def test_clean_then_merge_then_split(self, tmp_path):
        """Full pipeline: clean → merge → split."""
        from lib.ml.dataset_engineering.cleaners.cleaner import DataCleaner
        from lib.ml.dataset_engineering.cleaners.merger import DatasetMerger
        from lib.ml.dataset_engineering.splitters.splitter import DatasetSplitter

        # Create two synthetic mapped datasets
        mapped_dir = tmp_path / "mapped"
        ds1 = _make_mapped_dataset(mapped_dir, "DS1", n=20)
        ds2 = _make_mapped_dataset(mapped_dir, "DS2", n=15)

        # Clean
        for ds in [ds1, ds2]:
            cleaner = DataCleaner(ds, num_classes=109)
            report = cleaner.clean(remove=True)
            assert report["total_images"] > 0

        # Merge
        merged_dir = tmp_path / "merged"
        merger = DatasetMerger([ds1, ds2], merged_dir, dedup_across_sources=True)
        merge_report = merger.merge()
        assert merge_report["total_images"] > 0
        assert (merged_dir / "images").exists()
        assert (merged_dir / "labels").exists()

        # Split
        processed_dir = tmp_path / "processed"
        splitter = DatasetSplitter(
            merged_dir=merged_dir,
            output_dir=processed_dir,
            train_ratio=0.70, val_ratio=0.15, test_ratio=0.15,
            seed=42,
        )
        split_report = splitter.split()
        total_split = sum(split_report["splits"].values())
        assert total_split == merge_report["total_images"]

        # Check all split dirs populated
        for split in ["train", "val", "test"]:
            imgs = list((processed_dir / split / "images").iterdir())
            assert len(imgs) > 0, f"{split} split is empty"

    def test_statistics_after_merge(self, tmp_path):
        """Statistics computed correctly after merge."""
        from lib.ml.dataset_engineering.cleaners.merger import DatasetMerger
        from lib.ml.dataset_engineering.stats.statistics import DatasetStatistics

        mapped_dir = tmp_path / "mapped"
        ds1 = _make_mapped_dataset(mapped_dir, "StatDS1", n=15)
        ds2 = _make_mapped_dataset(mapped_dir, "StatDS2", n=10)

        merged_dir = tmp_path / "merged"
        merger = DatasetMerger([ds1, ds2], merged_dir, dedup_across_sources=False)
        merger.merge()

        stats_obj = DatasetStatistics(merged_dir, name="test_merge")
        report = stats_obj.compute()

        assert report["total_images"] == 25
        assert report["total_annotations"] > 0
        assert "group_coverage" in report
        assert "plastic" in report["group_coverage"]

        # Save and verify JSON
        out = tmp_path / "stats.json"
        stats_obj.save_json(out)
        data = json.loads(out.read_text())
        assert data["total_images"] == 25

    def test_versioning_after_split(self, tmp_path):
        """Version created with correct metadata after split."""
        from lib.ml.dataset_engineering.cleaners.merger import DatasetMerger
        from lib.ml.dataset_engineering.splitters.splitter import DatasetSplitter
        from lib.ml.dataset_engineering.versioning.versioning import DatasetVersioning

        mapped_dir = tmp_path / "mapped"
        ds = _make_mapped_dataset(mapped_dir, "VerDS", n=20)

        merged_dir = tmp_path / "merged"
        DatasetMerger([ds], merged_dir, dedup_across_sources=False).merge()

        processed_dir = tmp_path / "processed"
        DatasetSplitter(merged_dir, processed_dir, seed=42).split()

        dv = DatasetVersioning(tmp_path / "versions")
        vid = dv.create_version(
            processed_dir=processed_dir,
            sources=["VerDS"],
            notes="Integration test version",
        )

        entry = dv.get_version(vid)
        assert entry["total_images"] == 20
        assert entry["sources"] == ["VerDS"]
        assert "dataset_hash" in entry
        assert len(entry["dataset_hash"]) == 32  # MD5 hex

    def test_augmentation_after_split(self, tmp_path):
        """Augmentation multiplies training images correctly."""
        from lib.ml.dataset_engineering.cleaners.merger import DatasetMerger
        from lib.ml.dataset_engineering.splitters.splitter import DatasetSplitter
        from lib.ml.dataset_engineering.augmentation.pipeline import AugmentationPipeline

        mapped_dir = tmp_path / "mapped"
        ds = _make_mapped_dataset(mapped_dir, "AugDS", n=10)

        merged_dir = tmp_path / "merged"
        DatasetMerger([ds], merged_dir, dedup_across_sources=False).merge()

        processed_dir = tmp_path / "processed"
        DatasetSplitter(merged_dir, processed_dir, seed=42).split()

        train_dir = processed_dir / "train"
        orig_count = len(list((train_dir / "images").iterdir()))

        aug = AugmentationPipeline(image_size=(128, 128), factor=2, seed=0)
        written = aug.augment_split(train_dir)

        new_count = len(list((train_dir / "images").iterdir()))
        assert written > 0
        assert new_count == orig_count + written

    def test_visualization_generates_charts(self, tmp_path):
        """Visualization generates all chart files."""
        from lib.ml.dataset_engineering.cleaners.merger import DatasetMerger
        from lib.ml.dataset_engineering.stats.statistics import DatasetStatistics
        from lib.ml.dataset_engineering.visualization.charts import DatasetVisualizer

        mapped_dir = tmp_path / "mapped"
        ds = _make_mapped_dataset(mapped_dir, "VizDS", n=15)

        merged_dir = tmp_path / "merged"
        DatasetMerger([ds], merged_dir, dedup_across_sources=False).merge()

        stats_obj = DatasetStatistics(merged_dir, name="viz_test")
        report = stats_obj.compute()

        charts_dir = tmp_path / "charts"
        viz = DatasetVisualizer(report, charts_dir)
        viz.generate_all()

        expected_charts = [
            "class_histogram.png",
            "bbox_histogram.png",
            "resolution_scatter.png",
            "dataset_composition.png",
            "split_distribution.png",
            "group_coverage.png",
            "imbalance_heatmap.png",
        ]
        for chart in expected_charts:
            assert (charts_dir / chart).exists(), f"Missing chart: {chart}"

    def test_cleaning_report_written(self, tmp_path):
        """Cleaning report JSON is written correctly."""
        from lib.ml.dataset_engineering.cleaners.cleaner import DataCleaner

        ds = _make_mapped_dataset(tmp_path, "CleanDS", n=10)
        # Inject one corrupt image
        (ds / "images" / "corrupt.jpg").write_bytes(b"\x00" * 50)
        (ds / "labels" / "corrupt.txt").touch()

        cleaner = DataCleaner(ds, num_classes=109)
        cleaner.clean(remove=False)

        out = tmp_path / "cleaning_report.json"
        cleaner.save_report(out)

        assert out.exists()
        data = json.loads(out.read_text())
        assert data["total_images"] == 11
        assert len(data["corrupt_images"]) >= 1

    def test_markdown_report_content(self, tmp_path):
        """Markdown report contains required sections."""
        from lib.ml.dataset_engineering.cleaners.merger import DatasetMerger
        from lib.ml.dataset_engineering.stats.statistics import DatasetStatistics

        mapped_dir = tmp_path / "mapped"
        ds = _make_mapped_dataset(mapped_dir, "MdDS", n=10)

        merged_dir = tmp_path / "merged"
        DatasetMerger([ds], merged_dir, dedup_across_sources=False).merge()

        stats_obj = DatasetStatistics(merged_dir, name="MdDS")
        stats_obj.compute()

        out = tmp_path / "report.md"
        stats_obj.generate_markdown_report(out)

        content = out.read_text()
        assert "## Summary" in content
        assert "## Class Distribution" in content
        assert "## Group Coverage" in content
        assert "## Bounding Box Statistics" in content
