"""Tests for DatasetSplitter."""
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


class TestDatasetSplitter:
    def test_splits_sum_to_total(self, yolo_dataset, tmp_path):
        from lib.ml.dataset_engineering.splitters.splitter import DatasetSplitter

        splitter = DatasetSplitter(
            merged_dir=yolo_dataset,
            output_dir=tmp_path / "splits",
            train_ratio=0.70,
            val_ratio=0.15,
            test_ratio=0.15,
            seed=42,
        )
        report = splitter.split()
        total = sum(report["splits"].values())
        assert total == 30  # 30 images in yolo_dataset fixture

    def test_no_overlap_between_splits(self, yolo_dataset, tmp_path):
        from lib.ml.dataset_engineering.splitters.splitter import DatasetSplitter

        splitter = DatasetSplitter(
            merged_dir=yolo_dataset,
            output_dir=tmp_path / "splits",
            seed=0,
        )
        splitter.split()

        out = tmp_path / "splits"
        for split in ["train", "val", "test"]:
            stems_file = out / f"{split}.txt"
            assert stems_file.exists()

        train_stems = set((out / "train.txt").read_text().splitlines())
        val_stems = set((out / "val.txt").read_text().splitlines())
        test_stems = set((out / "test.txt").read_text().splitlines())

        assert len(train_stems & val_stems) == 0
        assert len(train_stems & test_stems) == 0
        assert len(val_stems & test_stems) == 0

    def test_split_dirs_created(self, yolo_dataset, tmp_path):
        from lib.ml.dataset_engineering.splitters.splitter import DatasetSplitter

        splitter = DatasetSplitter(
            merged_dir=yolo_dataset, output_dir=tmp_path / "splits", seed=42,
        )
        splitter.split()
        for split in ["train", "val", "test"]:
            assert (tmp_path / "splits" / split / "images").exists()
            assert (tmp_path / "splits" / split / "labels").exists()

    def test_ratios_must_sum_to_one(self, yolo_dataset, tmp_path):
        from lib.ml.dataset_engineering.splitters.splitter import DatasetSplitter
        with pytest.raises(AssertionError):
            DatasetSplitter(
                merged_dir=yolo_dataset, output_dir=tmp_path / "s",
                train_ratio=0.5, val_ratio=0.5, test_ratio=0.5,
            )

    def test_reproducible_with_same_seed(self, yolo_dataset, tmp_path):
        from lib.ml.dataset_engineering.splitters.splitter import DatasetSplitter

        out1 = tmp_path / "s1"
        out2 = tmp_path / "s2"
        for out in [out1, out2]:
            s = DatasetSplitter(merged_dir=yolo_dataset, output_dir=out, seed=99)
            s.split()

        train1 = set((out1 / "train.txt").read_text().splitlines())
        train2 = set((out2 / "train.txt").read_text().splitlines())
        assert train1 == train2

    def test_different_seeds_produce_different_splits(self, yolo_dataset, tmp_path):
        from lib.ml.dataset_engineering.splitters.splitter import DatasetSplitter

        out1 = tmp_path / "sa"
        out2 = tmp_path / "sb"
        DatasetSplitter(merged_dir=yolo_dataset, output_dir=out1, seed=1).split()
        DatasetSplitter(merged_dir=yolo_dataset, output_dir=out2, seed=2).split()

        train1 = set((out1 / "train.txt").read_text().splitlines())
        train2 = set((out2 / "train.txt").read_text().splitlines())
        # With 30 images two different seeds should produce at least one difference
        assert train1 != train2
