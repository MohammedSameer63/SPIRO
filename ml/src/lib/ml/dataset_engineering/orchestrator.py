"""
SPIRO ML — DatasetOrchestrator
Top-level pipeline that wires all dataset engineering components together.

Pipeline steps:
  1. Download datasets
  2. Map to SPIRO taxonomy
  3. Clean (dedup, corrupt detection, bbox validation)
  4. Merge all mapped datasets
  5. Split (train/val/test)
  6. Augment training split
  7. Compute statistics
  8. Generate visualizations
  9. Create version snapshot
  10. Write all reports
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from lib.ml.core.logger import get_logger
from lib.ml.dataset_engineering.cleaners.cleaner import DataCleaner
from lib.ml.dataset_engineering.cleaners.merger import DatasetMerger
from lib.ml.dataset_engineering.mappers.mapper import (
    build_taco_mapper,
    build_trashnet_mapper,
    build_zerowaste_mapper,
    build_olm_mapper,
    build_kaggle_gc_mapper,
)
from lib.ml.dataset_engineering.splitters.splitter import DatasetSplitter
from lib.ml.dataset_engineering.augmentation.pipeline import AugmentationPipeline
from lib.ml.dataset_engineering.stats.statistics import DatasetStatistics
from lib.ml.dataset_engineering.versioning.versioning import DatasetVersioning
from lib.ml.dataset_engineering.visualization.charts import DatasetVisualizer

log = get_logger(__name__)


class DatasetOrchestrator:
    """
    Runs the complete SPIRO dataset engineering pipeline.

    Parameters
    ----------
    raw_dir : Path
        Root of downloaded raw datasets.
    mapped_dir : Path
        Output dir for mapped datasets.
    merged_dir : Path
        Output dir for merged dataset.
    processed_dir : Path
        Output dir for train/val/test splits.
    reports_dir : Path
        Output dir for all reports and charts.
    skip_download : bool
        Skip download step (data already present).
    datasets : list[str], optional
        Subset of datasets to process. None = all.
    augmentation_factor : int
        N× augmentation multiplier for training split.
    seed : int
        Random seed.

    Example
    -------
    >>> orch = DatasetOrchestrator()
    >>> orch.run_full_pipeline(skip_download=True)
    """

    _ALL_DATASETS = ["taco", "trashnet", "zerowaste", "openlittermap", "kaggle_gc"]

    def __init__(
        self,
        raw_dir: Path = Path("datasets/raw"),
        mapped_dir: Path = Path("datasets/mapped"),
        merged_dir: Path = Path("datasets/merged"),
        processed_dir: Path = Path("datasets/processed"),
        reports_dir: Path = Path("reports"),
        skip_download: bool = False,
        datasets: Optional[List[str]] = None,
        augmentation_factor: int = 3,
        seed: int = 42,
    ) -> None:
        self.raw_dir = Path(raw_dir)
        self.mapped_dir = Path(mapped_dir)
        self.merged_dir = Path(merged_dir)
        self.processed_dir = Path(processed_dir)
        self.reports_dir = Path(reports_dir)
        self.skip_download = skip_download
        self.datasets = datasets or self._ALL_DATASETS
        self.augmentation_factor = augmentation_factor
        self.seed = seed

        self._pipeline_report: Dict[str, Any] = {
            "started_at": datetime.utcnow().isoformat(),
            "steps": {},
        }

    # ------------------------------------------------------------------
    # Full pipeline
    # ------------------------------------------------------------------

    def run_full_pipeline(self) -> Dict[str, Any]:
        """Execute all pipeline steps in order."""
        log.info("═══════════════════════════════════════════")
        log.info("   SPIRO Dataset Engineering Pipeline")
        log.info("═══════════════════════════════════════════")

        # 1. Download
        if not self.skip_download:
            self._step_download()

        # 2. Map
        mapping_reports = self._step_map()

        # 3. Clean
        cleaning_reports = self._step_clean()

        # 4. Merge
        merge_report = self._step_merge()

        # 5. Split
        split_report = self._step_split()

        # 6. Augment
        self._step_augment()

        # 7. Statistics
        stats = self._step_statistics()

        # 8. Visualize
        self._step_visualize(stats)

        # 9. Version
        version_id = self._step_version(stats)

        # 10. Reports
        self._write_reports(
            mapping_reports, cleaning_reports, merge_report,
            split_report, stats, version_id
        )

        self._pipeline_report["completed_at"] = datetime.utcnow().isoformat()
        log.info("Pipeline complete ✓")
        return self._pipeline_report

    # ------------------------------------------------------------------
    # Individual steps
    # ------------------------------------------------------------------

    def _step_download(self) -> None:
        log.info("── Step 1: Downloading datasets")
        from lib.ml.dataset_engineering.downloaders.base import download_all
        results = download_all(raw_dir=self.raw_dir)
        self._pipeline_report["steps"]["download"] = {
            "downloaded": list(results.keys())
        }

    def _step_map(self) -> Dict[str, Any]:
        log.info("── Step 2: Mapping to SPIRO taxonomy")
        reports: Dict[str, Any] = {}

        mapper_factories = {
            "taco": lambda: build_taco_mapper(self.raw_dir, self.mapped_dir / "TACO"),
            "trashnet": lambda: build_trashnet_mapper(self.raw_dir, self.mapped_dir / "TrashNet"),
            "zerowaste": lambda: build_zerowaste_mapper(self.raw_dir, self.mapped_dir / "ZeroWaste"),
            "openlittermap": lambda: build_olm_mapper(self.raw_dir, self.mapped_dir / "OpenLitterMap"),
            "kaggle_gc": lambda: build_kaggle_gc_mapper(self.raw_dir, self.mapped_dir / "KaggleGC"),
        }

        for name in self.datasets:
            if name not in mapper_factories:
                continue
            try:
                mapper = mapper_factories[name]()
                imgs, anns = mapper.map()
                rpt = mapper.report()
                reports[name] = rpt
                log.info(f"  {name}: {imgs} images, {anns} annotations mapped")
            except Exception as e:
                log.error(f"  {name} mapping failed: {e}")
                reports[name] = {"error": str(e)}

        self._pipeline_report["steps"]["map"] = {"datasets": list(reports.keys())}
        return reports

    def _step_clean(self) -> Dict[str, Any]:
        log.info("── Step 3: Cleaning mapped datasets")
        reports: Dict[str, Any] = {}

        for dataset_dir in sorted(self.mapped_dir.iterdir()):
            if not dataset_dir.is_dir():
                continue
            if not (dataset_dir / "images").exists():
                continue
            try:
                cleaner = DataCleaner(dataset_dir)
                report = cleaner.clean(remove=True)
                cleaner.save_report(self.reports_dir / f"{dataset_dir.name}_cleaning_report.json")
                reports[dataset_dir.name] = report
                log.info(
                    f"  {dataset_dir.name}: kept {report['clean_images']} "
                    f"(removed corrupt={len(report['corrupt_images'])}, "
                    f"dups={len(report['exact_duplicates'])})"
                )
            except Exception as e:
                log.error(f"  {dataset_dir.name} cleaning failed: {e}")

        self._pipeline_report["steps"]["clean"] = {"datasets_cleaned": list(reports.keys())}
        return reports

    def _step_merge(self) -> Dict[str, Any]:
        log.info("── Step 4: Merging all mapped datasets")
        source_dirs = [
            d for d in sorted(self.mapped_dir.iterdir())
            if d.is_dir() and (d / "images").exists()
        ]
        if not source_dirs:
            log.warning("No mapped datasets found to merge")
            return {}

        merger = DatasetMerger(source_dirs=source_dirs, output_dir=self.merged_dir)
        report = merger.merge()
        merger.save_report(self.reports_dir / "merge_report.json")
        self._pipeline_report["steps"]["merge"] = {
            "total_images": report["total_images"],
            "total_annotations": report["total_annotations"],
        }
        return report

    def _step_split(self) -> Dict[str, Any]:
        log.info("── Step 5: Splitting dataset")
        splitter = DatasetSplitter(
            merged_dir=self.merged_dir,
            output_dir=self.processed_dir,
            seed=self.seed,
        )
        report = splitter.split()
        splitter.save_report(self.reports_dir / "split_report.json")
        self._pipeline_report["steps"]["split"] = report["splits"]
        return report

    def _step_augment(self) -> None:
        log.info(f"── Step 6: Augmenting training split (factor={self.augmentation_factor})")
        train_dir = self.processed_dir / "train"
        if not train_dir.exists():
            log.warning("No train/ dir found, skipping augmentation")
            return
        aug = AugmentationPipeline(factor=self.augmentation_factor, seed=self.seed)
        n = aug.augment_split(train_dir)
        self._pipeline_report["steps"]["augment"] = {"new_images": n}

    def _step_statistics(self) -> Dict[str, Any]:
        log.info("── Step 7: Computing statistics")
        stats_obj = DatasetStatistics(self.merged_dir, name="SPIRO-Merged")
        stats = stats_obj.compute()
        stats_obj.save_json(self.reports_dir / "dataset_statistics.json")
        stats_obj.generate_markdown_report(self.reports_dir / "dataset_report.md")
        self._pipeline_report["steps"]["statistics"] = {
            "total_images": stats.get("total_images"),
            "total_annotations": stats.get("total_annotations"),
            "represented_classes": stats.get("represented_classes"),
        }
        return stats

    def _step_visualize(self, stats: Dict) -> None:
        log.info("── Step 8: Generating visualizations")
        viz = DatasetVisualizer(stats, self.reports_dir / "charts")
        viz.generate_all()
        self._pipeline_report["steps"]["visualize"] = {
            "output_dir": str(self.reports_dir / "charts")
        }

    def _step_version(self, stats: Dict) -> str:
        log.info("── Step 9: Creating dataset version")
        dv = DatasetVersioning(self.processed_dir.parent)
        version_id = dv.create_version(
            processed_dir=self.processed_dir,
            sources=self.datasets,
            stats=stats,
            notes="Auto-generated by DatasetOrchestrator",
        )
        dv.save_json(self.reports_dir / "dataset_version.json")
        self._pipeline_report["steps"]["version"] = {"version_id": version_id}
        return version_id

    def _write_reports(
        self,
        mapping_reports: Dict,
        cleaning_reports: Dict,
        merge_report: Dict,
        split_report: Dict,
        stats: Dict,
        version_id: str,
    ) -> None:
        log.info("── Step 10: Writing consolidated reports")
        self.reports_dir.mkdir(parents=True, exist_ok=True)

        # mapping_report.json
        with open(self.reports_dir / "mapping_report.json", "w") as f:
            json.dump(mapping_reports, f, indent=2)

        # cleaning_report.json (consolidated)
        with open(self.reports_dir / "cleaning_report.json", "w") as f:
            json.dump(cleaning_reports, f, indent=2)

        # Full pipeline report
        with open(self.reports_dir / "pipeline_report.json", "w") as f:
            json.dump(self._pipeline_report, f, indent=2)

        log.info(f"All reports written to {self.reports_dir}")
