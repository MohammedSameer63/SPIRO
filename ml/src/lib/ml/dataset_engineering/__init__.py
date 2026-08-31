"""
SPIRO ML — Dataset Engineering Package

Provides the full pipeline for downloading, cleaning, mapping,
merging, splitting, augmenting, and versioning waste datasets.

Usage
-----
from lib.ml.dataset_engineering import DatasetOrchestrator
orch = DatasetOrchestrator(cfg)
orch.run_full_pipeline()
"""
from lib.ml.dataset_engineering.orchestrator import DatasetOrchestrator

__all__ = ["DatasetOrchestrator"]
