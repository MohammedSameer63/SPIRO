#!/usr/bin/env python3
"""
SPIRO ML — Dataset Manager CLI
Validates, preprocesses, splits, and analyses the SPIRO dataset.

Usage
-----
    python -m lib.ml.scripts.dataset_manager --config configs/base_config.yaml --action all
    python -m lib.ml.scripts.dataset_manager --action validate
    python -m lib.ml.scripts.dataset_manager --action split
    python -m lib.ml.scripts.dataset_manager --action augment --factor 3
    python -m lib.ml.scripts.dataset_manager --action analyze
"""
import argparse
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SPIRO Dataset Manager")
    p.add_argument("--config", default="configs/base_config.yaml", help="Path to YAML config")
    p.add_argument(
        "--action",
        choices=["validate", "preprocess", "split", "augment", "analyze", "all"],
        default="all",
        help="Pipeline step to run",
    )
    p.add_argument("--factor", type=int, default=3, help="Augmentation factor (--action augment)")
    p.add_argument("--output-stats", default="reports/dataset_stats.json", help="Stats output path")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Local imports after path setup
    sys.path.insert(0, str(Path(__file__).resolve().parents[5]))
    from lib.ml.core.config import ConfigManager
    from lib.ml.core.logger import get_logger
    from lib.ml.data.dataset_manager import DatasetManager
    from lib.ml.data.augmentation import OfflineAugmentor

    log = get_logger("dataset_manager")
    cfg = ConfigManager.load(args.config)
    dm = DatasetManager(cfg)

    def run_validate():
        log.info("=== STEP: Validate ===")
        dm.validate_raw()

    def run_preprocess():
        log.info("=== STEP: Preprocess ===")
        dm.preprocess()

    def run_split():
        log.info("=== STEP: Split ===")
        dm.create_splits()

    def run_augment():
        log.info(f"=== STEP: Augment (factor={args.factor}) ===")
        aug = OfflineAugmentor(cfg)
        train_dir = Path(cfg.paths.processed_data) / "train"
        aug.augment_split(train_dir, factor=args.factor)

    def run_analyze():
        log.info("=== STEP: Analyze ===")
        stats = dm.analyze()
        dm.export_stats(args.output_stats)
        dm.update_dataset_yaml()
        for split, s in stats.items():
            log.info(
                f"  {split}: {s['num_images']} images | "
                f"{s['num_boxes']} boxes | "
                f"avg_boxes={s['avg_boxes_per_image']:.1f}"
            )

    actions = {
        "validate": run_validate,
        "preprocess": run_preprocess,
        "split": run_split,
        "augment": run_augment,
        "analyze": run_analyze,
    }

    if args.action == "all":
        for name, fn in actions.items():
            fn()
    else:
        actions[args.action]()

    log.info("Dataset pipeline complete ✓")


if __name__ == "__main__":
    main()
