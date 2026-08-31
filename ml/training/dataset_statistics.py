#!/usr/bin/env python3
"""
SPIRO ML — training/dataset_statistics.py
Computes and reports statistics for any YOLO-format dataset.

Usage
-----
    python training/dataset_statistics.py --dir datasets/merged
    python training/dataset_statistics.py --dir datasets/processed/train --name "Train Split"
    python training/dataset_statistics.py --dir datasets/merged --charts --report reports/stats.json
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compute SPIRO dataset statistics")
    p.add_argument("--dir", required=True, help="Dataset root (images/ + labels/)")
    p.add_argument("--name", default=None, help="Human-readable dataset name")
    p.add_argument("--report", default="reports/dataset_statistics.json")
    p.add_argument("--markdown", default="reports/dataset_report.md")
    p.add_argument("--charts", action="store_true", help="Generate visualization charts")
    p.add_argument("--charts-dir", default="reports/charts")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    from lib.ml.core.logger import get_logger
    from lib.ml.dataset_engineering.stats.statistics import DatasetStatistics
    from lib.ml.dataset_engineering.visualization.charts import DatasetVisualizer

    log = get_logger("dataset_statistics")
    dataset_dir = Path(args.dir)
    name = args.name or dataset_dir.name

    log.info(f"Computing statistics for: {dataset_dir} ({name})")

    stats = DatasetStatistics(dataset_dir, name=name)
    report = stats.compute()
    stats.save_json(Path(args.report))
    stats.generate_markdown_report(Path(args.markdown))

    # Print summary
    log.info(f"  Total images:       {report['total_images']:,}")
    log.info(f"  Total annotations:  {report['total_annotations']:,}")
    log.info(f"  Classes represented:{report['represented_classes']}/109")
    log.info(f"  Imbalance ratio:    {report['imbalance_ratio']}x")
    log.info(f"  Rare classes:       {len(report['rare_classes'])}")
    log.info(f"  Missing classes:    {len(report['missing_classes'])}")

    if report["rare_classes"]:
        log.warning(f"  Rare: {report['rare_classes'][:10]}")
    if report["missing_classes"]:
        log.info(f"  Missing: {report['missing_classes'][:10]}")

    if args.charts:
        log.info(f"Generating charts → {args.charts_dir}")
        viz = DatasetVisualizer(report, Path(args.charts_dir))
        viz.generate_all()

    log.info(f"Report → {args.report}")
    log.info(f"Markdown → {args.markdown}")


if __name__ == "__main__":
    main()
