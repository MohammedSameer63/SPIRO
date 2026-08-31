#!/usr/bin/env python3
"""
SPIRO ML — training/quality_checker.py
Comprehensive quality check for a SPIRO dataset. Combines:
  - Corruption check
  - Resolution check
  - Label coverage
  - Class imbalance analysis
  - Bounding box sanity
  - Rare class flagging

Usage
-----
    python training/quality_checker.py --dir datasets/merged
    python training/quality_checker.py --dir datasets/processed --charts --report reports/qc.json
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Quality check a SPIRO dataset")
    p.add_argument("--dir", required=True, help="Dataset root (images/ + labels/)")
    p.add_argument("--report", default="reports/quality_report.json")
    p.add_argument("--charts", action="store_true", help="Generate quality charts")
    p.add_argument("--charts-dir", default="reports/charts/quality")
    p.add_argument("--min-width", type=int, default=64)
    p.add_argument("--min-height", type=int, default=64)
    p.add_argument("--num-classes", type=int, default=109)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    from lib.ml.core.logger import get_logger
    from lib.ml.dataset_engineering.cleaners.cleaner import DataCleaner
    from lib.ml.dataset_engineering.stats.statistics import DatasetStatistics
    from lib.ml.dataset_engineering.visualization.charts import DatasetVisualizer

    log = get_logger("quality_checker")
    ds_dir = Path(args.dir)

    log.info(f"Quality checking {ds_dir}")

    # Cleaning analysis (dry run)
    cleaner = DataCleaner(
        dataset_dir=ds_dir,
        min_resolution=(args.min_width, args.min_height),
        num_classes=args.num_classes,
    )
    clean_report = cleaner.clean(remove=False)

    # Statistics
    stats = DatasetStatistics(ds_dir, name=ds_dir.name)
    stat_report = stats.compute()

    # Compute duplicate percentage
    total = clean_report["total_images"]
    dup_count = sum(len(g) - 1 for g in clean_report["exact_duplicates"])
    near_dup_count = sum(len(g) - 1 for g in clean_report["near_duplicates"])

    quality_report = {
        "dataset": str(ds_dir),
        "total_images": total,
        "corrupt_images": len(clean_report["corrupt_images"]),
        "corrupt_pct": round(100 * len(clean_report["corrupt_images"]) / max(total, 1), 2),
        "exact_duplicate_pct": round(100 * dup_count / max(total, 1), 2),
        "near_duplicate_pct": round(100 * near_dup_count / max(total, 1), 2),
        "low_resolution_count": len(clean_report["low_resolution"]),
        "missing_label_count": len(clean_report["missing_labels"]),
        "orphan_label_count": len(clean_report["orphan_labels"]),
        "invalid_bbox_count": len(clean_report["invalid_bbox_images"]),
        "invalid_class_count": len(clean_report["invalid_class_images"]),
        "total_annotations": stat_report["total_annotations"],
        "represented_classes": stat_report["represented_classes"],
        "missing_classes": stat_report["missing_classes"],
        "rare_classes": stat_report["rare_classes"],
        "imbalance_ratio": stat_report["imbalance_ratio"],
        "group_coverage": stat_report["group_coverage"],
        "class_distribution": stat_report["class_distribution"],
    }

    # Quality score (simple heuristic)
    score = 100.0
    score -= quality_report["corrupt_pct"] * 5
    score -= quality_report["exact_duplicate_pct"] * 2
    score -= quality_report["near_duplicate_pct"] * 1
    score -= min(len(quality_report["missing_classes"]) * 0.5, 20)
    score -= min(len(quality_report["rare_classes"]) * 0.2, 10)
    score -= max(0, (quality_report["imbalance_ratio"] - 10) * 0.1)
    quality_report["quality_score"] = round(max(0.0, score), 1)

    # Print summary
    log.info(f"Quality Score: {quality_report['quality_score']}/100")
    log.info(f"  Total images:        {total:,}")
    log.info(f"  Corrupt:             {quality_report['corrupt_pct']}%")
    log.info(f"  Exact dups:          {quality_report['exact_duplicate_pct']}%")
    log.info(f"  Near dups:           {quality_report['near_duplicate_pct']}%")
    log.info(f"  Missing labels:      {quality_report['missing_label_count']}")
    log.info(f"  Classes covered:     {quality_report['represented_classes']}/109")
    log.info(f"  Imbalance ratio:     {quality_report['imbalance_ratio']}x")
    log.info(f"  Rare classes:        {len(quality_report['rare_classes'])}")

    out = Path(args.report)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(quality_report, f, indent=2)
    log.info(f"Quality report → {out}")

    if args.charts:
        viz = DatasetVisualizer(stat_report, Path(args.charts_dir))
        viz.generate_all()
        log.info(f"Charts → {args.charts_dir}")


if __name__ == "__main__":
    main()
