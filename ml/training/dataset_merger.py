#!/usr/bin/env python3
"""
SPIRO ML — training/dataset_merger.py
Merges all mapped + cleaned datasets into a single unified SPIRO dataset.

Usage
-----
    python training/dataset_merger.py
    python training/dataset_merger.py --mapped-dir datasets/mapped --merged-dir datasets/merged
    python training/dataset_merger.py --no-dedup  # skip cross-source deduplication
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Merge SPIRO mapped datasets")
    p.add_argument("--mapped-dir", default="datasets/mapped")
    p.add_argument("--merged-dir", default="datasets/merged")
    p.add_argument("--no-dedup", action="store_true",
                   help="Disable cross-source deduplication")
    p.add_argument("--report", default="reports/merge_report.json")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    from lib.ml.core.logger import get_logger
    from lib.ml.dataset_engineering.cleaners.merger import DatasetMerger

    log = get_logger("dataset_merger")
    mapped_dir = Path(args.mapped_dir)
    merged_dir = Path(args.merged_dir)

    source_dirs = [
        d for d in sorted(mapped_dir.iterdir())
        if d.is_dir() and (d / "images").exists()
    ]

    if not source_dirs:
        log.error(f"No mapped dataset dirs found in {mapped_dir}")
        log.error("Run dataset_mapper.py first.")
        sys.exit(1)

    log.info(f"Merging {len(source_dirs)} datasets → {merged_dir}")
    for d in source_dirs:
        n_imgs = len(list((d / "images").iterdir()))
        log.info(f"  {d.name}: {n_imgs} images")

    merger = DatasetMerger(
        source_dirs=source_dirs,
        output_dir=merged_dir,
        dedup_across_sources=not args.no_dedup,
    )
    report = merger.merge()
    merger.save_report(Path(args.report))

    log.info(
        f"Merge complete — "
        f"{report['total_images']} images | "
        f"{report['total_annotations']} annotations | "
        f"{report['cross_source_duplicates_removed']} cross-source dups removed"
    )


if __name__ == "__main__":
    main()
