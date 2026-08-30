#!/usr/bin/env python3
"""
SPIRO ML — training/dataset_cleaner.py
Runs the full cleaning pipeline on a mapped dataset directory.

Usage
-----
    # Clean all datasets in datasets/mapped/
    python training/dataset_cleaner.py --mapped-dir datasets/mapped

    # Clean a single dataset
    python training/dataset_cleaner.py --dir datasets/mapped/TACO

    # Dry run (report only, no deletion)
    python training/dataset_cleaner.py --mapped-dir datasets/mapped --dry-run

    # Write per-dataset reports
    python training/dataset_cleaner.py --mapped-dir datasets/mapped --reports-dir reports/cleaning
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Clean mapped SPIRO datasets")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--mapped-dir", help="Root of all mapped datasets (process all subdirs)")
    g.add_argument("--dir", help="Single dataset directory to clean")
    p.add_argument("--reports-dir", default="reports/cleaning")
    p.add_argument("--dry-run", action="store_true", help="Report only, do not remove files")
    p.add_argument("--min-width", type=int, default=64)
    p.add_argument("--min-height", type=int, default=64)
    p.add_argument("--phash-threshold", type=int, default=8,
                   help="Hamming distance threshold for near-duplicate detection")
    p.add_argument("--min-bbox-area", type=float, default=0.0004,
                   help="Minimum bbox area as fraction of image")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    from lib.ml.core.logger import get_logger
    from lib.ml.dataset_engineering.cleaners.cleaner import DataCleaner

    log = get_logger("dataset_cleaner")
    reports_dir = Path(args.reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)

    if args.dir:
        dataset_dirs = [Path(args.dir)]
    else:
        mapped_root = Path(args.mapped_dir)
        dataset_dirs = [
            d for d in sorted(mapped_root.iterdir())
            if d.is_dir() and (d / "images").exists()
        ]

    log.info(f"Cleaning {len(dataset_dirs)} dataset(s) (dry_run={args.dry_run})")

    for ds_dir in dataset_dirs:
        log.info(f"  Cleaning {ds_dir.name}...")
        try:
            cleaner = DataCleaner(
                dataset_dir=ds_dir,
                min_resolution=(args.min_width, args.min_height),
                phash_threshold=args.phash_threshold,
                min_bbox_area=args.min_bbox_area,
            )
            report = cleaner.clean(remove=not args.dry_run)
            out = reports_dir / f"{ds_dir.name}_cleaning_report.json"
            cleaner.save_report(out)

            log.info(
                f"  ✓ {ds_dir.name}: "
                f"kept={report['clean_images']} | "
                f"corrupt={len(report['corrupt_images'])} | "
                f"exact_dups={len(report['exact_duplicates'])} | "
                f"near_dups={len(report['near_duplicates'])} | "
                f"low_res={len(report['low_resolution'])}"
            )
        except Exception as e:
            log.error(f"  ✗ {ds_dir.name}: {e}")

    log.info(f"Cleaning complete. Reports → {reports_dir}")


if __name__ == "__main__":
    main()
