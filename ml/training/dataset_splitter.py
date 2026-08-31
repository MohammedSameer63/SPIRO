#!/usr/bin/env python3
"""
SPIRO ML — training/dataset_splitter.py
Creates stratified train/val/test splits from the merged dataset.

Usage
-----
    python training/dataset_splitter.py
    python training/dataset_splitter.py --train 0.75 --val 0.15 --test 0.10
    python training/dataset_splitter.py --merged-dir datasets/merged --out datasets/processed
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Split SPIRO merged dataset")
    p.add_argument("--merged-dir", default="datasets/merged")
    p.add_argument("--out", default="datasets/processed")
    p.add_argument("--train", type=float, default=0.70)
    p.add_argument("--val", type=float, default=0.15)
    p.add_argument("--test", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--min-class-samples", type=int, default=20,
                   help="Flag classes with fewer samples as rare")
    p.add_argument("--report", default="reports/split_report.json")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    from lib.ml.core.logger import get_logger
    from lib.ml.dataset_engineering.splitters.splitter import DatasetSplitter

    log = get_logger("dataset_splitter")

    total = args.train + args.val + args.test
    if abs(total - 1.0) > 0.001:
        log.error(f"Split ratios must sum to 1.0, got {total:.3f}")
        sys.exit(1)

    splitter = DatasetSplitter(
        merged_dir=Path(args.merged_dir),
        output_dir=Path(args.out),
        train_ratio=args.train,
        val_ratio=args.val,
        test_ratio=args.test,
        seed=args.seed,
        min_class_samples=args.min_class_samples,
    )
    report = splitter.split()
    splitter.save_report(Path(args.report))

    log.info(
        f"Split complete:\n"
        f"  train: {report['splits']['train']}\n"
        f"  val:   {report['splits']['val']}\n"
        f"  test:  {report['splits']['test']}"
    )
    if report["rare_classes"]:
        log.warning(f"Rare classes ({len(report['rare_classes'])}): {report['rare_classes'][:10]}")


if __name__ == "__main__":
    main()
