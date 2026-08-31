#!/usr/bin/env python3
"""
SPIRO ML — training/dataset_download.py
Downloads all supported waste datasets from their canonical sources.

Usage
-----
    python training/dataset_download.py
    python training/dataset_download.py --datasets taco trashnet
    python training/dataset_download.py --datasets taco --raw-dir /data/raw
"""
import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Download SPIRO waste datasets")
    p.add_argument(
        "--datasets", nargs="*",
        choices=["taco", "trashnet", "zerowaste", "openlittermap", "kaggle_gc", "mju_waste"],
        default=None,
        help="Datasets to download (default: all)",
    )
    p.add_argument("--raw-dir", default="datasets/raw", help="Root download directory")
    p.add_argument("--skip", nargs="*", default=[], help="Datasets to skip")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    from lib.ml.core.logger import get_logger
    from lib.ml.dataset_engineering.downloaders.base import DOWNLOADERS, download_all

    log = get_logger("dataset_download")
    raw_dir = Path(args.raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)

    if args.datasets:
        subset = {k: v for k, v in DOWNLOADERS.items() if k in args.datasets}
        skip = args.skip
    else:
        subset = DOWNLOADERS
        skip = args.skip

    log.info(f"Downloading {list(subset.keys())} → {raw_dir}")
    results = {}
    for name, cls in subset.items():
        if name in skip:
            log.info(f"Skipping {name}")
            continue
        try:
            dl = cls(raw_dir=raw_dir)
            path = dl.download()
            results[name] = str(path)
            log.info(f"✓ {name} → {path}")
        except Exception as e:
            log.error(f"✗ {name}: {e}")

    log.info(f"Download complete: {len(results)}/{len(subset)} succeeded")


if __name__ == "__main__":
    main()
