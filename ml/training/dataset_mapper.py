#!/usr/bin/env python3
"""
SPIRO ML — training/dataset_mapper.py
Maps one or all raw datasets into SPIRO YOLO format.

Usage
-----
    # Map all datasets
    python training/dataset_mapper.py --raw-dir datasets/raw

    # Map specific datasets
    python training/dataset_mapper.py --datasets taco trashnet --raw-dir datasets/raw

    # Map and save report
    python training/dataset_mapper.py --all --report reports/mapping_report.json
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Map raw datasets to SPIRO YOLO format")
    p.add_argument("--raw-dir", default="datasets/raw")
    p.add_argument("--mapped-dir", default="datasets/mapped")
    p.add_argument(
        "--datasets", nargs="*",
        choices=["taco", "trashnet", "zerowaste", "openlittermap", "kaggle_gc"],
        default=None,
        help="Datasets to map (default: all available)",
    )
    p.add_argument("--report", default="reports/mapping_report.json")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    from lib.ml.core.logger import get_logger
    from lib.ml.dataset_engineering.mappers.mapper import (
        build_taco_mapper,
        build_trashnet_mapper,
        build_zerowaste_mapper,
        build_olm_mapper,
        build_kaggle_gc_mapper,
    )

    log = get_logger("dataset_mapper")
    raw_dir = Path(args.raw_dir)
    mapped_dir = Path(args.mapped_dir)

    mapper_factories = {
        "taco": lambda: build_taco_mapper(raw_dir, mapped_dir / "TACO"),
        "trashnet": lambda: build_trashnet_mapper(raw_dir, mapped_dir / "TrashNet"),
        "zerowaste": lambda: build_zerowaste_mapper(raw_dir, mapped_dir / "ZeroWaste"),
        "openlittermap": lambda: build_olm_mapper(raw_dir, mapped_dir / "OpenLitterMap"),
        "kaggle_gc": lambda: build_kaggle_gc_mapper(raw_dir, mapped_dir / "KaggleGC"),
    }

    target = args.datasets or list(mapper_factories.keys())
    reports = {}

    for name in target:
        if name not in mapper_factories:
            log.warning(f"Unknown dataset: {name}")
            continue
        # Only attempt if raw data exists
        dataset_raw = raw_dir / name.upper().replace("_", "-")
        fallback = raw_dir / name
        if not dataset_raw.exists() and not fallback.exists():
            log.warning(f"Raw data not found for {name} — skipping (run dataset_download.py first)")
            continue

        log.info(f"Mapping {name}...")
        try:
            mapper = mapper_factories[name]()
            imgs, anns = mapper.map()
            rpt = mapper.report()
            reports[name] = rpt
            log.info(f"  ✓ {name}: {imgs} images, {anns} annotations")
            if rpt.get("unmapped_classes"):
                log.warning(f"  Unmapped source classes: {rpt['unmapped_classes']}")
        except Exception as e:
            log.error(f"  ✗ {name}: {e}")
            reports[name] = {"error": str(e)}

    out = Path(args.report)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(reports, f, indent=2)
    log.info(f"Mapping report → {out}")


if __name__ == "__main__":
    main()
