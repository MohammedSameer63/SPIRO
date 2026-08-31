#!/usr/bin/env python3
"""
SPIRO ML — training/dataset_versioning.py
Manages dataset version snapshots.

Usage
-----
    # Create a version snapshot
    python training/dataset_versioning.py create --processed-dir datasets/processed

    # List all versions
    python training/dataset_versioning.py list

    # Show specific version
    python training/dataset_versioning.py show --version v20240115_143022_abc12345

    # Delete a version entry
    python training/dataset_versioning.py delete --version v20240115_143022_abc12345
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SPIRO Dataset Version Manager")
    sub = p.add_subparsers(dest="command", required=True)

    # create
    create = sub.add_parser("create", help="Create a version snapshot")
    create.add_argument("--processed-dir", default="datasets/processed")
    create.add_argument("--sources", nargs="*",
                        default=["taco", "trashnet", "zerowaste", "openlittermap", "kaggle_gc"])
    create.add_argument("--notes", default="")
    create.add_argument("--versions-dir", default="datasets")

    # list
    ls = sub.add_parser("list", help="List all dataset versions")
    ls.add_argument("--versions-dir", default="datasets")

    # show
    show = sub.add_parser("show", help="Show a specific version")
    show.add_argument("--version", required=True)
    show.add_argument("--versions-dir", default="datasets")

    # delete
    delete = sub.add_parser("delete", help="Delete a version entry")
    delete.add_argument("--version", required=True)
    delete.add_argument("--versions-dir", default="datasets")

    return p.parse_args()


def main() -> None:
    args = parse_args()
    from lib.ml.core.logger import get_logger
    from lib.ml.dataset_engineering.versioning.versioning import DatasetVersioning

    log = get_logger("dataset_versioning")
    dv = DatasetVersioning(Path(args.versions_dir))

    if args.command == "create":
        version_id = dv.create_version(
            processed_dir=Path(args.processed_dir),
            sources=args.sources,
            notes=args.notes,
        )
        dv.save_json(Path("reports/dataset_version.json"))
        log.info(f"Created version: {version_id}")

    elif args.command == "list":
        versions = dv.list_versions()
        if not versions:
            log.info("No versions found")
            return
        log.info(f"{'Version ID':<40} {'Created':<25} {'Images':>8} {'Annotations':>12}")
        log.info("-" * 90)
        for v in versions:
            log.info(
                f"{v['version_id']:<40} "
                f"{v['created_at'][:19]:<25} "
                f"{v.get('total_images', 0):>8,} "
                f"{v.get('total_annotations', 0):>12,}"
            )

    elif args.command == "show":
        try:
            entry = dv.get_version(args.version)
            print(json.dumps(entry, indent=2))
        except KeyError as e:
            log.error(str(e))
            sys.exit(1)

    elif args.command == "delete":
        try:
            dv.delete_version(args.version)
            log.info(f"Deleted version: {args.version}")
        except KeyError as e:
            log.error(str(e))
            sys.exit(1)


if __name__ == "__main__":
    main()
