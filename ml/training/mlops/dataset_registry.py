#!/usr/bin/env python3
"""SPIRO ML — training/mlops/dataset_registry.py
Manage dataset versions from CLI.

Usage
-----
    python training/mlops/dataset_registry.py list
    python training/mlops/dataset_registry.py snapshot --dataset-id spiro_v1 --root datasets/processed
    python training/mlops/dataset_registry.py show --dataset-id spiro_v1 --version v1.0.0
    python training/mlops/dataset_registry.py export
"""
import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

def main():
    from lib.ml.mlops.registry.dataset_registry import DatasetRegistry
    from lib.ml.core.logger import get_logger
    log = get_logger("dataset_registry")

    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("list")
    snap = sub.add_parser("snapshot")
    snap.add_argument("--dataset-id", required=True)
    snap.add_argument("--root", required=True)
    snap.add_argument("--description", default="")
    snap.add_argument("--parent-version", default=None)
    show = sub.add_parser("show")
    show.add_argument("--dataset-id", required=True)
    show.add_argument("--version", required=True)
    sub.add_parser("export")
    args = p.parse_args()

    registry = DatasetRegistry()
    if args.command == "list":
        for did in registry.list_datasets():
            latest = registry.get_latest(did)
            log.info(f"  {did}: {len(registry.get_all_versions(did))} versions, latest={latest.version if latest else 'N/A'}")
    elif args.command == "snapshot":
        dv = registry.snapshot_from_disk(
            dataset_id=args.dataset_id,
            dataset_root=Path(args.root),
            description=args.description,
            parent_version=args.parent_version,
        )
        registry.register(dv)
        log.info(f"Snapshot: {dv.dataset_id} {dv.version} ({dv.image_count} images)")
    elif args.command == "show":
        dv = registry.get(args.dataset_id, args.version)
        if dv: print(dv.to_json())
        else: log.error("Not found")
    elif args.command == "export":
        p = registry.export_versions_json()
        log.info(f"Exported: {p}")

if __name__ == "__main__":
    main()
