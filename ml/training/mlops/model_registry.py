#!/usr/bin/env python3
"""SPIRO ML — training/mlops/model_registry.py
Manage model registry from CLI.

Usage
-----
    python training/mlops/model_registry.py list
    python training/mlops/model_registry.py show --model-id yolov11s --version v1.0.0
    python training/mlops/model_registry.py promote --model-id yolov11s --version v1.0.0 --stage production
    python training/mlops/model_registry.py export
"""
import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

def main():
    import json
    from lib.ml.mlops.registry.model_registry import ModelRegistry
    from lib.ml.core.logger import get_logger
    log = get_logger("model_registry")

    p = argparse.ArgumentParser(description="SPIRO Model Registry")
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("list")
    show = sub.add_parser("show")
    show.add_argument("--model-id", required=True)
    show.add_argument("--version", required=True)
    prom = sub.add_parser("promote")
    prom.add_argument("--model-id", required=True)
    prom.add_argument("--version", required=True)
    prom.add_argument("--stage", choices=["staging", "production", "archived"], required=True)
    prom.add_argument("--approved-by", default="cli")
    sub.add_parser("export")
    args = p.parse_args()

    registry = ModelRegistry()
    if args.command == "list":
        s = registry.summary()
        print(json.dumps(s, indent=2))
    elif args.command == "show":
        m = registry.get(args.model_id, args.version)
        if m: print(m.to_json())
        else: log.error(f"Not found: {args.model_id} v{args.version}")
    elif args.command == "promote":
        registry.promote(args.model_id, args.version, args.stage, args.approved_by)
        log.info(f"Promoted {args.model_id} v{args.version} → {args.stage}")
    elif args.command == "export":
        p = registry.export_registry_json()
        log.info(f"Exported: {p}")

if __name__ == "__main__":
    main()
