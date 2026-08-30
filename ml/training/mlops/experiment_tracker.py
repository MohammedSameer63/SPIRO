#!/usr/bin/env python3
"""SPIRO ML — training/mlops/experiment_tracker.py
Query experiment history.

Usage
-----
    python training/mlops/experiment_tracker.py list
    python training/mlops/experiment_tracker.py list --model-id yolov11s
    python training/mlops/experiment_tracker.py export
"""
import argparse, sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

def main():
    from lib.ml.mlops.registry.experiment_tracker import ExperimentTracker
    from lib.ml.core.logger import get_logger
    log = get_logger("experiment_tracker")
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="command", required=True)
    ls = sub.add_parser("list")
    ls.add_argument("--model-id", default=None)
    sub.add_parser("export")
    args = p.parse_args()

    tracker = ExperimentTracker()
    if args.command == "list":
        records = tracker.list_experiments(model_id=args.model_id)
        for r in records[:20]:
            log.info(f"  {r.experiment_id} [{r.model_id}|{r.status}] {r.started_at}")
    elif args.command == "export":
        out = tracker.export_history_json()
        log.info(f"Exported: {out}")

if __name__ == "__main__":
    main()
