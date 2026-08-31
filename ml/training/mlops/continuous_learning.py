#!/usr/bin/env python3
"""SPIRO ML — training/mlops/continuous_learning.py
Run continuous learning cycles.

Usage
-----
    python training/mlops/continuous_learning.py collect --source path/to/new_images/
    python training/mlops/continuous_learning.py validate
    python training/mlops/continuous_learning.py run-cycle --force
    python training/mlops/continuous_learning.py report
"""
import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

def main():
    from lib.ml.mlops.continuous_learning.continuous_learning import ContinuousLearningManager
    from lib.ml.core.logger import get_logger
    log = get_logger("continuous_learning")

    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="command", required=True)
    col = sub.add_parser("collect")
    col.add_argument("--source", required=True)
    col.add_argument("--model-id", default="yolov11s")
    sub.add_parser("validate")
    rc = sub.add_parser("run-cycle")
    rc.add_argument("--model-id", default="yolov11s")
    rc.add_argument("--force", action="store_true")
    rc.add_argument("--auto-promote", action="store_true")
    sub.add_parser("report")
    args = p.parse_args()

    if args.command == "collect":
        clm = ContinuousLearningManager(model_id="yolov11s")
        sids = clm.collect_from_directory(Path(args.source))
        log.info(f"Collected {len(sids)} samples")
    elif args.command == "validate":
        clm = ContinuousLearningManager()
        counts = clm.validate_staged_samples()
        log.info(f"Approved: {counts['approved']}, Rejected: {counts['rejected']}")
    elif args.command == "run-cycle":
        clm = ContinuousLearningManager(model_id=args.model_id, auto_promote=args.auto_promote)
        cycle = clm.run_cycle(force=args.force)
        log.info(f"Cycle {cycle.cycle_id}: {cycle.status}")
    elif args.command == "report":
        clm = ContinuousLearningManager()
        out = clm.export_cl_report()
        log.info(f"Report: {out}")

if __name__ == "__main__":
    main()
