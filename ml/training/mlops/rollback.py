#!/usr/bin/env python3
"""SPIRO ML — training/mlops/rollback.py
Emergency rollback shortcut.

Usage
-----
    python training/mlops/rollback.py --task detection
    python training/mlops/rollback.py --task verification --reason "accuracy drop"
"""
import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

def main():
    from lib.ml.mlops.deployment.model_deployer import ModelDeployer
    from lib.ml.core.logger import get_logger
    log = get_logger("rollback")
    p = argparse.ArgumentParser()
    p.add_argument("--task", default="detection")
    p.add_argument("--reason", default="emergency rollback")
    args = p.parse_args()
    deployer = ModelDeployer()
    rec = deployer.rollback(task=args.task, reason=args.reason, deployed_by="cli")
    if rec:
        log.info(f"Rolled back {args.task} to {rec.version}")
    else:
        log.warning("Rollback: no previous deployment found")
        sys.exit(1)

if __name__ == "__main__":
    main()
