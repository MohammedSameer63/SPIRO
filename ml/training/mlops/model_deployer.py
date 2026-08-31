#!/usr/bin/env python3
"""SPIRO ML — training/mlops/model_deployer.py
Deploy / rollback models from CLI.

Usage
-----
    python training/mlops/model_deployer.py deploy \
        --model-id yolov11s --version v2.0.0 \
        --onnx models/exports/best.onnx --strategy blue_green

    python training/mlops/model_deployer.py rollback --task detection

    python training/mlops/model_deployer.py history
    python training/mlops/model_deployer.py active --task detection
"""
import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

def main():
    from lib.ml.mlops.deployment.model_deployer import ModelDeployer
    from lib.ml.core.logger import get_logger
    log = get_logger("model_deployer")

    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="command", required=True)
    dep = sub.add_parser("deploy")
    dep.add_argument("--model-id", required=True)
    dep.add_argument("--version", required=True)
    dep.add_argument("--onnx", required=True)
    dep.add_argument("--strategy", default="immediate",
                     choices=["immediate","blue_green","canary"])
    dep.add_argument("--task", default="detection")
    dep.add_argument("--no-validate", action="store_true")
    dep.add_argument("--canary-pct", type=int, default=100)

    rb = sub.add_parser("rollback")
    rb.add_argument("--task", default="detection")
    rb.add_argument("--reason", default="manual rollback")

    sub.add_parser("history")
    act = sub.add_parser("active")
    act.add_argument("--task", default="detection")
    args = p.parse_args()

    deployer = ModelDeployer()
    if args.command == "deploy":
        rec = deployer.deploy(
            model_id=args.model_id, version=args.version,
            onnx_path=Path(args.onnx), strategy=args.strategy,
            task=args.task, validate=not args.no_validate,
            canary_percentage=args.canary_pct,
        )
        log.info(f"Deployed: {rec.deployment_id}")
    elif args.command == "rollback":
        rec = deployer.rollback(task=args.task, reason=args.reason)
        if rec: log.info(f"Rolled back to: {rec.version}")
        else: log.warning("Nothing to roll back to")
    elif args.command == "history":
        deployer.export_history_json()
        log.info("History exported")
    elif args.command == "active":
        p = deployer.get_active_path(args.task)
        print(str(p) if p else "No active model")

if __name__ == "__main__":
    main()
