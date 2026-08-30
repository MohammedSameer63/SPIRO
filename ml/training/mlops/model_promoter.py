#!/usr/bin/env python3
"""SPIRO ML — training/mlops/model_promoter.py
Automated model promotion pipeline.

Usage
-----
    python training/mlops/model_promoter.py \
        --model-id yolov11s --version v2.0.0 \
        --task detection --auto-deploy
"""
import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

def main():
    from lib.ml.mlops.deployment.model_promoter import ModelPromoter
    from lib.ml.core.logger import get_logger
    log = get_logger("model_promoter")

    p = argparse.ArgumentParser()
    p.add_argument("--model-id", required=True)
    p.add_argument("--version", required=True)
    p.add_argument("--task", default="detection")
    p.add_argument("--auto-deploy", action="store_true")
    p.add_argument("--strategy", default="blue_green",
                   choices=["immediate","blue_green","canary"])
    p.add_argument("--deployed-by", default="cli")
    p.add_argument("--primary-metric", default="mAP50_95")
    p.add_argument("--min-improvement", type=float, default=0.001)
    args = p.parse_args()

    promoter = ModelPromoter(
        auto_deploy=args.auto_deploy,
        primary_metric=args.primary_metric,
        min_improvement=args.min_improvement,
        deployment_strategy=args.strategy,
    )
    decision = promoter.evaluate_and_promote(
        model_id=args.model_id,
        candidate_version=args.version,
        task=args.task,
        deployed_by=args.deployed_by,
    )
    status = "APPROVED+DEPLOYED" if decision.deployed else ("APPROVED" if decision.approved else "REJECTED")
    print(f"\n{status}: {args.model_id} v{args.version}")
    for r in decision.reasons:
        print(f"  • {r}")
    sys.exit(0 if decision.approved else 1)

if __name__ == "__main__":
    main()
