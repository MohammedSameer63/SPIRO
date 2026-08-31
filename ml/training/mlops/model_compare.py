#!/usr/bin/env python3
"""SPIRO ML — training/mlops/model_compare.py
Compare candidate vs production model.

Usage
-----
    python training/mlops/model_compare.py \
        --candidate-id yolov11s --candidate-version v2.0.0 \
        --production-id yolov11s --production-version v1.0.0
"""
import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

def main():
    from lib.ml.mlops.registry.model_registry import ModelRegistry
    from lib.ml.mlops.validation.model_compare import ModelComparator
    from lib.ml.core.logger import get_logger
    log = get_logger("model_compare")

    p = argparse.ArgumentParser()
    p.add_argument("--candidate-id", required=True)
    p.add_argument("--candidate-version", required=True)
    p.add_argument("--production-id", required=True)
    p.add_argument("--production-version", required=True)
    p.add_argument("--primary-metric", default="mAP50_95")
    p.add_argument("--min-improvement", type=float, default=0.001)
    args = p.parse_args()

    registry = ModelRegistry()
    candidate  = registry.get(args.candidate_id, args.candidate_version)
    production = registry.get(args.production_id, args.production_version)
    if not candidate or not production:
        log.error("Model(s) not found in registry")
        sys.exit(1)

    comparator = ModelComparator(
        primary_metric=args.primary_metric,
        min_improvement=args.min_improvement,
    )
    report = comparator.compare(candidate, production)
    print(f"\nRecommendation: {report.recommendation.upper()}")
    for r in report.reasons:
        print(f"  • {r}")

if __name__ == "__main__":
    main()
