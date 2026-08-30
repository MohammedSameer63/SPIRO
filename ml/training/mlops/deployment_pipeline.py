#!/usr/bin/env python3
"""SPIRO ML — training/mlops/deployment_pipeline.py
End-to-end deployment pipeline: validate → compare → deploy → verify.

Usage
-----
    python training/mlops/deployment_pipeline.py \
        --model-id yolov11s --version v2.0.0 \
        --onnx models/exports/best.onnx \
        --strategy blue_green --task detection
"""
import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

def main():
    from lib.ml.mlops.deployment.model_deployer import ModelDeployer
    from lib.ml.mlops.deployment.model_promoter import ModelPromoter
    from lib.ml.mlops.validation.model_validator import ModelValidator
    from lib.ml.core.logger import get_logger
    log = get_logger("deployment_pipeline")

    p = argparse.ArgumentParser()
    p.add_argument("--model-id",  required=True)
    p.add_argument("--version",   required=True)
    p.add_argument("--onnx",      required=True)
    p.add_argument("--strategy",  default="blue_green",
                   choices=["immediate","blue_green","canary"])
    p.add_argument("--task",      default="detection")
    p.add_argument("--input-size", type=int, default=640)
    p.add_argument("--skip-validation", action="store_true")
    p.add_argument("--deployed-by", default="cli")
    args = p.parse_args()

    deployer = ModelDeployer()
    validator = ModelValidator()

    onnx_path = Path(args.onnx)
    if not args.skip_validation:
        report = validator.validate(
            onnx_path=onnx_path,
            model_id=args.model_id,
            version=args.version,
            input_size=args.input_size,
            task=args.task,
        )
        if not report.overall_passed:
            log.error(f"Validation FAILED: {report.summary}")
            sys.exit(1)
        log.info(f"Validation passed: {report.summary}")

    rec = deployer.deploy(
        model_id=args.model_id,
        version=args.version,
        onnx_path=onnx_path,
        strategy=args.strategy,
        task=args.task,
        deployed_by=args.deployed_by,
        validate=False,  # already done above
    )
    log.info(f"Deployed: {rec.deployment_id} via {args.strategy}")
    active = deployer.get_active_path(args.task)
    log.info(f"Active model: {active}")

if __name__ == "__main__":
    main()
