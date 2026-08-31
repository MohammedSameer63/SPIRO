#!/usr/bin/env python3
"""SPIRO ML — training/mlops/model_validator.py
Validate an ONNX model before deployment.

Usage
-----
    python training/mlops/model_validator.py \
        --onnx models/exports/spiro_yolov11s_best.onnx \
        --model-id yolov11s --version v1.0.0 \
        --task detection --input-size 640
"""
import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

def main():
    from lib.ml.mlops.validation.model_validator import ModelValidator
    from lib.ml.core.logger import get_logger
    log = get_logger("model_validator")

    p = argparse.ArgumentParser(description="Validate SPIRO ONNX model")
    p.add_argument("--onnx", required=True)
    p.add_argument("--model-id", required=True)
    p.add_argument("--version", required=True)
    p.add_argument("--task", default="detection", choices=["detection", "verification"])
    p.add_argument("--input-size", type=int, default=640)
    p.add_argument("--num-classes", type=int, default=109)
    p.add_argument("--min-accuracy", type=float, default=0.0)
    p.add_argument("--max-latency-ms", type=float, default=500.0)
    p.add_argument("--accuracy", type=float, default=None)
    args = p.parse_args()

    validator = ModelValidator(
        min_accuracy=args.min_accuracy,
        max_latency_ms=args.max_latency_ms,
    )
    report = validator.validate(
        onnx_path=args.onnx,
        model_id=args.model_id,
        version=args.version,
        input_size=args.input_size,
        num_classes=args.num_classes,
        task=args.task,
        accuracy=args.accuracy,
    )
    out = validator.save_report(report)
    print(f"\n{'PASSED' if report.overall_passed else 'FAILED'}: {report.summary}")
    print(f"Report: {out}")
    sys.exit(0 if report.overall_passed else 1)

if __name__ == "__main__":
    main()
