#!/usr/bin/env python3
"""SPIRO ML — training/mlops/model_metadata.py
Register model metadata directly into the model registry.

Usage
-----
    python training/mlops/model_metadata.py register \
        --model-id yolov11s --version v1.0.0 \
        --architecture yolo11s --task detection \
        --onnx models/exports/best.onnx \
        --map50-95 0.72 --top1-acc 0.0 \
        --dataset-version-id v1.0.0 --experiment-id exp_001
"""
import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

def main():
    from lib.ml.mlops.registry.model_registry import ModelRegistry
    from lib.ml.mlops.registry.model_metadata import ModelMetadata, ModelMetrics
    from lib.ml.core.logger import get_logger
    log = get_logger("model_metadata")

    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="command", required=True)
    reg = sub.add_parser("register")
    reg.add_argument("--model-id", required=True)
    reg.add_argument("--version", required=True)
    reg.add_argument("--architecture", required=True)
    reg.add_argument("--task", default="detection")
    reg.add_argument("--onnx", default="")
    reg.add_argument("--pytorch", default="")
    reg.add_argument("--map50-95", type=float, default=0.0)
    reg.add_argument("--map50", type=float, default=0.0)
    reg.add_argument("--top1-acc", type=float, default=0.0)
    reg.add_argument("--dataset-version-id", default="")
    reg.add_argument("--experiment-id", default="")
    reg.add_argument("--num-classes", type=int, default=109)
    reg.add_argument("--input-size", type=int, default=640)
    args = p.parse_args()

    registry = ModelRegistry()
    if args.command == "register":
        metrics = ModelMetrics(mAP50_95=args.map50_95, mAP50=args.map50, top1_accuracy=args.top1_acc)
        meta = ModelMetadata(
            model_id=args.model_id, version=args.version,
            architecture=args.architecture, task=args.task,
            onnx_path=args.onnx, pytorch_path=args.pytorch,
            dataset_version_id=args.dataset_version_id,
            experiment_id=args.experiment_id,
            num_classes=args.num_classes, input_size=args.input_size,
            metrics=metrics,
        )
        registry.register(meta)
        log.info(f"Registered: {args.model_id} v{args.version}")

if __name__ == "__main__":
    main()
