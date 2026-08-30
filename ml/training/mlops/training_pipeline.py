#!/usr/bin/env python3
"""SPIRO ML — training/mlops/training_pipeline.py
Full MLOps-integrated training pipeline:
train → register → validate → compare → promote.

Usage
-----
    python training/mlops/training_pipeline.py \
        --model-id yolov11s --version v2.0.0 \
        --config configs/training/yolo11s.yaml \
        --dataset-version v1.0.0 \
        --onnx models/exports/best.onnx \
        --auto-promote
"""
import argparse, json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

def main():
    from lib.ml.mlops import (
        ModelRegistry, ExperimentTracker, ModelMetadata, ModelMetrics,
        ModelValidator, ModelComparator, ModelPromoter,
    )
    from lib.ml.core.logger import get_logger
    log = get_logger("training_pipeline")

    p = argparse.ArgumentParser(description="Full MLOps training pipeline")
    p.add_argument("--model-id",          required=True)
    p.add_argument("--version",           required=True)
    p.add_argument("--config",            required=True)
    p.add_argument("--dataset-version",   default="")
    p.add_argument("--onnx",              default="")
    p.add_argument("--pytorch",           default="")
    p.add_argument("--map50-95",          type=float, default=0.0)
    p.add_argument("--map50",             type=float, default=0.0)
    p.add_argument("--top1-acc",          type=float, default=0.0)
    p.add_argument("--architecture",      default="yolo11s")
    p.add_argument("--task",              default="detection")
    p.add_argument("--input-size",        type=int, default=640)
    p.add_argument("--epochs",            type=int, default=0)
    p.add_argument("--training-time-s",   type=float, default=0.0)
    p.add_argument("--auto-promote",      action="store_true")
    p.add_argument("--primary-metric",    default="mAP50_95")
    p.add_argument("--min-improvement",   type=float, default=0.001)
    p.add_argument("--strategy",          default="blue_green")
    args = p.parse_args()

    registry   = ModelRegistry()
    tracker    = ExperimentTracker()

    # 1. Start experiment tracking
    exp_id = tracker.start_run(
        experiment_name=f"{args.model_id}_training",
        model_id=args.model_id,
        model_version=args.version,
        dataset_version_id=args.dataset_version,
        hyperparameters={"config": args.config, "epochs": args.epochs},
    )
    log.info(f"Experiment started: {exp_id}")

    # 2. Register model
    metrics = ModelMetrics(
        mAP50_95=args.map50_95,
        mAP50=args.map50,
        top1_accuracy=args.top1_acc,
    )
    meta = ModelMetadata(
        model_id=args.model_id,
        version=args.version,
        architecture=args.architecture,
        task=args.task,
        onnx_path=args.onnx,
        pytorch_path=args.pytorch,
        dataset_version_id=args.dataset_version,
        experiment_id=exp_id,
        num_classes=109,
        input_size=args.input_size,
        training_epochs=args.epochs,
        training_time_s=args.training_time_s,
        metrics=metrics,
    )
    registry.register(meta)
    log.info(f"Registered: {args.model_id} v{args.version}")

    tracker.log_metrics(exp_id, metrics.to_dict())
    tracker.log_artefact(exp_id, onnx_path=args.onnx, pytorch_path=args.pytorch)

    # 3. Validate + promote
    promoter = ModelPromoter(
        registry=registry,
        auto_deploy=args.auto_promote,
        primary_metric=args.primary_metric,
        min_improvement=args.min_improvement,
        deployment_strategy=args.strategy,
    )
    decision = promoter.evaluate_and_promote(
        model_id=args.model_id,
        candidate_version=args.version,
        task=args.task,
        deployed_by="training_pipeline",
    )

    tracker.end_run(exp_id, status="completed",
                    final_metrics={"promoted": float(decision.approved)})

    # 4. Summary
    print(f"\n{'='*50}")
    print(f"  Training Pipeline Summary")
    print(f"{'='*50}")
    print(f"  Model:      {args.model_id} v{args.version}")
    print(f"  mAP50-95:   {args.map50_95:.4f}")
    print(f"  Promotion:  {'APPROVED' if decision.approved else 'REJECTED'}")
    print(f"  Deployed:   {decision.deployed}")
    for r in decision.reasons:
        print(f"    • {r}")
    print(f"{'='*50}")

if __name__ == "__main__":
    main()
