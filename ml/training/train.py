#!/usr/bin/env python3
"""
SPIRO ML — training/train.py
Primary entry point for YOLOv11 training.

Usage
-----
    # Train with default config
    python training/train.py

    # Train specific variant
    python training/train.py --config configs/training/yolo11s.yaml

    # Train with overrides
    python training/train.py --config configs/training/yolo11n.yaml \\
        --set training.epochs=100 training.batch_size=32 model.weights=yolo11n.pt

    # Resume interrupted training
    python training/train.py --config configs/training/yolo11s.yaml --resume

    # Resume from specific checkpoint
    python training/train.py --config configs/training/yolo11s.yaml \\
        --resume --checkpoint models/checkpoints/spiro_yolo11s/weights/last.pt

    # Multi-GPU (DDP)
    python training/train.py --config configs/training/yolo11m.yaml \\
        --set training.device=0,1

    # Validate only (no training)
    python training/train.py --config configs/training/yolo11s.yaml \\
        --validate-only --weights models/checkpoints/spiro_yolo11s/weights/best.pt
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="SPIRO YOLOv11 Training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--config",
        default="configs/training/yolov11_training.yaml",
        help="Training config YAML path",
    )
    p.add_argument(
        "--set", nargs="*", default=[],
        metavar="KEY=VALUE",
        help="Override config values: --set training.epochs=50 model.weights=yolo11s.pt",
    )
    p.add_argument(
        "--resume", action="store_true",
        help="Resume training from last checkpoint",
    )
    p.add_argument(
        "--checkpoint", type=str, default=None,
        help="Explicit checkpoint to resume from",
    )
    p.add_argument(
        "--validate-only", action="store_true",
        help="Skip training; run validation only",
    )
    p.add_argument(
        "--weights", type=str, default=None,
        help="Weights for --validate-only mode",
    )
    p.add_argument(
        "--split", choices=["val", "test"], default="val",
        help="Dataset split for --validate-only",
    )
    p.add_argument(
        "--export-onnx", action="store_true",
        help="Export best model to ONNX after training",
    )
    return p.parse_args()


def _parse_overrides(set_args: list) -> dict:
    """Parse --set KEY=VALUE pairs into a flat dict with type inference."""
    overrides = {}
    for item in set_args:
        if "=" not in item:
            print(f"[WARNING] Ignoring malformed override (no '='): {item}")
            continue
        key, value = item.split("=", 1)
        # Type inference
        try:
            value = int(value)
        except ValueError:
            try:
                value = float(value)
            except ValueError:
                if value.lower() in ("true", "false"):
                    value = value.lower() == "true"
                elif value.lower() == "null":
                    value = None
        overrides[key] = value
    return overrides


def main() -> None:
    args = parse_args()

    from lib.ml.training.training_config import TrainingConfig
    from lib.ml.training.trainer import SPIROYOLOTrainer
    from lib.ml.core.logger import get_logger

    log = get_logger("train")

    # Build overrides
    overrides = _parse_overrides(args.set)
    if args.resume:
        overrides["training.resume"] = True
    if args.checkpoint:
        overrides["training.resume_checkpoint"] = args.checkpoint
    if args.export_onnx:
        overrides["export.auto_export_onnx"] = True

    # Load config
    log.info(f"Loading config: {args.config}")
    try:
        config = TrainingConfig.load(args.config, overrides=overrides)
    except Exception as e:
        log.error(f"Config load failed: {e}")
        sys.exit(1)

    # Log key settings
    log.info(f"Model:    {config.model.weights}")
    log.info(f"Classes:  {config.model.num_classes}")
    log.info(f"Epochs:   {config.training.epochs}")
    log.info(f"Batch:    {config.training.batch_size}")
    log.info(f"Dataset:  {config.dataset.yaml}")

    trainer = SPIROYOLOTrainer(config)

    if args.validate_only:
        weights = args.weights or str(
            trainer.checkpoint_mgr.best_path() or config.model.weights
        )
        log.info(f"Validate-only mode: {weights} on {args.split}")
        metrics = trainer.validate(weights=Path(weights), split=args.split)
        log.info("Validation results:")
        for k, v in metrics.items():
            if not isinstance(v, dict):
                log.info(f"  {k}: {v}")
        return

    # Train
    summary = trainer.train()
    log.info("Training summary:")
    for k, v in summary.items():
        if not isinstance(v, dict):
            log.info(f"  {k}: {v}")


if __name__ == "__main__":
    main()
