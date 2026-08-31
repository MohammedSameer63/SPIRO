#!/usr/bin/env python3
"""
SPIRO ML — training/trainer.py
Alternative trainer CLI with variant shortcuts and status display.

Usage
-----
    # Quick start with variant shortcut
    python training/trainer.py --variant n
    python training/trainer.py --variant s --epochs 150
    python training/trainer.py --variant m --batch 8 --device 0

    # Full config path
    python training/trainer.py --config configs/training/yolo11l.yaml

    # Resume last run for a variant
    python training/trainer.py --variant s --resume

    # List available variants
    python training/trainer.py --list-variants
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

VARIANT_MAP = {
    "n": "configs/training/yolo11n.yaml",
    "s": "configs/training/yolo11s.yaml",
    "m": "configs/training/yolo11m.yaml",
    "l": "configs/training/yolo11l.yaml",
    "x": "configs/training/yolo11x.yaml",
    "nano":   "configs/training/yolo11n.yaml",
    "small":  "configs/training/yolo11s.yaml",
    "medium": "configs/training/yolo11m.yaml",
    "large":  "configs/training/yolo11l.yaml",
    "xlarge": "configs/training/yolo11x.yaml",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SPIRO YOLOv11 Trainer")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--variant", choices=list(VARIANT_MAP.keys()),
                   help="YOLOv11 variant shortcut (n/s/m/l/x)")
    g.add_argument("--config", type=str,
                   help="Full path to training config YAML")
    p.add_argument("--list-variants", action="store_true",
                   help="Show available variants and exit")
    # Common overrides
    p.add_argument("--epochs",  type=int,   default=None)
    p.add_argument("--batch",   type=int,   default=None)
    p.add_argument("--lr",      type=float, default=None, help="Initial learning rate")
    p.add_argument("--device",  type=str,   default=None,
                   help="Device: auto | cpu | 0 | 0,1 | mps")
    p.add_argument("--resume",  action="store_true")
    p.add_argument("--checkpoint", type=str, default=None)
    p.add_argument("--name",    type=str,   default=None,
                   help="Override experiment name")
    p.add_argument("--no-aug",  action="store_true",
                   help="Disable augmentation (useful for debugging)")
    p.add_argument("--workers", type=int,   default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.list_variants:
        print("\nAvailable SPIRO YOLOv11 variants:")
        print("-" * 45)
        info = {
            "n": ("yolo11n.pt", "~2.6M params", "Fastest, edge deployment"),
            "s": ("yolo11s.pt", "~9.4M params", "Balanced speed/accuracy"),
            "m": ("yolo11m.pt", "~20.1M params", "Good accuracy"),
            "l": ("yolo11l.pt", "~25.3M params", "High accuracy"),
            "x": ("yolo11x.pt", "~56.9M params", "Maximum accuracy"),
        }
        for v, (weights, params, desc) in info.items():
            print(f"  {v:6s}  {weights:12s}  {params:15s}  {desc}")
        print()
        return

    # Determine config path
    if args.variant:
        config_path = VARIANT_MAP[args.variant]
    elif args.config:
        config_path = args.config
    else:
        config_path = "configs/training/yolov11_training.yaml"

    from lib.ml.training.training_config import TrainingConfig
    from lib.ml.training.trainer import SPIROYOLOTrainer
    from lib.ml.core.logger import get_logger

    log = get_logger("trainer")

    # Build overrides
    overrides = {}
    if args.epochs:   overrides["training.epochs"]     = args.epochs
    if args.batch:    overrides["training.batch_size"] = args.batch
    if args.lr:       overrides["optimizer.lr0"]       = args.lr
    if args.device:   overrides["training.device"]     = args.device
    if args.resume:   overrides["training.resume"]     = True
    if args.checkpoint: overrides["training.resume_checkpoint"] = args.checkpoint
    if args.name:     overrides["experiment.name"]     = args.name
    if args.workers:  overrides["dataset.workers"]     = args.workers
    if args.no_aug:   overrides["augmentation.enabled"] = False

    try:
        config = TrainingConfig.load(config_path, overrides=overrides)
    except FileNotFoundError:
        log.error(f"Config not found: {config_path}")
        log.error("Run `make install` to set up the project.")
        sys.exit(1)

    log.info(
        f"Starting training: "
        f"{config.model.weights} | "
        f"epochs={config.training.epochs} | "
        f"batch={config.training.batch_size} | "
        f"device={config.training.device}"
    )

    trainer = SPIROYOLOTrainer(config)
    summary = trainer.train()

    # Print final results
    metrics = summary.get("val_metrics", {})
    if metrics:
        print("\n" + "=" * 50)
        print("  TRAINING COMPLETE")
        print("=" * 50)
        for k, v in metrics.items():
            if not isinstance(v, dict):
                print(f"  {k:20s}: {v}")
        print(f"  Best weights: {summary.get('best_weights', 'N/A')}")
        print(f"  ONNX:         {summary.get('onnx_path', 'N/A')}")
        print(f"  Time:         {summary.get('elapsed_seconds', 0)/3600:.2f}h")
        print("=" * 50)


if __name__ == "__main__":
    main()
