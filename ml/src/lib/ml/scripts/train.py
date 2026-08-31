#!/usr/bin/env python3
"""
SPIRO ML — Training CLI

Usage
-----
    # YOLO detection
    python -m lib.ml.scripts.train --config configs/experiments/yolov11_baseline.yaml

    # EfficientNetV2 classification
    python -m lib.ml.scripts.train \
        --config configs/experiments/efficientnetv2_classifier.yaml \
        --train-dir datasets/processed/train \
        --val-dir datasets/processed/val

    # Resume YOLO training
    python -m lib.ml.scripts.train --config configs/base_config.yaml --resume --checkpoint models/checkpoints/spiro_yolo/weights/last.pt
"""
import argparse
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SPIRO ML Training")
    p.add_argument("--config", default="configs/base_config.yaml")
    p.add_argument("--architecture", choices=["yolov11", "efficientnetv2", "auto"], default="auto",
                   help="Override architecture. 'auto' reads from config.")
    p.add_argument("--train-dir", default="datasets/processed/train",
                   help="ImageFolder train dir (EfficientNet only)")
    p.add_argument("--val-dir", default="datasets/processed/val",
                   help="ImageFolder val dir (EfficientNet only)")
    p.add_argument("--resume", action="store_true", help="Resume YOLO training")
    p.add_argument("--checkpoint", type=str, default=None,
                   help="Checkpoint to resume from")
    p.add_argument("--register", action="store_true", default=True,
                   help="Register model in ModelRegistry after training")
    p.add_argument("--version", type=str, default="v1",
                   help="Registry version tag")
    p.add_argument("--overrides", nargs="*", default=[],
                   help="Config overrides: training.epochs=50 training.batch_size=8")
    return p.parse_args()


def _parse_overrides(overrides):
    result = {}
    for item in overrides:
        if "=" not in item:
            continue
        k, v = item.split("=", 1)
        # Try to cast
        try:
            v = int(v)
        except ValueError:
            try:
                v = float(v)
            except ValueError:
                if v.lower() in ("true", "false"):
                    v = v.lower() == "true"
        result[k] = v
    return result


def main() -> None:
    args = parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parents[5]))

    from lib.ml.core.config import ConfigManager
    from lib.ml.core.logger import get_logger
    from lib.ml.models.registry import ModelRegistry

    log = get_logger("train")
    overrides = _parse_overrides(args.overrides)

    # Resume override
    if args.resume and args.checkpoint:
        overrides["training.resume"] = True
        overrides["training.resume_checkpoint"] = args.checkpoint

    cfg = ConfigManager.load(args.config, overrides=overrides)
    arch = args.architecture if args.architecture != "auto" else cfg.model.architecture

    registry = ModelRegistry(cfg.paths.registry_dir)
    metrics: dict = {}

    if arch == "yolov11":
        from lib.ml.training.yolo_trainer import YOLOTrainer
        trainer = YOLOTrainer(cfg)
        trainer.train()
        metrics = trainer.validate()
        best_weights = trainer.best_weights

        if args.register and best_weights:
            registry.register(
                version=args.version,
                architecture="yolov11",
                weights_path=best_weights,
                metrics=metrics,
                tags=["yolov11", "detection"],
            )
            log.info(f"Registered as version {args.version!r}")

    elif arch == "efficientnetv2":
        from lib.ml.training.efficientnet_trainer import EfficientNetTrainer
        trainer = EfficientNetTrainer(cfg)
        history = trainer.train(train_dir=args.train_dir, val_dir=args.val_dir)
        best_ckpt = trainer.best_checkpoint
        metrics = {
            "best_val_acc": trainer._best_val_acc,
            "final_train_loss": history["train_loss"][-1],
        }
        if args.register and best_ckpt:
            registry.register(
                version=args.version,
                architecture="efficientnetv2",
                weights_path=best_ckpt,
                metrics=metrics,
                tags=["efficientnetv2", "classification"],
            )
            log.info(f"Registered as version {args.version!r}")

    else:
        log.error(f"Unknown architecture: {arch}")
        sys.exit(1)

    log.info(f"Training complete. Final metrics: {metrics}")


if __name__ == "__main__":
    main()
