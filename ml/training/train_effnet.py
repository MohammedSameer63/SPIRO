#!/usr/bin/env python3
"""
SPIRO ML — training/train_effnet.py
Primary entry point for EfficientNetV2 verifier training.

Usage
-----
    # Default config (efficientnetv2_s)
    python training/train_effnet.py

    # Specific variant
    python training/train_effnet.py --config configs/verification/effnetv2_b0.yaml
    python training/train_effnet.py --variant s
    python training/train_effnet.py --variant m --epochs 60 --batch 16

    # Override config values
    python training/train_effnet.py --variant s \\
        --set training.epochs=30 optimizer.lr0=0.0005 loss.name=focal

    # Resume training
    python training/train_effnet.py --variant s --resume

    # With explicit checkpoint
    python training/train_effnet.py --variant s \\
        --resume --checkpoint models/verification_checkpoints/spiro_effnetv2_s/last_effnet.pt

    # List available variants
    python training/train_effnet.py --list-variants
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

VARIANT_MAP = {
    "b0": "configs/verification/effnetv2_b0.yaml",
    "b1": "configs/verification/effnetv2_verify.yaml",  # uses base with override
    "b2": "configs/verification/effnetv2_verify.yaml",
    "b3": "configs/verification/effnetv2_verify.yaml",
    "s":  "configs/verification/effnetv2_s.yaml",
    "m":  "configs/verification/effnetv2_m.yaml",
    "l":  "configs/verification/effnetv2_verify.yaml",
}

VARIANT_INFO = {
    "b0": ("efficientnetv2_b0", "~7M", "192px", "Fastest — edge deployment"),
    "b1": ("efficientnetv2_b1", "~8M", "240px", "Small — mobile"),
    "b2": ("efficientnetv2_b2", "~10M", "260px", "Compact"),
    "b3": ("efficientnetv2_b3", "~14M", "300px", "Good balance"),
    "s":  ("efficientnetv2_s",  "~22M", "300px", "Recommended default"),
    "m":  ("efficientnetv2_m",  "~54M", "384px", "High accuracy"),
    "l":  ("efficientnetv2_l",  "~119M","480px", "Maximum accuracy"),
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train SPIRO EfficientNetV2 Verifier",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    g = p.add_mutually_exclusive_group()
    g.add_argument("--variant", choices=list(VARIANT_MAP.keys()),
                   help="EfficientNetV2 variant shortcut")
    g.add_argument("--config", type=str,
                   help="Full path to verification config YAML")
    p.add_argument("--list-variants", action="store_true")
    p.add_argument("--epochs",   type=int,   default=None)
    p.add_argument("--batch",    type=int,   default=None)
    p.add_argument("--lr",       type=float, default=None)
    p.add_argument("--device",   type=str,   default=None)
    p.add_argument("--loss",     type=str,   default=None,
                   choices=["cross_entropy", "label_smoothing", "focal", "weighted"])
    p.add_argument("--resume",   action="store_true")
    p.add_argument("--checkpoint", type=str, default=None)
    p.add_argument("--name",     type=str,   default=None)
    p.add_argument("--set",      nargs="*",  default=[], metavar="KEY=VALUE")
    p.add_argument("--extract-crops", action="store_true",
                   help="Extract crop patches from YOLO splits before training")
    return p.parse_args()


def _parse_overrides(items):
    overrides = {}
    for item in items:
        if "=" not in item:
            continue
        k, v = item.split("=", 1)
        try: v = int(v)
        except ValueError:
            try: v = float(v)
            except ValueError:
                if v.lower() in ("true", "false"):
                    v = v.lower() == "true"
        overrides[k] = v
    return overrides


def main() -> None:
    args = parse_args()

    if args.list_variants:
        print("\nAvailable EfficientNetV2 Variants:")
        print("-" * 70)
        for k, (name, params, size, desc) in VARIANT_INFO.items():
            print(f"  {k:3s}  {name:<20} {params:<8} {size:<8} {desc}")
        print()
        return

    from lib.ml.verification.verify_config import VerifyConfig
    from lib.ml.verification.training.verify_trainer import VerifyTrainer
    from lib.ml.core.logger import get_logger
    log = get_logger("train_effnet")

    # Determine config
    if args.variant:
        config_path = VARIANT_MAP[args.variant]
        overrides = {"model.variant": f"efficientnetv2_{args.variant}"}
    elif args.config:
        config_path = args.config
        overrides = {}
    else:
        config_path = "configs/verification/effnetv2_verify.yaml"
        overrides = {}

    # Apply CLI overrides
    overrides.update(_parse_overrides(args.set))
    if args.epochs:   overrides["training.epochs"]     = args.epochs
    if args.batch:    overrides["training.batch_size"] = args.batch
    if args.lr:       overrides["optimizer.lr0"]       = args.lr
    if args.device:   overrides["training.device"]     = args.device
    if args.loss:     overrides["loss.name"]           = args.loss
    if args.name:     overrides["experiment.name"]     = args.name
    if args.resume:   overrides["training.resume"]     = True
    if args.checkpoint: overrides["training.resume_checkpoint"] = args.checkpoint

    # Load config
    try:
        cfg = VerifyConfig.load(config_path, overrides=overrides)
    except Exception as e:
        log.error(f"Config load failed: {e}")
        sys.exit(1)

    log.info(f"Variant:  {cfg.model.variant} ({cfg.timm_name})")
    log.info(f"Input:    {cfg.input_size}×{cfg.input_size}")
    log.info(f"Classes:  {cfg.model.num_classes}")
    log.info(f"Epochs:   {cfg.training.epochs} (ph1={cfg.training.phase1_epochs} ph2={cfg.training.phase2_epochs})")
    log.info(f"Loss:     {cfg.loss.name}")
    log.info(f"Device:   {cfg.training.device}")

    # Optional crop extraction
    if args.extract_crops:
        log.info("Extracting crops from YOLO dataset splits...")
        from lib.ml.verification.training.verify_dataset import CropExtractor
        extractor = CropExtractor(output_dir=Path(cfg.dataset.crop_dir))
        for split in ["train", "val", "test"]:
            split_dir = Path(cfg.dataset.root) / split
            if (split_dir / "images").exists():
                counts = extractor.extract_from_split(
                    split_dir / "images", split_dir / "labels", split=split
                )
                log.info(f"  {split}: {sum(counts.values())} crops extracted")

    # Train
    trainer = VerifyTrainer(cfg)
    summary = trainer.train()

    log.info("\n" + "=" * 50)
    log.info("  TRAINING COMPLETE")
    log.info("=" * 50)
    log.info(f"  Best top-1:   {summary.get('best_top1', 0):.4f}")
    log.info(f"  Best ckpt:    {summary.get('best_checkpoint', 'N/A')}")
    log.info(f"  ONNX:         {summary.get('onnx_path', 'N/A')}")
    log.info(f"  Time:         {summary.get('elapsed_s', 0)/60:.1f}min")
    log.info("=" * 50)


if __name__ == "__main__":
    main()
