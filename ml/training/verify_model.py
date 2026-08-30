#!/usr/bin/env python3
"""
SPIRO ML — training/verify_model.py
Inspect, summarise, and test a trained VerifierModel.

Usage
-----
    # Show model architecture info
    python training/verify_model.py info --variant s

    # Test forward pass
    python training/verify_model.py test --variant s

    # Test inference on an image
    python training/verify_model.py predict \\
        --weights models/verification_checkpoints/spiro_effnetv2_s/best_effnet.pt \\
        --config configs/verification/effnetv2_s.yaml \\
        --image path/to/crop.jpg

    # Generate Grad-CAM explanation
    python training/verify_model.py explain \\
        --weights models/verification_checkpoints/spiro_effnetv2_s/best_effnet.pt \\
        --config configs/verification/effnetv2_s.yaml \\
        --image path/to/crop.jpg \\
        --output-dir reports/gradcam
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="VerifierModel inspection")
    sub = p.add_subparsers(dest="command", required=True)

    # info
    info = sub.add_parser("info", help="Show model info")
    info.add_argument("--variant", default="s")
    info.add_argument("--config", default=None)

    # test
    test = sub.add_parser("test", help="Test forward pass")
    test.add_argument("--variant", default="s")
    test.add_argument("--config", default=None)
    test.add_argument("--batch-size", type=int, default=4)

    # predict
    pred = sub.add_parser("predict", help="Predict on an image")
    pred.add_argument("--weights", required=True)
    pred.add_argument("--config",  required=True)
    pred.add_argument("--image",   required=True)
    pred.add_argument("--top-k",   type=int, default=5)

    # explain
    exp = sub.add_parser("explain", help="Generate Grad-CAM explanation")
    exp.add_argument("--weights",    required=True)
    exp.add_argument("--config",     required=True)
    exp.add_argument("--image",      required=True)
    exp.add_argument("--output-dir", default="reports/gradcam")
    exp.add_argument("--method",     default="gradcam",
                     choices=["gradcam", "gradcam++"])

    return p.parse_args()


def main() -> None:
    args = parse_args()
    from lib.ml.core.logger import get_logger
    log = get_logger("verify_model")

    def _load_cfg(variant=None, config_path=None):
        from lib.ml.verification.verify_config import VerifyConfig
        if config_path:
            return VerifyConfig.load(config_path)
        base = "configs/verification/effnetv2_verify.yaml"
        ov = {"model.variant": f"efficientnetv2_{variant}"} if variant else {}
        return VerifyConfig.load(base, overrides=ov)

    if args.command == "info":
        from lib.ml.verification.model.verify_model import VerifierModel
        cfg = _load_cfg(args.variant, args.config)
        model = VerifierModel(cfg)
        counts = model.parameter_count()
        log.info(f"\nModel: {cfg.model.variant} ({cfg.timm_name})")
        log.info(f"  Parameters:  {counts['total']:,}")
        log.info(f"  Trainable:   {counts['trainable']:,}")
        log.info(f"  Input size:  {cfg.input_size}×{cfg.input_size}")
        log.info(f"  Num classes: {cfg.model.num_classes}")
        log.info(f"  Feature dim: {model.feature_dim()}")
        log.info(f"  Device:      {model.device}")

    elif args.command == "test":
        import torch
        from lib.ml.verification.model.verify_model import VerifierModel
        cfg = _load_cfg(args.variant, args.config)
        model = VerifierModel(cfg)
        model.eval()
        dummy = torch.randn(args.batch_size, 3, cfg.input_size, cfg.input_size).to(model.device)
        with torch.no_grad():
            out = model(dummy)
        log.info(f"Forward pass OK: input={tuple(dummy.shape)} → output={tuple(out.shape)}")
        assert out.shape == (args.batch_size, cfg.model.num_classes), "Shape mismatch!"
        log.info("Shape check ✓")

    elif args.command == "predict":
        from lib.ml.verification.model.verify_model import VerifierModel
        cfg = _load_cfg(config_path=args.config)
        model = VerifierModel.load(cfg, Path(args.weights))
        result = model.predict(args.image, top_k=args.top_k)
        log.info(f"\nPrediction for {Path(args.image).name}:")
        log.info(f"  Top-1: {result.class_name} ({result.confidence:.4f})")
        log.info(f"  Top-{len(result.top5_names)}:")
        for i, (name, conf) in enumerate(zip(result.top5_names, result.top5_confidences)):
            log.info(f"    {i+1}. {name:<30} {conf:.4f}")

    elif args.command == "explain":
        from lib.ml.verification.model.verify_model import VerifierModel
        from lib.ml.verification.explainability.gradcam import GradCAM
        cfg = _load_cfg(config_path=args.config)
        model = VerifierModel.load(cfg, Path(args.weights))
        cam = GradCAM(model, method=args.method)
        paths = cam.save_explanation(
            source=args.image,
            output_dir=Path(args.output_dir),
            stem=Path(args.image).stem,
        )
        log.info(f"\nExplanation files:")
        for k, v in paths.items():
            log.info(f"  {k}: {v}")
        cam.remove_hooks()


if __name__ == "__main__":
    main()
