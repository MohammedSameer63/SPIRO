#!/usr/bin/env python3
"""
SPIRO ML — Evaluation CLI

Usage
-----
    # Evaluate detection model on test split
    python -m lib.ml.scripts.evaluate \
        --config configs/base_config.yaml \
        --weights models/checkpoints/yolov11_baseline/weights/best.pt \
        --task detect

    # Evaluate from registry production model
    python -m lib.ml.scripts.evaluate --config configs/base_config.yaml --production
"""
import argparse
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SPIRO ML Evaluation")
    p.add_argument("--config", default="configs/base_config.yaml")
    p.add_argument("--weights", type=str, default=None, help="Path to .pt weights")
    p.add_argument("--task", choices=["detect", "classify"], default="detect")
    p.add_argument("--split", choices=["val", "test"], default="test")
    p.add_argument("--production", action="store_true",
                   help="Evaluate the current production model from registry")
    p.add_argument("--output-dir", default="reports")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parents[5]))

    from lib.ml.core.config import ConfigManager
    from lib.ml.core.logger import get_logger
    from lib.ml.models.registry import ModelRegistry

    log = get_logger("evaluate")
    cfg = ConfigManager.load(args.config)

    weights = args.weights
    if args.production:
        registry = ModelRegistry(cfg.paths.registry_dir)
        prod = registry.get_production()
        if not prod:
            log.error("No production model in registry")
            sys.exit(1)
        weights = prod["weights_path"]
        log.info(f"Evaluating production model: {prod['version']} @ {weights}")

    if not weights:
        log.error("Provide --weights or use --production")
        sys.exit(1)

    if args.task == "detect":
        from lib.ml.evaluation.evaluator import DetectionEvaluator
        ev = DetectionEvaluator(cfg)
        report = ev.evaluate(weights, split=args.split, save_report=True)
        log.info(f"mAP50={report['mAP50']:.4f} | mAP50-95={report['mAP50-95']:.4f}")

    else:
        log.warning(
            "Classification evaluation requires y_true/y_pred arrays. "
            "Use ClassificationEvaluator directly in your pipeline."
        )

    log.info("Evaluation complete ✓")


if __name__ == "__main__":
    main()
