#!/usr/bin/env python3
"""
SPIRO ML — ONNX Export CLI

Usage
-----
    # Export YOLO
    python -m lib.ml.scripts.export_onnx \
        --config configs/base_config.yaml \
        --weights models/checkpoints/yolov11_baseline/weights/best.pt \
        --architecture yolov11

    # Export EfficientNet
    python -m lib.ml.scripts.export_onnx \
        --config configs/base_config.yaml \
        --weights models/checkpoints/phase2_best.pt \
        --architecture efficientnetv2

    # Export production model from registry
    python -m lib.ml.scripts.export_onnx --config configs/base_config.yaml --production
"""
import argparse
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SPIRO ML ONNX Export")
    p.add_argument("--config", default="configs/base_config.yaml")
    p.add_argument("--weights", type=str, default=None)
    p.add_argument("--architecture", choices=["yolov11", "efficientnetv2"], default="yolov11")
    p.add_argument("--output-name", type=str, default=None,
                   help="Output filename (e.g. spiro_detector.onnx)")
    p.add_argument("--half", action="store_true", help="Export in FP16")
    p.add_argument("--production", action="store_true",
                   help="Export the production model from registry")
    p.add_argument("--verify", action="store_true", default=True,
                   help="Verify export with ONNXRuntime")
    p.add_argument("--register-onnx", action="store_true", default=True,
                   help="Update registry entry with onnx_path")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parents[5]))

    from lib.ml.core.config import ConfigManager
    from lib.ml.core.logger import get_logger
    from lib.ml.export.onnx_exporter import ONNXExporter
    from lib.ml.models.registry import ModelRegistry

    log = get_logger("export_onnx")
    cfg = ConfigManager.load(args.config)
    registry = ModelRegistry(cfg.paths.registry_dir)

    weights = args.weights
    arch = args.architecture

    if args.production:
        prod = registry.get_production()
        if not prod:
            log.error("No production model in registry")
            sys.exit(1)
        weights = prod["weights_path"]
        arch = prod["architecture"]
        log.info(f"Exporting production model: {prod['version']}")

    if not weights:
        log.error("Provide --weights or --production")
        sys.exit(1)

    exporter = ONNXExporter(cfg)
    out_name = args.output_name or f"spiro_{arch}.onnx"

    if arch == "yolov11":
        onnx_path = exporter.export_yolo(weights, output_name=out_name, half=args.half)
    else:
        from lib.ml.models.efficientnet_model import SPIROClassifier
        model = SPIROClassifier.load(cfg, weights)
        input_shape = (1, 3, *cfg.dataset.image_size[::-1])
        onnx_path = exporter.export_efficientnet(
            model, output_name=out_name, input_shape=input_shape, half=args.half
        )

    if args.verify:
        input_h, input_w = cfg.dataset.image_size[1], cfg.dataset.image_size[0]
        exporter.verify(onnx_path, input_shape=(1, 3, input_h, input_w))

    log.info(f"ONNX export complete: {onnx_path} ✓")


if __name__ == "__main__":
    main()
