#!/usr/bin/env python3
"""
SPIRO ML — training/verify_export.py
Export EfficientNetV2 verifier to ONNX and verify the export.

Usage
-----
    python training/verify_export.py \\
        --config configs/verification/effnetv2_s.yaml \\
        --weights models/verification_checkpoints/spiro_effnetv2_s/best_effnet.pt

    python training/verify_export.py \\
        --config configs/verification/effnetv2_s.yaml \\
        --weights models/verification_checkpoints/spiro_effnetv2_s/best_effnet.pt \\
        --benchmark --n-runs 200

    python training/verify_export.py \\
        --config configs/verification/effnetv2_s.yaml \\
        --weights path/to/best.pt \\
        --output-name my_model.onnx \\
        --half
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export EfficientNetV2 to ONNX")
    p.add_argument("--config",  required=True, help="Verification config YAML")
    p.add_argument("--weights", required=True, help="Path to best_effnet.pt")
    p.add_argument("--output-name", default=None, help="Output .onnx filename")
    p.add_argument("--half", action="store_true", help="FP16 export")
    p.add_argument("--no-verify", action="store_true", help="Skip ORT verification")
    p.add_argument("--benchmark", action="store_true", help="Benchmark ORT after export")
    p.add_argument("--n-runs", type=int, default=100)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    from lib.ml.core.logger import get_logger
    from lib.ml.verification.verify_config import VerifyConfig
    from lib.ml.verification.model.verify_model import VerifierModel
    from lib.ml.verification.export.verify_export import VerifyExporter

    log = get_logger("verify_export")

    overrides = {}
    if args.half:
        overrides["export.half"] = True
    if args.no_verify:
        overrides["export.verify"] = False

    cfg = VerifyConfig.load(args.config, overrides=overrides)
    log.info(f"Loading model from {args.weights}")
    model = VerifierModel.load(cfg, Path(args.weights))

    exporter = VerifyExporter(cfg)
    onnx_path = exporter.export(model, output_name=args.output_name)
    log.info(f"ONNX exported: {onnx_path}")

    if args.benchmark:
        stats = exporter.benchmark_ort(onnx_path, n_runs=args.n_runs)
        log.info(
            f"ORT benchmark ({args.n_runs} runs):\n"
            f"  mean={stats['mean_ms']:.2f}ms  fps={stats['fps']:.1f}"
        )


if __name__ == "__main__":
    main()
