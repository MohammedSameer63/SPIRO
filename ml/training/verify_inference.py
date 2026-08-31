#!/usr/bin/env python3
"""
SPIRO ML — training/verify_inference.py
Run EfficientNetV2 verification inference on images or crops.

Usage
-----
    # Verify single image
    python training/verify_inference.py \\
        --onnx models/exports/spiro_effnetv2_s_best.onnx \\
        --source path/to/crop.jpg

    # Verify a directory of crops
    python training/verify_inference.py \\
        --onnx models/exports/spiro_effnetv2_s_best.onnx \\
        --source datasets/crops/train/ \\
        --output-dir results/verification/

    # Full two-stage pipeline (YOLO + EfficientNet + fusion)
    python training/verify_inference.py \\
        --onnx models/exports/spiro_effnetv2_s_best.onnx \\
        --yolo-onnx models/exports/spiro_yolov11s_best.onnx \\
        --source path/to/image.jpg \\
        --fusion weighted_average

    # Benchmark latency
    python training/verify_inference.py \\
        --onnx models/exports/spiro_effnetv2_s_best.onnx \\
        --benchmark --n-runs 200

    # Calibrate temperature
    python training/verify_inference.py \\
        --onnx models/exports/spiro_effnetv2_s_best.onnx \\
        --calibrate --calibration-dir datasets/processed/val
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SPIRO EfficientNetV2 Inference")
    p.add_argument("--onnx", required=True, help="Path to verification .onnx model")
    p.add_argument("--source", type=str, default=None,
                   help="Image path or directory")
    p.add_argument("--yolo-onnx", type=str, default=None,
                   help="YOLOv11 ONNX for two-stage pipeline")
    p.add_argument("--output-dir", default="results/verification")
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--fusion", type=str, default="weighted_average",
                   choices=["weighted_average", "geometric_mean", "harmonic_mean",
                            "bayesian", "temperature"])
    p.add_argument("--config", default="configs/verification/effnetv2_verify.yaml")
    p.add_argument("--benchmark", action="store_true")
    p.add_argument("--n-runs", type=int, default=100)
    p.add_argument("--calibrate", action="store_true",
                   help="Calibrate temperature on --calibration-dir")
    p.add_argument("--calibration-dir", type=str, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    from lib.ml.core.logger import get_logger
    from lib.ml.verification.verify_config import VerifyConfig
    from lib.ml.verification.inference.verify_inference import VerifyInferenceEngine

    log = get_logger("verify_inference")

    try:
        cfg = VerifyConfig.load(args.config)
        input_size = cfg.input_size
    except Exception:
        input_size = 300

    engine = VerifyInferenceEngine(
        onnx_path=args.onnx,
        input_size=input_size,
        temperature=args.temperature,
    )

    if args.benchmark:
        stats = engine.benchmark(n_runs=args.n_runs)
        log.info(
            f"Benchmark ({args.n_runs} runs):\n"
            f"  mean={stats['mean_ms']:.2f}ms  std={stats['std_ms']:.2f}ms\n"
            f"  min={stats['min_ms']:.2f}ms  p95={stats['p95_ms']:.2f}ms\n"
            f"  FPS={stats['fps']:.1f}"
        )
        return

    if args.calibrate and args.calibration_dir:
        import numpy as np
        import cv2
        log.info(f"Calibrating temperature on {args.calibration_dir}")
        cal_dir = Path(args.calibration_dir)
        all_logits, all_labels = [], []
        from lib.ml.dataset_engineering.taxonomy import TaxonomyLoader
        tax = TaxonomyLoader()

        for class_dir in sorted(cal_dir.iterdir()):
            if not class_dir.is_dir():
                continue
            try:
                cls_id = tax.name_to_id(class_dir.name)
            except KeyError:
                continue
            for img_path in list(class_dir.iterdir())[:50]:
                if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                    continue
                img = cv2.imread(str(img_path))
                if img is None:
                    continue
                blob = engine._preprocess(img)
                logits = engine.session.run(None, {engine.input_name: blob})[0][0]
                all_logits.append(logits)
                all_labels.append(cls_id)

        if all_logits:
            T = engine.calibrate_temperature(
                np.array(all_logits), np.array(all_labels)
            )
            log.info(f"Optimal temperature: {T:.4f}")
            out = Path(args.output_dir) / "calibration_result.json"
            out.parent.mkdir(parents=True, exist_ok=True)
            with open(out, "w") as f:
                json.dump({"optimal_temperature": T}, f)
        return

    if not args.source:
        log.error("Provide --source or --benchmark or --calibrate")
        sys.exit(1)

    source = Path(args.source)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    paths = sorted(source.iterdir()) if source.is_dir() else [source]
    paths = [p for p in paths if p.suffix.lower() in IMAGE_EXTS]

    # Two-stage pipeline
    if args.yolo_onnx:
        from lib.ml.inference.onnx_engine import ONNXInferenceEngine
        from lib.ml.verification.fusion.confidence_fusion import ConfidenceFusion

        detector = ONNXInferenceEngine(args.yolo_onnx, warmup_runs=2)
        fusion = ConfidenceFusion(method=args.fusion)

        all_results = []
        for img_path in paths:
            import cv2
            img_bgr = cv2.imread(str(img_path))
            if img_bgr is None:
                continue
            det_result = detector.infer(img_bgr)
            detections = det_result.get("detections", [])
            enriched = engine.verify_detections(img_bgr, detections, top_k=args.top_k)
            fused = []
            for det in enriched:
                if det.get("verification") is not None:
                    fr = fusion.fuse(det, det["verification"])
                    fused.append({**fr.to_dict(), "original_detection": det})
                else:
                    fused.append(det)
            all_results.append({"image": img_path.name, "results": fused})
            log.info(f"{img_path.name}: {len(fused)} detections verified")
    else:
        # Verification-only mode
        all_results = []
        for img_path in paths:
            result, elapsed = engine.verify(img_path, top_k=args.top_k, return_timing=True)
            all_results.append({
                "image": img_path.name,
                "class_name": result["class_name"],
                "confidence": result["confidence"],
                "top_k": result["top_k"],
                "elapsed_ms": round(elapsed, 2),
            })
            log.info(
                f"{img_path.name} → {result['class_name']} "
                f"({result['confidence']:.3f}) | {elapsed:.1f}ms"
            )

    out = out_dir / "verification_results.json"
    with open(out, "w") as f:
        json.dump(all_results, f, indent=2)
    log.info(f"Results → {out}")


if __name__ == "__main__":
    main()
