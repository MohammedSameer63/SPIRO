#!/usr/bin/env python3
"""
SPIRO ML — Inference CLI

Usage
-----
    # Single image
    python -m lib.ml.scripts.infer \
        --onnx models/exports/spiro_yolov11.onnx \
        --source path/to/image.jpg

    # Directory
    python -m lib.ml.scripts.infer \
        --onnx models/exports/spiro_yolov11.onnx \
        --source path/to/images/ \
        --output-dir results/

    # Benchmark
    python -m lib.ml.scripts.infer \
        --onnx models/exports/spiro_yolov11.onnx \
        --benchmark --n-runs 200
"""
import argparse
import json
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SPIRO ML Inference")
    p.add_argument("--onnx", required=True, help="Path to .onnx model")
    p.add_argument("--source", type=str, default=None,
                   help="Image path or directory")
    p.add_argument("--output-dir", type=str, default="results",
                   help="Directory to write annotated images + JSON results")
    p.add_argument("--conf", type=float, default=None, help="Confidence threshold override")
    p.add_argument("--iou", type=float, default=None, help="IoU threshold override")
    p.add_argument("--benchmark", action="store_true", help="Run latency benchmark")
    p.add_argument("--n-runs", type=int, default=100, help="Number of benchmark runs")
    p.add_argument("--class-names", nargs="*", default=None,
                   help="Class names (space-separated). Reads from ONNX metadata if not set.")
    p.add_argument("--save-images", action="store_true", default=True,
                   help="Save annotated images")
    p.add_argument("--config", default="configs/base_config.yaml")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parents[5]))

    import cv2
    from lib.ml.core.config import ConfigManager
    from lib.ml.core.logger import get_logger
    from lib.ml.inference.onnx_engine import ONNXInferenceEngine

    log = get_logger("infer")
    cfg = ConfigManager.load(args.config)

    class_names = args.class_names or list(cfg.dataset.class_names)
    engine = ONNXInferenceEngine(
        onnx_path=args.onnx,
        input_size=tuple(cfg.model.input_size),
        conf_threshold=args.conf or cfg.inference.conf_threshold,
        iou_threshold=args.iou or cfg.inference.iou_threshold,
        class_names=class_names,
    )

    # Benchmark mode
    if args.benchmark:
        stats = engine.benchmark(n_runs=args.n_runs)
        log.info(
            f"Benchmark results:\n"
            f"  mean={stats['mean_ms']:.2f}ms  std={stats['std_ms']:.2f}ms\n"
            f"  min={stats['min_ms']:.2f}ms  max={stats['max_ms']:.2f}ms\n"
            f"  FPS={stats['fps']:.1f}"
        )
        print(json.dumps(stats, indent=2))
        return

    if not args.source:
        log.error("Provide --source or --benchmark")
        sys.exit(1)

    source = Path(args.source)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect image paths
    IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    if source.is_dir():
        paths = sorted(p for p in source.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    else:
        paths = [source]

    log.info(f"Running inference on {len(paths)} image(s)")
    all_results = []

    for img_path in paths:
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            log.warning(f"Cannot read {img_path}")
            continue

        result, elapsed = engine.infer(img_bgr, return_timing=True)
        result["image"] = img_path.name
        result["elapsed_ms"] = round(elapsed, 2)
        all_results.append(result)

        log.info(f"{img_path.name} — {elapsed:.1f}ms | "
                 f"{len(result.get('detections', []))} detections")

        # Draw and save
        if args.save_images and result.get("type") == "detection":
            canvas = img_bgr.copy()
            for det in result["detections"]:
                x1, y1, x2, y2 = [int(v) for v in det["bbox_xyxy"]]
                cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 200, 80), 2)
                label = f"{det['class_name']} {det['confidence']:.2f}"
                cv2.putText(canvas, label, (x1, max(0, y1 - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 80), 1)
            out_img = output_dir / f"pred_{img_path.name}"
            cv2.imwrite(str(out_img), canvas)

    # Write JSON
    out_json = output_dir / "results.json"
    with open(out_json, "w") as f:
        json.dump(all_results, f, indent=2)
    log.info(f"Results written to {out_json}")


if __name__ == "__main__":
    main()
