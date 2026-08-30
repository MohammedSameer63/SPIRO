#!/usr/bin/env python3
"""
SPIRO ML — training/infer_pipeline.py
Production inference CLI for the full SPIRO pipeline.

Usage
-----
    # Single image
    python training/infer_pipeline.py --image path/to/photo.jpg

    # Directory
    python training/infer_pipeline.py --source path/to/images/ --output-dir results/

    # With custom config
    python training/infer_pipeline.py --image photo.jpg \\
        --config configs/pipeline/inference_pipeline.yaml

    # Override thresholds
    python training/infer_pipeline.py --image photo.jpg \\
        --conf 0.35 --iou 0.5 --fusion bayesian

    # Benchmark
    python training/infer_pipeline.py --benchmark --n-runs 100

    # Pretty JSON output
    python training/infer_pipeline.py --image photo.jpg --pretty

    # Save annotated visualisation
    python training/infer_pipeline.py --image photo.jpg --save-viz
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="SPIRO Full Inference Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    g = p.add_mutually_exclusive_group()
    g.add_argument("--image",  type=str, help="Single image path")
    g.add_argument("--source", type=str, help="Directory of images")
    g.add_argument("--benchmark", action="store_true", help="Run pipeline benchmark")

    p.add_argument("--config",
                   default="configs/pipeline/inference_pipeline.yaml",
                   help="Pipeline config YAML")
    p.add_argument("--output-dir", default="results/pipeline", help="Output directory")
    p.add_argument("--conf", type=float, default=None, help="Detection confidence threshold")
    p.add_argument("--iou",  type=float, default=None, help="NMS IoU threshold")
    p.add_argument("--fusion", type=str, default=None,
                   choices=["weighted_average", "geometric_mean",
                            "harmonic_mean", "bayesian", "temperature"])
    p.add_argument("--pretty", action="store_true", help="Pretty-print JSON output")
    p.add_argument("--save-viz", action="store_true", help="Save annotated images")
    p.add_argument("--n-runs", type=int, default=50, help="Benchmark runs")
    p.add_argument("--no-contamination", action="store_true")
    p.add_argument("--no-guidance",      action="store_true")
    p.add_argument("--no-explainability", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    from lib.ml.core.logger import get_logger
    log = get_logger("infer_pipeline")

    # Build overrides
    overrides: dict = {}
    if args.conf:      overrides["detection.conf_threshold"] = args.conf
    if args.iou:       overrides["detection.iou_threshold"]  = args.iou
    if args.fusion:    overrides["fusion.method"]            = args.fusion
    if args.save_viz:  overrides["explainability.save_visualizations"] = True
    if args.no_contamination: overrides["contamination.enabled"] = False
    if args.no_guidance:      overrides["guidance.enabled"]      = False
    if args.no_explainability:overrides["explainability.enabled"] = False

    # Build pipeline
    try:
        from lib.ml.pipeline import SPIROPipeline
        pipeline = SPIROPipeline.from_config(args.config, overrides=overrides)
    except FileNotFoundError as e:
        log.error(str(e))
        sys.exit(1)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    indent = 2 if args.pretty else None

    # ── Benchmark mode ────────────────────────────────────────────────
    if args.benchmark:
        log.info(f"Benchmarking pipeline ({args.n_runs} runs)...")
        stats = pipeline.benchmark(n_runs=args.n_runs)
        print(json.dumps(stats, indent=2))
        out = out_dir / "benchmark_results.json"
        with open(out, "w") as f:
            json.dump(stats, f, indent=2)
        log.info(f"Benchmark results → {out}")
        return

    # ── Single image ──────────────────────────────────────────────────
    if args.image:
        result = pipeline.infer(args.image)
        if args.pretty:
            print(json.dumps(result, indent=2))
        else:
            log.info(
                f"Status: {result['status']} | "
                f"Detections: {result['detection_count']} | "
                f"Total: {result.get('timings', {}).get('total_ms', 0):.1f}ms"
            )
            for det in result.get("detections", []):
                log.info(
                    f"  [{det['detection_id']}] {det['final_class_name']} "
                    f"({det['final_confidence']:.3f}) → {det.get('waste_stream','?')}"
                )

        out = out_dir / (Path(args.image).stem + "_result.json")
        with open(out, "w") as f:
            json.dump(result, f, indent=indent)
        log.info(f"Result → {out}")
        return

    # ── Directory mode ────────────────────────────────────────────────
    if args.source:
        source = Path(args.source)
        images = sorted(p for p in source.iterdir()
                        if p.suffix.lower() in IMAGE_EXTS)
        log.info(f"Processing {len(images)} images from {source}")

        all_results = []
        for img_path in images:
            result = pipeline.infer(str(img_path))
            all_results.append({"image": img_path.name, "result": result})
            log.info(
                f"  {img_path.name}: "
                f"{result['detection_count']} detections | "
                f"{result.get('timings', {}).get('total_ms', 0):.1f}ms"
            )

        # Save combined results
        out = out_dir / "batch_results.json"
        with open(out, "w") as f:
            json.dump(all_results, f, indent=indent)
        log.info(f"Batch results → {out}")

        # Summary
        log.info(f"\nSummary: {len(images)} images processed")
        total_dets = sum(r["result"]["detection_count"] for r in all_results)
        log.info(f"Total detections: {total_dets}")
        return

    log.error("Provide --image, --source, or --benchmark")
    sys.exit(1)


if __name__ == "__main__":
    main()
