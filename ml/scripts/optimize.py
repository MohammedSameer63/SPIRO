#!/usr/bin/env python3
"""
SPIRO ML — scripts/optimize.py
Optimize ONNX models for production deployment.

Usage
-----
    python scripts/optimize.py \
        --onnx models/exports/spiro_yolov11s_best.onnx \
        --model-id yolov11s \
        --input-size 640

    python scripts/optimize.py \
        --onnx models/exports/effnet.onnx \
        --model-id effnetv2_s \
        --input-size 300 \
        --skip-int8

    python scripts/optimize.py \
        --onnx models/exports/yolo.onnx \
        --model-id yolov11s \
        --all-formats \
        --report
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


def main():
    from lib.ml.optimization.onnx_optimizer import ONNXOptimizer
    from lib.ml.core.logger import get_logger
    log = get_logger("optimize")

    p = argparse.ArgumentParser(description="SPIRO ONNX Model Optimizer")
    p.add_argument("--onnx",        required=True, help="Input ONNX model path")
    p.add_argument("--model-id",    required=True)
    p.add_argument("--input-size",  type=int, default=640)
    p.add_argument("--batch-size",  type=int, default=1)
    p.add_argument("--skip-fp16",   action="store_true")
    p.add_argument("--skip-int8",   action="store_true")
    p.add_argument("--all-formats", action="store_true",
                   help="Run all optimization formats")
    p.add_argument("--report-dir",  default="reports/optimization")
    p.add_argument("--max-diff",    type=float, default=0.01)
    p.add_argument("--n-runs",      type=int, default=30)
    args = p.parse_args()

    onnx_path = Path(args.onnx)
    if not onnx_path.exists():
        log.error(f"ONNX file not found: {onnx_path}")
        sys.exit(1)

    optimizer = ONNXOptimizer(
        report_dir=Path(args.report_dir),
        max_diff_threshold=args.max_diff,
        n_benchmark_runs=args.n_runs,
    )

    log.info(f"Optimizing: {onnx_path.name} ({onnx_path.stat().st_size/1e6:.1f}MB)")
    results = optimizer.optimize_all(
        onnx_path=onnx_path,
        model_id=args.model_id,
        input_size=args.input_size,
        batch_size=args.batch_size,
        skip_int8=args.skip_int8 and not args.all_formats,
        skip_fp16=args.skip_fp16 and not args.all_formats,
    )

    print(f"\n{'='*60}")
    print(f"  Optimization Results: {args.model_id}")
    print(f"{'='*60}")
    for r in results:
        passed = "✓" if r.passed_validation else "✗"
        print(f"  [{passed}] {r.optimization_type:<15} "
              f"size:{r.size_reduction_pct:+.1f}%  "
              f"speed:{r.speedup_x:.2f}x  "
              f"diff:{r.max_output_diff:.6f}")
    print(f"\nReport: {args.report_dir}/{args.model_id}_optimization_report.json")


if __name__ == "__main__":
    main()
