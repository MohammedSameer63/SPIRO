#!/usr/bin/env python3
"""
SPIRO ML — scripts/benchmark.py
Benchmark ONNX models and the full pipeline.

Usage
-----
    python scripts/benchmark.py --onnx models/exports/yolo.onnx \
        --model-id yolov11s --input-size 640

    python scripts/benchmark.py --onnx models/exports/effnet.onnx \
        --model-id effnetv2_s --input-size 300 \
        --n-runs 200 --load-test

    python scripts/benchmark.py \
        --pipeline --config configs/production/pipeline.yaml \
        --n-runs 100
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


def main():
    from lib.ml.benchmarking.pipeline_benchmarker import (
        PipelineBenchmarker, OnnxSessionBenchmarker
    )
    from lib.ml.core.logger import get_logger
    log = get_logger("benchmark")

    p = argparse.ArgumentParser(description="SPIRO ML Benchmarker")
    p.add_argument("--onnx",        default=None)
    p.add_argument("--model-id",    default="spiro")
    p.add_argument("--input-size",  type=int, default=640)
    p.add_argument("--n-runs",      type=int, default=100)
    p.add_argument("--n-warmup",    type=int, default=5)
    p.add_argument("--load-test",   action="store_true")
    p.add_argument("--concurrent",  action="store_true")
    p.add_argument("--report-dir",  default="reports")
    args = p.parse_args()

    if not args.onnx:
        log.error("Provide --onnx path")
        sys.exit(1)

    onnx_path = Path(args.onnx)
    if not onnx_path.exists():
        log.error(f"ONNX not found: {onnx_path}")
        sys.exit(1)

    benchmarker = PipelineBenchmarker(
        report_dir=Path(args.report_dir),
        n_warmup=args.n_warmup,
        n_runs=args.n_runs,
    )
    report = benchmarker.benchmark_model(
        onnx_path=onnx_path,
        model_id=args.model_id,
        input_size=args.input_size,
        run_load_test=args.load_test,
        run_concurrent=args.concurrent,
    )
    benchmarker.save_report(report)

    print(f"\n{'='*55}")
    print(f"  Benchmark: {args.model_id}")
    print(f"{'='*55}")
    print(f"  FPS (warm):     {report.fps:.1f}")
    print(f"  Cold start:     {report.cold_start_ms:.0f}ms")
    print(f"  Warm mean:      {report.warm_start_ms:.1f}ms")
    for s in report.stages:
        print(f"  {s.name:<20}: mean={s.mean_ms:.1f}ms p95={s.p95_ms:.1f}ms")
    print(f"\nReports → {args.report_dir}/")


if __name__ == "__main__":
    main()
