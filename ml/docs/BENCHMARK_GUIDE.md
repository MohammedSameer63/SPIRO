# SPIRO ML — Benchmark Guide

## Running Benchmarks

### Single model benchmark

```bash
# Benchmark YOLOv11 ONNX
python scripts/benchmark.py \
    --onnx models/exports/spiro_yolov11s_best.onnx \
    --model-id yolov11s --input-size 640 \
    --n-runs 200 --load-test

# Benchmark EfficientNetV2
python scripts/benchmark.py \
    --onnx models/exports/spiro_effnetv2_s_best_effnet.onnx \
    --model-id effnetv2_s --input-size 300 \
    --n-runs 200 --concurrent
```

### Full pipeline benchmark

```bash
python training/infer_pipeline.py --benchmark --n-runs 100
```

### Makefile shortcut

```bash
make pipeline-benchmark
```

## Metrics Collected

| Metric | Description |
|---|---|
| `mean_ms` | Mean inference latency |
| `p50_ms` | Median latency |
| `p95_ms` | 95th percentile latency |
| `p99_ms` | 99th percentile latency |
| `fps` | Frames per second (warm) |
| `cold_start_ms` | First-inference latency (includes model load) |
| `batch_throughput` | FPS at batch sizes 1, 4, 8, 16, 32 |
| `concurrent.throughput_fps` | FPS under N concurrent threads |
| `peak_ram_mb` | Peak process RAM usage |
| `peak_gpu_mb` | Peak GPU memory allocation |

## Programmatic Benchmarking

```python
from lib.ml.benchmarking.pipeline_benchmarker import PipelineBenchmarker
from pathlib import Path

benchmarker = PipelineBenchmarker(
    report_dir=Path("reports"),
    n_warmup=10,
    n_runs=200,
)
report = benchmarker.benchmark_model(
    onnx_path=Path("models/exports/yolo.onnx"),
    model_id="yolov11s",
    input_size=640,
    run_load_test=True,
    run_concurrent=True,
)
benchmarker.save_report(report)

print(f"FPS:        {report.fps:.1f}")
print(f"Cold start: {report.cold_start_ms:.0f}ms")
print(f"P95:        {report.stages[0].p95_ms:.1f}ms")
```

## Robustness Testing

```python
from lib.ml.robustness.robustness_tester import RobustnessTester
from lib.ml.pipeline import SPIROPipeline

pipeline = SPIROPipeline.from_config("configs/testing/pipeline.yaml")
tester = RobustnessTester(
    infer_fn=pipeline.infer,
    n_per_category=50,
    report_dir=Path("reports"),
)
report = tester.run_all(model_id="yolov11s")
tester.save_report(report)

print(f"Overall stability: {report.overall_stability:.1%}")
print(f"Overall pass rate: {report.overall_pass_rate:.1%}")
for cat in report.categories:
    print(f"  {cat.name:<22}: stability={cat.stability_rate:.1%}")
```

## Accuracy Validation

```python
from lib.ml.robustness.accuracy_validator import AccuracyValidator

validator = AccuracyValidator(
    abs_diff_threshold=0.01,
    top1_agreement_threshold=0.95,
    n_samples=100,
)
report = validator.compare_formats(
    fp32_path="models/exports/effnet.onnx",
    fp16_path="reports/optimization/effnetv2_s_fp16.onnx",
    int8_path="reports/optimization/effnetv2_s_int8_dynamic.onnx",
    model_id="effnetv2_s",
    input_size=300,
    task="verification",
)
print(f"All passed: {report.all_passed}")
```

## Expected Benchmark Numbers

### YOLOv11s — CPU (Intel i9, 4 threads)

| Metric | Value |
|---|---|
| Cold start | ~800ms |
| Mean latency | ~45ms |
| P95 latency | ~65ms |
| FPS (warm) | ~22 |
| Batch-8 FPS | ~35 |

### YOLOv11s — GPU (NVIDIA T4)

| Metric | Value |
|---|---|
| Cold start | ~2500ms |
| Mean latency | ~8ms |
| P95 latency | ~12ms |
| FPS (warm) | ~125 |
| Batch-8 FPS | ~320 |

### EfficientNetV2-S — CPU

| Metric | Value |
|---|---|
| Cold start | ~600ms |
| Mean latency | ~25ms |
| P95 latency | ~35ms |
| FPS (warm) | ~40 |
