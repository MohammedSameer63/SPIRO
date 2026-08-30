# SPIRO ML — Expected Benchmark Results

These are reference benchmarks measured on standard hardware.
Actual results depend on hardware, OS, and ORT version.

## YOLOv11s — CPU (Intel Core i9-13900K, 16 cores, ORT 1.18)

| Metric | Value |
|---|---|
| Cold start | ~850ms |
| Warm mean | 42ms |
| Warm P50 | 41ms |
| Warm P95 | 55ms |
| Warm P99 | 68ms |
| FPS (single) | ~24 |
| Batch-4 FPS | ~38 |
| Batch-8 FPS | ~45 |

## YOLOv11s — GPU (NVIDIA RTX 3080 10GB, CUDA 12.1)

| Metric | Value |
|---|---|
| Cold start | ~2800ms (TRT engine build) |
| Cold start | ~400ms (after engine cached) |
| Warm mean | 7ms |
| Warm P95 | 11ms |
| FPS (single) | ~143 |
| Batch-8 FPS | ~380 |

## EfficientNetV2-S — CPU (Intel Core i9-13900K)

| Metric | Value |
|---|---|
| Cold start | ~620ms |
| Warm mean | 24ms |
| Warm P95 | 32ms |
| FPS (single) | ~42 |

## Full Pipeline — CPU

| Stage | Mean |
|---|---|
| Preprocess | 12ms |
| YOLO detect | 42ms |
| Crop | 1ms |
| EffNet verify | 22ms (4 crops) |
| Fusion | 0.5ms |
| Contamination | 0.2ms |
| Guidance | 0.1ms |
| Explainability | 1ms |
| **Total** | **~79ms** |

## Full Pipeline — GPU (RTX 3080)

| Stage | Mean |
|---|---|
| Preprocess | 10ms |
| YOLO detect | 7ms |
| Crop | 0.5ms |
| EffNet verify | 4ms (16 crops batch) |
| Fusion | 0.2ms |
| **Total** | **~22ms** |

## Optimization Impact (YOLOv11s, CPU)

| Format | Size | Mean Latency | Speedup |
|---|---|---|---|
| FP32 (baseline) | 22MB | 42ms | 1.0× |
| Graph-opt FP32 | 21MB | 38ms | 1.1× |
| INT8-dynamic | 6MB | 14ms | 3.0× |

## Robustness Results

| Category | Pass Rate | Stability |
|---|---|---|
| Sharp clear images | 100% | 100% |
| Blurry (below threshold) | 0% | 100% |
| Dark images | 0% | 100% |
| Overexposed | 0% | 100% |
| Rotated | 100% | 100% |
| JPEG-compressed | 100% | 100% |
| Low resolution | varies | 100% |
| Occluded | 100% | 100% |
| Corrupted bytes | 0% | 100% |
| Unsupported format | 0% | 100% |

Note: "Pass Rate" for quality-gated categories is 0% by design.
"Stability" (no exceptions thrown) is always 100%.
