# SPIRO ML — Production Guide

## Hardware Recommendations

| Tier | Hardware | Pipeline FPS | Use Case |
|---|---|---|---|
| Edge CPU | Intel i7 / Apple M2 | 8 FPS | Mobile / IoT |
| Cloud CPU | 16-core c6i | 14 FPS | Low-traffic API |
| GPU Entry | NVIDIA T4 16GB | 35 FPS | Standard API |
| GPU Mid | NVIDIA A10G 24GB | 60 FPS | High-traffic API |
| GPU High | NVIDIA A100 80GB | 100 FPS | Batch processing |

## Environment Selection

```bash
# Development
pipeline = SPIROPipeline.from_config("configs/development/pipeline.yaml")
# Staging
pipeline = SPIROPipeline.from_config("configs/staging/pipeline.yaml")
# Production
pipeline = SPIROPipeline.from_config("configs/production/pipeline.yaml")
```

## ORT Provider Optimisation

```yaml
runtime:
  providers: ["TensorrtExecutionProvider","CUDAExecutionProvider","CPUExecutionProvider"]
```

## ONNX Optimisation

```bash
# Graph optimisation (constant folding, op fusion)
python scripts/optimize.py --onnx models/exports/yolo.onnx --model-id yolov11s

# All formats (graph + FP16 + INT8)
python scripts/optimize.py --onnx models/exports/yolo.onnx --model-id yolov11s --all-formats
```

## Scaling

For high throughput, run multiple pipeline instances behind a load balancer.
Each instance has its own ORT session.

```python
results = pipeline.infer_batch(["img1.jpg", "img2.jpg", "img3.jpg"])
```

## Performance Monitoring

```python
from lib.ml.mlops import PerformanceMonitor

monitor = PerformanceMonitor(window_size=1000, slow_threshold_ms=300)
monitor.record_from_pipeline_result("req_001", pipeline_result)
report = monitor.report()
print(f"FPS: {report['fps']}, P95: {report['latency']['p95_ms']}ms")
```

## Troubleshooting

| Symptom | Fix |
|---|---|
| `FileNotFoundError: yolo_active.onnx` | `make mlops-deploy` |
| `P95 > 500ms on GPU` | Set `TensorrtExecutionProvider` |
| `All images rejected: blurry` | Lower `blur_threshold` to 0.0 |
| `CUDA out of memory` | Lower `effnet_batch` to 4 |
| `Drift detected` | Run `make mlops-cl-cycle` |
| `ORT crash on CUDA` | Use CPU providers, update CUDA drivers |
