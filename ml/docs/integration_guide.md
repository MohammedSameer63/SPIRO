# SPIRO ML — Integration Guide

This document is for teams integrating the SPIRO ML subsystem into a backend or application layer.

---

## What the ML package delivers

| Artefact | Location | Description |
|---|---|---|
| ONNX model | `models/exports/spiro_yolov11.onnx` | Self-describing, runs on CPU or GPU |
| ONNX metadata | embedded in the file | class names, input size, architecture |
| Python engine | `lib.ml.inference.ONNXInferenceEngine` | Production inference wrapper |
| JSON result | standard dict | bbox, confidence, class name |

---

## Minimal integration (ONNX only, any language)

The ONNX model accepts:
- **Input name**: `images`
- **Input shape**: `[1, 3, 640, 640]` float32, values in `[0, 1]`, RGB channel order
- **Preprocessing**: letterbox resize (keep aspect ratio, pad with 114)

### Python (ONNXRuntime)

```python
import onnxruntime as ort
import numpy as np
import cv2

session = ort.InferenceSession("models/exports/spiro_yolov11.onnx",
                                providers=["CUDAExecutionProvider", "CPUExecutionProvider"])

# Preprocess
img = cv2.imread("image.jpg")
img = cv2.resize(img, (640, 640))
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
blob = img.transpose(2, 0, 1)[np.newaxis]   # NCHW

# Infer
outputs = session.run(None, {"images": blob})
```

---

## Recommended integration (Python engine)

```python
from lib.ml import ConfigManager, ONNXInferenceEngine

cfg = ConfigManager.load("configs/base_config.yaml")

engine = ONNXInferenceEngine(
    onnx_path="models/exports/spiro_yolov11.onnx",
    input_size=(640, 640),
    conf_threshold=0.4,
    iou_threshold=0.45,
    warmup_runs=3,
)

# Single image
result = engine.infer("image.jpg")
# → {"type": "detection", "detections": [...]}

# With timing
result, elapsed_ms = engine.infer("image.jpg", return_timing=True)

# Batch
results = engine.infer_batch(["a.jpg", "b.jpg", "c.jpg"])

# Benchmark
stats = engine.benchmark(n_runs=200)
# → {"mean_ms": 8.2, "fps": 121.9, ...}
```

---

## Result schema

```python
# Detection
{
    "type": "detection",
    "detections": [
        {
            "bbox_xyxy": [x1, y1, x2, y2],   # pixels, original image coords
            "confidence": float,               # 0.0–1.0
            "class_id": int,                   # 0-indexed
            "class_name": str,                 # from config/metadata
        },
        ...
    ]
}

# Classification
{
    "type": "classification",
    "class_id": int,
    "class_name": str,
    "confidence": float,
    "probabilities": [float, ...]              # one per class, sums to 1
}
```

---

## Continuous Learning integration

```python
from lib.ml.continuous_learning import ContinuousLearningManager
from lib.ml.models import ModelRegistry

registry = ModelRegistry("models/registry")

def my_retrain():
    """Called automatically when drift is detected."""
    from lib.ml.training import YOLOTrainer
    t = YOLOTrainer(cfg)
    t.train()
    return t.validate()

clm = ContinuousLearningManager(cfg, registry, retrain_callback=my_retrain)

# Feed each inference result into the monitor
clm.observe(frame_bgr, result["detections"], confidence=0.85)

# Check and trigger retraining
if clm.should_retrain():
    entry = clm.trigger_retrain(new_version="v2", auto_promote=True)
    if entry:
        # Reload engine with new weights
        engine = ONNXInferenceEngine(entry["onnx_path"])
```

---

## Model Registry queries

```python
from lib.ml.models import ModelRegistry

registry = ModelRegistry("models/registry")

# Get production model
prod = registry.get_production()
print(prod["version"], prod["metrics"])

# Get best by metric
best = registry.best_version(metric="mAP50")

# List all
for entry in registry.list_versions():
    print(entry["version"], entry["metrics"].get("mAP50"))
```

---

## Performance expectations

| Hardware | Model | Latency | FPS |
|---|---|---|---|
| NVIDIA RTX 3080 | yolo11n ONNX | ~4ms | ~250 |
| NVIDIA RTX 3080 | yolo11s ONNX | ~7ms | ~140 |
| Intel Core i9 (CPU) | yolo11n ONNX | ~45ms | ~22 |
| Apple M2 (CPU ORT) | yolo11n ONNX | ~25ms | ~40 |

Run `engine.benchmark()` for your specific hardware.

---

## Environment variables

| Variable | Default | Description |
|---|---|---|
| `OMP_NUM_THREADS` | 4 | CPU inference threads |
| `CUDA_VISIBLE_DEVICES` | (all) | GPU selection |
| `MLFLOW_TRACKING_URI` | `logs/mlflow` | MLflow server |

---

## Errors and troubleshooting

| Error | Cause | Fix |
|---|---|---|
| `FileNotFoundError: *.onnx` | Model not exported yet | Run `export_onnx` CLI |
| `InvalidGraph` from onnxruntime | Opset mismatch | Ensure ORT ≥ 1.16 |
| `CUDA out of memory` | Batch too large | Reduce `inference.batch_size` |
| `mAP = 0.0` | Wrong dataset.yaml path | Check `path:` in dataset.yaml |
| Drift detected immediately | Baseline not set | Register a production model first |
