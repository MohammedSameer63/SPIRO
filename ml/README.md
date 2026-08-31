# SPIRO ML — Machine Learning Subsystem

Production-ready AI/ML package for SPIRO. Standalone. No frontend. No backend. Fully integrable.

---

## Overview

| Layer | Component | Technology |
|---|---|---|
| Detection | `SPIRODetector` | YOLOv11 (Ultralytics) |
| Classification | `SPIROClassifier` | EfficientNetV2 (timm) |
| Augmentation | `OfflineAugmentor` | Albumentations |
| Inference | `ONNXInferenceEngine` | ONNXRuntime |
| Export | `ONNXExporter` | ONNX opset 17 |
| Evaluation | `DetectionEvaluator`, `ClassificationEvaluator` | scikit-learn + matplotlib |
| Monitoring | `ContinuousLearningManager` + `DriftDetector` | Page-Hinkley test |
| Registry | `ModelRegistry` | File-based JSON |
| Tracking | TensorBoard + MLflow | Integrated in trainers |

---

## Directory Structure

```
spiro_ml/
├── configs/
│   ├── base_config.yaml              # Master config
│   ├── dataset.yaml                  # YOLO dataset spec
│   └── experiments/
│       ├── yolov11_baseline.yaml
│       └── efficientnetv2_classifier.yaml
├── datasets/
│   ├── raw/          # images/ + labels/ (your input)
│   ├── processed/    # train/ val/ test/ (generated)
│   ├── splits/       # stem index files (generated)
│   └── augmented/    # offline augmentation output
├── models/
│   ├── checkpoints/  # .pt files from training
│   ├── exports/      # .onnx files
│   └── registry/     # registry.json + versioned weight copies
├── src/lib/ml/
│   ├── core/         # config, logger, device
│   ├── data/         # DatasetManager, augmentation
│   ├── models/       # SPIRODetector, SPIROClassifier, ModelRegistry
│   ├── training/     # YOLOTrainer, EfficientNetTrainer
│   ├── evaluation/   # DetectionEvaluator, ClassificationEvaluator
│   ├── export/       # ONNXExporter
│   ├── inference/    # ONNXInferenceEngine
│   ├── continuous_learning/  # ContinuousLearningManager, DriftDetector
│   └── scripts/      # CLI entry points
├── tests/
│   ├── unit/         # Fast, no GPU, no I/O
│   └── integration/  # Full pipeline tests
├── docs/             # Architecture + API docs
├── reports/          # Generated evaluation reports
├── logs/             # TensorBoard + MLflow
└── training/         # Training run outputs (symlink-friendly)
```

---

## Dataset Engineering Pipeline

SPIRO ML includes a complete dataset engineering system for 6 waste datasets.

```bash
# Full pipeline (download → map → clean → merge → split → stats → version)
make pipeline

# Or step by step:
python training/dataset_download.py                   # Download all datasets
python training/dataset_mapper.py                     # Map to SPIRO taxonomy
python training/dataset_cleaner.py --mapped-dir datasets/mapped
python training/dataset_merger.py                     # Merge all
python training/dataset_splitter.py                   # Train/val/test split
python training/dataset_statistics.py --dir datasets/merged --charts
python training/quality_checker.py --dir datasets/merged
python training/dataset_versioning.py create          # Snapshot version
```

See `docs/DATASET_GUIDE.md` for the full pipeline documentation.

---

## Quick Start

### 1. Install

```bash
pip install -e ".[dev]"
# GPU (Linux):
pip install -e ".[gpu]"
```

### 2. Prepare Dataset

Place raw data in `datasets/raw/`:
```
datasets/raw/
  images/   ← .jpg / .png
  labels/   ← YOLO .txt  (class_id cx cy w h, normalised)
```

Run the dataset pipeline:
```bash
python -m lib.ml.scripts.dataset_manager \
    --config configs/base_config.yaml \
    --action all
```

### 3. Train

```bash
# YOLOv11 detection
python -m lib.ml.scripts.train \
    --config configs/experiments/yolov11_baseline.yaml \
    --version v1

# EfficientNetV2 classification
python -m lib.ml.scripts.train \
    --config configs/experiments/efficientnetv2_classifier.yaml \
    --architecture efficientnetv2 \
    --train-dir datasets/processed/train \
    --val-dir datasets/processed/val \
    --version v1
```

### 4. Evaluate

```bash
python -m lib.ml.scripts.evaluate \
    --config configs/base_config.yaml \
    --weights models/checkpoints/yolov11_baseline/weights/best.pt \
    --split test
```

### 5. Export ONNX

```bash
python -m lib.ml.scripts.export_onnx \
    --config configs/base_config.yaml \
    --weights models/checkpoints/yolov11_baseline/weights/best.pt \
    --architecture yolov11 \
    --verify
```

### 6. Run Inference

```bash
# Single image
python -m lib.ml.scripts.infer \
    --onnx models/exports/spiro_yolov11.onnx \
    --source path/to/image.jpg

# Directory + save annotated images
python -m lib.ml.scripts.infer \
    --onnx models/exports/spiro_yolov11.onnx \
    --source datasets/processed/test/images/ \
    --output-dir results/

# Latency benchmark
python -m lib.ml.scripts.infer \
    --onnx models/exports/spiro_yolov11.onnx \
    --benchmark --n-runs 200
```

### 7. Run Tests

```bash
# Unit tests only (fast, no GPU)
pytest tests/unit/ -m unit

# Integration tests
pytest tests/integration/ -m integration

# All tests with coverage
pytest --cov=src/lib/ml --cov-report=html
```

---

## Python API

```python
from lib.ml import ConfigManager, SPIRODetector, ONNXInferenceEngine

# Load config
cfg = ConfigManager.load("configs/base_config.yaml")

# PyTorch detector
detector = SPIRODetector(cfg, weights="models/checkpoints/best.pt")
detections = detector.predict("image.jpg")
for det in detections:
    print(det.class_name, det.confidence, det.bbox_xyxy)

# ONNX engine (production)
engine = ONNXInferenceEngine("models/exports/spiro_yolov11.onnx")
result = engine.infer("image.jpg")
print(result["detections"])
```

---

## Monitoring TensorBoard

```bash
tensorboard --logdir logs/tensorboard
```

## Monitoring MLflow

```bash
mlflow ui --backend-store-uri logs/mlflow
```

---

## Config Overrides

Override any config key at runtime:

```bash
python -m lib.ml.scripts.train \
    --config configs/base_config.yaml \
    --overrides training.epochs=200 training.batch_size=32 model.variant=yolo11s.pt
```

---

## Continuous Learning

```python
from lib.ml import ConfigManager, ONNXInferenceEngine
from lib.ml.continuous_learning import ContinuousLearningManager
from lib.ml.models import ModelRegistry

cfg = ConfigManager.load("configs/base_config.yaml")
registry = ModelRegistry()
engine = ONNXInferenceEngine("models/exports/spiro_yolov11.onnx")

def retrain_callback():
    # Your retraining logic here
    from lib.ml.training import YOLOTrainer
    trainer = YOLOTrainer(cfg)
    trainer.train()
    return trainer.validate()

clm = ContinuousLearningManager(cfg, registry, retrain_callback=retrain_callback)

# In your inference loop:
import cv2
for frame in video_stream:
    result = engine.infer(frame)
    mean_conf = sum(d["confidence"] for d in result["detections"]) / max(1, len(result["detections"]))
    clm.observe(frame, result["detections"], confidence=mean_conf)
    if clm.should_retrain():
        clm.trigger_retrain(new_version="v2", auto_promote=True)
```

---

## Integration Handoff

The integration team needs:

| Artefact | Path |
|---|---|
| ONNX model | `models/exports/spiro_*.onnx` |
| Class names | `configs/base_config.yaml → dataset.class_names` |
| Inference engine | `from lib.ml.inference import ONNXInferenceEngine` |
| Input size | 640×640 (configurable) |
| Output format | `{"type": "detection", "detections": [{bbox_xyxy, confidence, class_id, class_name}]}` |

The ONNX file embeds metadata (class names, input size, architecture) so it is self-describing.
