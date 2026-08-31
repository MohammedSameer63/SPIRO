# SPIRO ML — Deployment Guide

## Pre-Deployment Checklist

```
□ YOLOv11 ONNX exports successfully (opset 17)
□ EfficientNetV2 ONNX exports successfully (opset 17)
□ ModelValidator passes all 10 checks for both models
□ 109-class taxonomy consistency verified
□ Latency within threshold (detection < 200ms CPU)
□ Memory usage < 2GB per model
□ Regression test passes (deterministic output)
□ Numerical parity with PyTorch (max diff < 0.01)
□ Model registered in ModelRegistry with full metadata
□ Dataset version linked in metadata
□ Comparison against current production completed
□ Promotion decision saved to mlops/reports/promotions/
□ serve/ directory contains yolo_active.onnx + effnet_active.onnx
□ configs/production/pipeline.yaml points to serve/ models
□ Docker healthcheck passes (scripts/healthcheck.py)
```

---

## Step-by-Step Deployment

### 1. Train and export models

```bash
python training/trainer.py --variant s --epochs 300
python training/experiment.py export --name spiro_yolov11s
python training/train_effnet.py --variant s --epochs 50
python training/verify_export.py \
    --config configs/verification/effnetv2_s.yaml \
    --weights models/verification_checkpoints/spiro_effnetv2_s/best_effnet.pt
```

### 2. Register in ModelRegistry

```bash
python training/mlops/training_pipeline.py \
    --model-id yolov11s --version v1.0.0 \
    --config configs/training/yolo11s.yaml \
    --onnx models/exports/spiro_yolov11s_best.onnx \
    --map50-95 0.72 --dataset-version v1.0.0

python training/mlops/training_pipeline.py \
    --model-id effnetv2_s --version v1.0.0 \
    --architecture efficientnetv2_s --task verification \
    --onnx models/exports/spiro_effnetv2_s_best_effnet.onnx \
    --top1-acc 0.84 --dataset-version v1.0.0
```

### 3. Validate

```bash
python training/mlops/model_validator.py \
    --onnx models/exports/spiro_yolov11s_best.onnx \
    --model-id yolov11s --version v1.0.0 --task detection

python training/mlops/model_validator.py \
    --onnx models/exports/spiro_effnetv2_s_best_effnet.onnx \
    --model-id effnetv2_s --version v1.0.0 \
    --task verification --input-size 300
```

### 4. Optimize (recommended)

```bash
python scripts/optimize.py \
    --onnx models/exports/spiro_yolov11s_best.onnx \
    --model-id yolov11s --input-size 640

python scripts/optimize.py \
    --onnx models/exports/spiro_effnetv2_s_best_effnet.onnx \
    --model-id effnetv2_s --input-size 300
```

### 5. Deploy

```bash
python training/mlops/model_deployer.py deploy \
    --model-id yolov11s --version v1.0.0 \
    --onnx models/exports/spiro_yolov11s_best.onnx \
    --strategy blue_green --task detection

python training/mlops/model_deployer.py deploy \
    --model-id effnetv2_s --version v1.0.0 \
    --onnx models/exports/spiro_effnetv2_s_best_effnet.onnx \
    --strategy blue_green --task verification
```

### 6. Verify

```bash
python training/mlops/model_deployer.py active --task detection
python training/mlops/model_deployer.py active --task verification
python scripts/healthcheck.py
```

### 7. Integration

```python
from lib.ml.pipeline import SPIROPipeline

# Production
pipeline = SPIROPipeline.from_config("configs/production/pipeline.yaml")
result = pipeline.infer("photo.jpg")
```

---

## Deployment Strategies

| Strategy | Description | Zero Downtime | Risk |
|---|---|---|---|
| `immediate` | Atomic ONNX file swap | Yes (same FS) | Medium |
| `blue_green` | Stage in green slot, verify, then swap | Yes | Low |
| `canary` | Gradual traffic shift (e.g. 10% → 100%) | Yes | Lowest |
| `rollback` | Restore previous deployment | Yes | N/A |

---

## Rollback

```bash
# Emergency rollback
python training/mlops/rollback.py --task detection --reason "accuracy drop"
python training/mlops/rollback.py --task verification

# Programmatic
from lib.ml.mlops import ModelDeployer
deployer = ModelDeployer()
deployer.rollback(task="detection", reason="P95 latency exceeded 500ms")
```

---

## Docker Deployment

```bash
# CPU production image
docker build -f docker/Dockerfile.cpu -t spiro-ml-cpu:latest .
docker run --rm -v $(pwd)/models:/app/models:ro spiro-ml-cpu:latest

# GPU production image
docker build -f docker/Dockerfile.gpu -t spiro-ml-gpu:latest .
docker run --gpus all --rm -v $(pwd)/models:/app/models:ro spiro-ml-gpu:latest

# Full stack (CPU + MLflow + TensorBoard)
cd docker && docker-compose up spiro-cpu mlflow tensorboard

# GPU stack
cd docker && docker-compose --profile gpu up spiro-gpu
```

---

## Active Model Locations

```
models/serve/
  yolo_active.onnx     → current production YOLOv11
  effnet_active.onnx   → current production EfficientNetV2
  yolo_green.onnx      → (blue-green staging slot)
  yolo_canary.onnx     → (canary slot)
```

---

## Post-Deployment Monitoring

```bash
python training/mlops/drift_detector.py \
    --model-id yolov11s \
    --baseline-confidences mlops/baselines/baseline.npy \
    --current-confidences mlops/current/current.npy

python training/mlops/performance_monitor.py export
python training/mlops/model_registry.py list
make mlops-reports
```
