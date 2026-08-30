# SPIRO ML — System Operation Guide

## Day 1: Initial Setup

```bash
# 1. Install
pip install -e ".[dev]"

# 2. Download and process datasets
make pipeline

# 3. Train YOLOv11
make train-small   # yolo11s, ~12h on RTX 3080

# 4. Export ONNX
make export-best

# 5. Train EfficientNetV2
make train-effnet

# 6. Export EfficientNetV2 ONNX
make export-effnet

# 7. Register both models
python training/mlops/training_pipeline.py \
    --model-id yolov11s --version v1.0.0 \
    --onnx models/exports/spiro_yolov11s_best.onnx \
    --map50-95 0.72 --dataset-version v1.0.0

# 8. Deploy
python training/mlops/model_deployer.py deploy \
    --model-id yolov11s --version v1.0.0 \
    --onnx models/exports/spiro_yolov11s_best.onnx \
    --strategy blue_green --task detection

# 9. Verify
python scripts/healthcheck.py

# 10. Run inference
python training/infer_pipeline.py --image test.jpg --pretty
```

## Daily Operations

```bash
# Check registry status
make mlops-registry

# Run drift detection (requires saved baseline)
make mlops-drift MODEL=yolov11s BASELINE=b.npy CURRENT=c.npy

# Export performance report
make mlops-performance

# Check scheduled retraining queue
python training/mlops/retraining_scheduler.py list

# Run pending retraining jobs
python training/mlops/retraining_scheduler.py run-pending
```

## Model Upgrade Cycle

```bash
# 1. Train new candidate
python training/trainer.py --variant s --name spiro_yolov11s_v2

# 2. Register + validate + compare + promote
python training/mlops/training_pipeline.py \
    --model-id yolov11s --version v2.0.0 \
    --onnx models/exports/new_best.onnx \
    --map50-95 0.75 --auto-promote

# 3. Monitor
make mlops-performance
```

## Continuous Learning Cycle

```bash
# Collect new images from production traffic
python training/mlops/continuous_learning.py collect \
    --source /var/log/spiro/inference_images/

# Validate quality
python training/mlops/continuous_learning.py validate

# Run CL cycle (snapshot + queue retrain)
python training/mlops/continuous_learning.py run-cycle

# Export cycle report
python training/mlops/continuous_learning.py report
```

## Emergency Rollback

```bash
python training/mlops/rollback.py --task detection --reason "P95 > 500ms"
python training/mlops/rollback.py --task verification --reason "accuracy drop"
```

## Generate All Reports

```bash
make mlops-reports
```

Generates:
- `mlops/reports/model_registry.json`
- `mlops/reports/dataset_versions.json`
- `mlops/reports/training_history.json`
- `mlops/reports/deployment_history.json`
- `mlops/reports/performance_report.json`
- `mlops/reports/continuous_learning_report.json`
