# SPIRO ML — Model Registry Guide

## Overview

The ModelRegistry stores every trained model version with its full metadata,
metrics, artefact paths, and lifecycle stage.

## Storage Layout

```
mlops/models/
  registry.json          ← master index (model_id → version → stage/path)
  yolov11s/
    v1.0.0/
      metadata.json      ← full ModelMetadata record
    v2.0.0/
      metadata.json
  effnetv2_s/
    v1.0.0/
      metadata.json
```

## Lifecycle Stages

```
candidate → staging → production → archived
```

- **candidate**: just registered, not validated
- **staging**: validated, awaiting comparison
- **production**: currently serving traffic
- **archived**: superseded by newer production model

## CLI Commands

```bash
# Register a model
python training/mlops/model_metadata.py register \
    --model-id yolov11s --version v1.0.0 \
    --architecture yolo11s --task detection \
    --onnx models/exports/best.onnx --map50-95 0.72

# List all models
python training/mlops/model_registry.py list

# Show specific version
python training/mlops/model_registry.py show --model-id yolov11s --version v1.0.0

# Promote to production
python training/mlops/model_registry.py promote \
    --model-id yolov11s --version v1.0.0 --stage production

# Export full registry
python training/mlops/model_registry.py export
```

## Python API

```python
from lib.ml.mlops import ModelRegistry, ModelMetadata, ModelMetrics

registry = ModelRegistry()

# Register
meta = ModelMetadata(
    model_id="yolov11s", version="v2.0.0",
    architecture="yolo11s", task="detection",
    onnx_path="models/exports/best.onnx",
    dataset_version_id="v1.0.1",
    experiment_id="exp_042",
    metrics=ModelMetrics(mAP50_95=0.75, mAP50=0.88),
)
registry.register(meta)

# Query
prod = registry.get_production_model("yolov11s")
all_versions = registry.get_all_versions("yolov11s")

# Promote
registry.promote("yolov11s", "v2.0.0", "production", approved_by="ci_pipeline")

# Compare
v1 = registry.get("yolov11s", "v1.0.0")
v2 = registry.get("yolov11s", "v2.0.0")
print(v2.is_better_than(v1, primary_metric="mAP50_95"))  # True
```

## ModelMetadata Fields

| Field | Type | Description |
|---|---|---|
| model_id | str | Model family (e.g. "yolov11s") |
| version | str | Semantic version (e.g. "v2.0.0") |
| architecture | str | Backbone name |
| task | str | "detection" or "verification" |
| onnx_path | str | Path to .onnx file |
| pytorch_path | str | Path to .pt file |
| onnx_sha256 | str | Auto-computed on creation |
| dataset_version_id | str | Linked dataset version |
| experiment_id | str | Linked experiment record |
| metrics.mAP50_95 | float | Primary detection metric |
| metrics.top1_accuracy | float | Primary classification metric |
| metrics.latency_cpu_ms | float | CPU inference latency |
| stage | str | candidate/staging/production/archived |
| approved | bool | Whether validated by ModelPromoter |
| deployed_at | str | ISO timestamp of production deployment |
