# SPIRO ML Integration Guide

## Overview

This document describes how to integrate the SPIRO ML pipeline into the backend for automatic waste classification.

## Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   Citizen App   │────▶│   Backend API    │────▶│  ML Inference   │
│  (Submit Report)│     │ (Create Report)  │     │  (SPIROPipeline)│
└─────────────────┘     └──────────────────┘     └─────────────────┘
                               │                         │
                               ▼                         ▼
                        ┌──────────────────┐     ┌─────────────────┐
                        │   PostgreSQL     │◀────│  Save Prediction│
                        │  (WasteReport +  │     │  (Prediction)   │
                        │   Prediction)    │     └─────────────────┘
                        └──────────────────┘
```

## Required Models

The ML pipeline requires two ONNX models:

1. **YOLOv11 Detection Model** (`spiro_yolov11s_best.onnx`)
   - Detects waste objects in images
   - Input: 640x640 RGB image
   - Output: Bounding boxes + class probabilities

2. **EfficientNetV2 Classification Model** (`spiro_effnetv2_s_best_effnet.onnx`)
   - Verifies/Classifies detected objects
   - Input: 300x300 RGB crops
   - Output: 109-class probabilities

## Model Training & Export

### 1. Prepare Training Data

```bash
cd ml
# Download and prepare datasets
python -m lib.ml.scripts.dataset_manager download --datasets taco,trashnet,kaggle_gc
python -m lib.ml.scripts.dataset_manager map --taxonomy configs/taxonomy/spiro_taxonomy.yaml
python -m lib.ml.scripts.dataset_manager split --train 0.7 --val 0.15 --test 0.15
```

### 2. Train YOLOv11 Detection Model

```bash
cd ml
# Train YOLOv11s on SPIRO taxonomy
python -m lib.ml.scripts.train \
    --config configs/training/yolov11_training.yaml \
    --model yolov11s \
    --epochs 100 \
    --batch-size 32

# Export to ONNX
python -m lib.ml.scripts.export_onnx \
    --weights runs/train/yolov11s_best.pt \
    --output models/exports/spiro_yolov11s_best.onnx \
    --input-size 640
```

### 3. Train EfficientNetV2 Classification Model

```bash
cd ml
# Train EfficientNetV2-S on cropped detections
python -m lib.ml.scripts.train \
    --config configs/training/efficientnetv2_classifier.yaml \
    --model effnetv2_s \
    --epochs 50 \
    --batch-size 64

# Export to ONNX
python -m lib.ml.scripts.export_onnx \
    --weights runs/train/effnetv2_s_best.pth \
    --output models/exports/spiro_effnetv2_s_best_effnet.onnx \
    --input-size 300 \
    --model-type classification
```

### 4. Verify Exports

```bash
cd ml
python -m lib.ml.scripts.infer \
    --model models/exports/spiro_yolov11s_best.onnx \
    --source test_image.jpg

python -m lib.ml.scripts.infer \
    --model models/exports/spiro_effnetv2_s_best_effnet.onnx \
    --source test_crop.jpg \
    --model-type classification
```

### 5. Copy Models to Backend

```bash
# From project root
cp ml/models/exports/spiro_yolov11s_best.onnx backend/models/exports/
cp ml/models/exports/spiro_effnetv2_s_best_effnet.onnx backend/models/exports/
cp ml/configs/pipeline/inference_pipeline.yaml backend/configs/pipeline/
```

## Backend Configuration

### Environment Variables (.env)

```env
# ML Settings
ML_ENABLED=true
ML_MODELS_DIR=models
YOLO_ONNX_PATH=models/exports/spiro_yolov11s_best.onnx
EFFNET_ONNX_PATH=models/exports/spiro_effnetv2_s_best_effnet.onnx
PIPELINE_CONFIG_PATH=configs/pipeline/inference_pipeline.yaml
ML_CONFIDENCE_THRESHOLD=0.25
```

### Install ML Dependencies

```bash
cd backend
pip install -r requirements.txt
# This includes: onnxruntime, opencv-python-headless, numpy, omegaconf, pyyaml, loguru
```

## API Endpoints

### Internal ML Service Endpoints

```
POST   /api/v1/internal/predictions              # Save prediction (ML service)
POST   /api/v1/internal/predictions/infer/{id}   # Run inference for report
GET    /api/v1/internal/predictions/report/{id}  # Get prediction for report
```

### Frontend Integration

The prediction is automatically created when a citizen submits a report. The frontend can display predictions via:

```
GET /api/v1/reports/{report_id}  # Returns report with prediction data
```

## Database Schema

The `predictions` table stores ML results:

```sql
CREATE TABLE predictions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    report_id UUID NOT NULL REFERENCES waste_reports(id),
    category_id UUID NOT NULL REFERENCES waste_categories(id),
    confidence DECIMAL(5,2),  -- 0-100
    model_used VARCHAR(255),
    prediction_time TIMESTAMP DEFAULT NOW()
);
```

## Customization

### Category Mapping

The `_map_class_to_category` method in `waste_report_service.py` maps ML taxonomy classes (0-108) to database `waste_category` entries. Customize this based on your categories:

```python
def _map_class_to_category(self, class_id: int) -> UUID:
    # Implement your mapping logic
    # Options:
    # 1. Direct name matching (current)
    # 2. Lookup table in database
    # 3. Configuration file mapping
    pass
```

### Waste Streams

The ML pipeline returns waste streams (organic, recoverable, non_recoverable, hazardous, sanitary, e_waste, reject). Map these to your collection schedules.

## Troubleshooting

### ML Pipeline Not Initializing

1. Check model files exist at configured paths
2. Verify `onnxruntime` is installed: `pip list | grep onnx`
3. Check logs for specific errors

### Inference Errors

1. Image format: Must be readable by OpenCV (jpg, png, etc.)
2. Image size: Max 50MB (configurable in pipeline config)
3. CUDA: If using GPU, ensure `onnxruntime-gpu` is installed

### Performance

- First inference includes model loading (~2-3 seconds)
- Subsequent inferences: ~100-300ms per image
- Use batch inference for multiple images

## Monitoring

Check ML inference logs in backend output:
```
ML prediction saved for report {id}: class=plastic_bottle, confidence=0.92
```

## Future Improvements

1. **Async Processing**: Use Celery/RQ for background inference
2. **Model Versioning**: Track which model version generated each prediction
3. **Feedback Loop**: Collect worker corrections for continuous learning
4. **A/B Testing**: Compare model versions in production