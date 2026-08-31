# SPIRO ML — Architecture

## System Diagram

```
┌─────────────────────────────────────────────────────────┐
│                    SPIRO ML Package                      │
│                                                         │
│  ┌───────────┐    ┌──────────────┐    ┌─────────────┐  │
│  │  Dataset   │───▶│  Training    │───▶│  Evaluation │  │
│  │  Manager   │    │  (YOLO /     │    │  (mAP, PR,  │  │
│  │            │    │  EffNetV2)   │    │  confusion) │  │
│  └───────────┘    └──────┬───────┘    └──────┬──────┘  │
│       │                  │                   │          │
│       │ augment          │ best.pt           │          │
│       ▼                  ▼                   ▼          │
│  ┌──────────┐    ┌──────────────┐    ┌─────────────┐  │
│  │Augmentor │    │ ONNX Export  │    │   Model     │  │
│  │(Albumen- │    │ (opset 17,   │    │  Registry   │  │
│  │ tations) │    │  simplify,   │    │  (JSON)     │  │
│  └──────────┘    │  metadata)   │    └──────┬──────┘  │
│                  └──────┬───────┘           │          │
│                         │ .onnx             │ promote  │
│                         ▼                   ▼          │
│                  ┌──────────────┐    ┌─────────────┐  │
│                  │    ONNX      │    │ Continuous  │  │
│                  │  Inference   │───▶│  Learning   │  │
│                  │  Engine      │    │  Manager    │  │
│                  └──────────────┘    └─────────────┘  │
│                                                         │
│  Logging: TensorBoard │ MLflow │ loguru                 │
└─────────────────────────────────────────────────────────┘
         │
         ▼  Integration surface
    ┌─────────────────────────────────────────┐
    │  Backend API team                        │
    │  - ONNXInferenceEngine (Python)          │
    │  - .onnx file (any ORT-compatible lang)  │
    │  - JSON result format                    │
    └─────────────────────────────────────────┘
```

---

## Component Responsibilities

### `core/`
- **ConfigManager** — YAML loading with `_base_` inheritance, dot-access, runtime overrides, seed management.
- **Logger** — loguru-based structured logging to console + rotating file.
- **Device** — torch.device resolution (CUDA / MPS / CPU auto-detect).

### `data/`
- **DatasetManager** — validates YOLO annotations, letterbox-preprocesses images, creates stratified splits, computes statistics, patches `dataset.yaml`.
- **OfflineAugmentor** — multiplies the training split N× using Albumentations pipelines before training starts.
- **build_train_transform / build_val_transform** — returns composable Albumentations pipelines.

### `models/`
- **SPIRODetector** — wraps `ultralytics.YOLO`. Exposes `predict()`, `predict_batch()`, `draw()`. Clean `Detection` dataclass output.
- **SPIROClassifier** — `nn.Module` around `timm` EfficientNetV2. Two-phase fine-tuning (freeze → unfreeze). Clean `ClassificationResult` output.
- **ModelRegistry** — file-based versioned model registry. Supports register, promote, delete, best_version queries.

### `training/`
- **YOLOTrainer** — wraps `YOLO.train()` with TensorBoard + MLflow integration, config flattening for param logging, best-checkpoint tracking.
- **EfficientNetTrainer** — full PyTorch loop with AMP, label smoothing, cosine/step/plateau schedulers, two-phase training, early stopping.

### `evaluation/`
- **DetectionEvaluator** — runs `YOLO.val()`, generates per-class AP bar chart, JSON report.
- **ClassificationEvaluator** — full sklearn suite: accuracy, classification report, ROC-AUC, confusion matrix PNG, PR curve PNG, JSON report.

### `export/`
- **ONNXExporter** — YOLO via Ultralytics export API; EfficientNet via `torch.onnx.export`. Post-export onnxsim simplification, metadata embedding, ORT verification.

### `inference/`
- **ONNXInferenceEngine** — production ORT engine. Letterbox preprocess, YOLO output decode (supports `[N, 4+nc]` and `[4+nc, N]` layouts), pure-NumPy NMS, classification softmax branch, warmup, benchmark.

### `continuous_learning/`
- **DriftDetector** — Page-Hinkley test on confidence stream. Configurable delta + lambda.
- **NewSampleBuffer** — persists live frames + pseudo-labels as YOLO data.
- **ContinuousLearningManager** — observes stream, checks retrain conditions, merges buffer into dataset, runs callback, registers + promotes.

---

## Data Flow

### Training Pipeline
```
Raw images + labels
       ↓
  DatasetManager.validate_raw()          # annotation integrity check
       ↓
  DatasetManager.preprocess()            # letterbox → 640×640 JPEG
       ↓
  DatasetManager.create_splits()         # stratified 70/15/15
       ↓
  OfflineAugmentor.augment_split()       # 3× training data
       ↓
  YOLOTrainer.train()                    # Ultralytics YOLO.train()
       ↓
  YOLOTrainer.validate()                 # mAP50, precision, recall
       ↓
  ONNXExporter.export_yolo()             # → .onnx opset 17
       ↓
  ONNXExporter.verify()                  # ORT shape + numerical check
       ↓
  ModelRegistry.register()              # versioned entry
  ModelRegistry.promote()               # → production
```

### Inference Pipeline
```
Image (file / BGR array)
       ↓
  ONNXInferenceEngine._preprocess()     # letterbox + NCHW + /255
       ↓
  ORT session.run()                     # GPU or CPU
       ↓
  ONNXInferenceEngine._decode_yolo()    # → xyxy + class + conf
       ↓
  _nms()                               # pure NumPy NMS
       ↓
  {"type": "detection", "detections": [...]}
```

---

## Model Variants

| Variant | Size | mAP50 (COCO) | Params | Use Case |
|---|---|---|---|---|
| yolo11n.pt | Nano | ~39.5 | 2.6M | Edge / realtime |
| yolo11s.pt | Small | ~47.0 | 9.4M | Balanced |
| yolo11m.pt | Medium | ~51.5 | 20.1M | High accuracy |
| yolo11l.pt | Large | ~53.4 | 25.3M | Max accuracy |
| tf_efficientnetv2_s | S | ~84.9% (IN) | 21.5M | Classification |

---

## Configuration Inheritance

```yaml
# configs/experiments/my_experiment.yaml
_base_: "../base_config.yaml"

experiment:
  name: "my_experiment"

training:
  epochs: 200          # overrides base
  batch_size: 32       # overrides base
  # all other keys inherited from base
```

`ConfigManager` deep-merges the experiment YAML over the base using OmegaConf.

---

## Output Formats

### Detection result (inference)
```json
{
  "type": "detection",
  "detections": [
    {
      "bbox_xyxy": [x1, y1, x2, y2],
      "confidence": 0.87,
      "class_id": 0,
      "class_name": "object_a"
    }
  ]
}
```

### Classification result (inference)
```json
{
  "type": "classification",
  "class_id": 2,
  "class_name": "object_c",
  "confidence": 0.94,
  "probabilities": [0.02, 0.04, 0.94, 0.00, 0.00]
}
```

### Evaluation report (JSON)
```json
{
  "split": "test",
  "mAP50": 0.874,
  "mAP50-95": 0.621,
  "precision": 0.891,
  "recall": 0.843,
  "per_class": {
    "object_a": {"AP50": 0.912},
    "object_b": {"AP50": 0.856}
  }
}
```
