# SPIRO EfficientNetV2 Verification Guide

## Architecture Overview

The SPIRO verification pipeline is a **two-stage detection system**:

```
Image
 │
 ▼
YOLOv11 Detection  ─────────────────────────────────────────────┐
 │  (primary detector, 109 SPIRO classes)                        │
 │  bbox_xyxy, confidence, class_id                             │
 ▼                                                               │
Crop each detected object                                        │
 │  (padded bounding box patch)                                  │
 ▼                                                               │
EfficientNetV2 Verification  ───────────────────────────────────┤
 │  (second-stage classifier, 109 SPIRO classes)                 │
 │  probabilities[109], top-5 predictions                       │
 ▼                                                               │
Confidence Fusion  ◄────────────────────────────────────────────┘
 │  (weighted_average | geometric_mean | harmonic_mean |
 │   bayesian | temperature scaling)
 ▼
Final SPIRO Waste Category + Confidence
 │
 ▼
Grad-CAM Explainability (optional)
```

EfficientNetV2 does **not** replace YOLOv11 — it verifies and refines each
detected object's classification, reducing false positives from the detector.

---

## Supported Variants

| Variant | timm name | Input | Params | Use Case |
|---|---|---|---|---|
| b0 | tf_efficientnetv2_b0 | 192px | ~7M | Fastest — edge/mobile |
| b1 | tf_efficientnetv2_b1 | 240px | ~8M | Mobile |
| b2 | tf_efficientnetv2_b2 | 260px | ~10M | Compact |
| b3 | tf_efficientnetv2_b3 | 300px | ~14M | Balanced |
| **s** | tf_efficientnetv2_s | 300px | ~22M | **Recommended default** |
| m | tf_efficientnetv2_m | 384px | ~54M | High accuracy |
| l | tf_efficientnetv2_l | 480px | ~119M | Maximum accuracy |

---

## Training

### Quick start

```bash
# Train with recommended variant (s)
python training/train_effnet.py --variant s

# Specify variant
python training/train_effnet.py --variant b0   # fastest
python training/train_effnet.py --variant m    # high accuracy

# With overrides
python training/train_effnet.py --variant s \
    --epochs 50 --batch 32 --lr 0.001 --loss focal

# Full config path
python training/train_effnet.py --config configs/verification/effnetv2_m.yaml
```

### Two-phase training

Phase 1 (frozen backbone → head only, fast):

```yaml
training:
  phase1_epochs: 10  # head-only
  phase2_epochs: 40  # full fine-tune
```

Phase 1 uses `optimizer.lr_head` (default 0.005, higher).
Phase 2 uses `optimizer.lr0` (default 0.001, lower) with full backbone unfreeze.

### Crop extraction (optional but recommended)

Extract per-class crop patches from YOLOv11-labeled splits for targeted training:

```bash
python training/train_effnet.py --variant s --extract-crops
```

This calls `CropExtractor` to create `datasets/crops/<split>/<class_name>/` directories.

### Resume training

```bash
python training/train_effnet.py --variant s --resume
python training/train_effnet.py --variant s --resume \
    --checkpoint models/verification_checkpoints/spiro_effnetv2_s/last_effnet.pt
```

---

## Config Options

All options in `configs/verification/effnetv2_verify.yaml`.

### Key parameters

| Section | Key | Default | Description |
|---|---|---|---|
| model | variant | efficientnetv2_s | Backbone variant |
| model | pretrained | true | ImageNet pretrained weights |
| model | drop_rate | 0.2 | Classifier dropout |
| training | phase1_epochs | 10 | Frozen backbone epochs |
| training | phase2_epochs | 40 | Fine-tune epochs |
| training | amp | true | Automatic Mixed Precision |
| training | patience | 15 | Early stopping patience |
| optimizer | name | AdamW | AdamW \| Adam \| SGD |
| optimizer | lr0 | 0.001 | Learning rate (phase 2) |
| optimizer | lr_head | 0.005 | Learning rate (phase 1 head) |
| scheduler | name | cosine | cosine \| step \| plateau |
| loss | name | label_smoothing | cross_entropy \| label_smoothing \| focal \| weighted |
| loss | label_smoothing | 0.1 | Smoothing factor |
| loss | focal_gamma | 2.0 | Focal loss gamma |
| dataset | use_weighted_sampler | true | Handle class imbalance |
| augmentation | horizontal_flip | 0.5 | |
| augmentation | random_crop | true | RandomResizedCrop |
| augmentation | clahe_prob | 0.2 | CLAHE contrast enhancement |
| augmentation | cutout_prob | 0.3 | Cutout occlusion |
| augmentation | random_erasing_prob | 0.2 | Random erasing |
| export | auto_export_onnx | true | Export ONNX after training |
| fusion | method | weighted_average | Fusion strategy |
| fusion | yolo_weight | 0.4 | YOLOv11 weight in fusion |
| fusion | effnet_weight | 0.6 | EfficientNetV2 weight |

---

## Loss Functions

| Loss | When to use |
|---|---|
| `cross_entropy` | Balanced dataset, no special requirements |
| `label_smoothing` | Default — prevents overconfidence, smoother gradients |
| `focal` | Severe class imbalance (many rare classes) |
| `weighted` | Auto-computed class frequency weights |

---

## Augmentation Pipeline

Training augmentation (Albumentations + torchvision):

**Geometric**: RandomResizedCrop, HorizontalFlip, Rotation ±20°, Perspective distortion

**Color**: ColorJitter (brightness/contrast/saturation/hue), CLAHE, Grayscale

**Noise / Blur**: GaussianBlur, GaussNoise

**Dropout / Erasing**: CoarseDropout (Cutout), RandomErasing

---

## Inference

### Python API (PyTorch)

```python
from lib.ml.verification import VerifierModel, VerifyConfig

cfg = VerifyConfig.load("configs/verification/effnetv2_s.yaml")
model = VerifierModel.load(cfg, "models/verification_checkpoints/spiro_effnetv2_s/best_effnet.pt")

result = model.predict("crop.jpg")
print(result.class_name, result.confidence)
print(result.top5_names)

results = model.predict_batch(["crop1.jpg", "crop2.jpg"])
```

### ONNX inference (production)

```python
from lib.ml.verification.inference import VerifyInferenceEngine

engine = VerifyInferenceEngine(
    onnx_path="models/exports/spiro_effnetv2_s_best.onnx",
    input_size=300,
    temperature=1.0,
)

# Single image
result = engine.verify("crop.jpg")
result, elapsed_ms = engine.verify("crop.jpg", return_timing=True)

# Batch
results = engine.verify_batch(["a.jpg", "b.jpg"])

# Verify YOLO detections
enriched = engine.verify_detections(image_bgr, yolo_detections)
```

### CLI

```bash
# Single image
python training/verify_inference.py \
    --onnx models/exports/spiro_effnetv2_s_best.onnx \
    --source crop.jpg

# Two-stage pipeline
python training/verify_inference.py \
    --onnx models/exports/spiro_effnetv2_s_best.onnx \
    --yolo-onnx models/exports/spiro_yolov11s_best.onnx \
    --source image.jpg --fusion weighted_average

# Benchmark
python training/verify_inference.py \
    --onnx models/exports/spiro_effnetv2_s_best.onnx \
    --benchmark --n-runs 200
```

---

## Confidence Fusion

Fuses YOLOv11 and EfficientNetV2 probability vectors.

```python
from lib.ml.verification.fusion import ConfidenceFusion

fusion = ConfidenceFusion(
    method="weighted_average",
    yolo_weight=0.4,
    effnet_weight=0.6,
)

result = fusion.fuse(yolo_detection_dict, effnet_result_dict)
print(result.class_name, result.fused_confidence, result.agreement)
```

### Methods

| Method | Formula | Best for |
|---|---|---|
| `weighted_average` | α·P(YOLO) + β·P(EffNet) | General use |
| `geometric_mean` | P(YOLO)^α × P(EffNet)^β (normalised) | Sharp disagreements |
| `harmonic_mean` | 2·P·Q/(P+Q) | Penalising low scores |
| `bayesian` | prior × P(YOLO) × P(EffNet) | When class priors known |
| `temperature` | softmax((α·log P + β·log Q) / T) | Calibrated outputs |

### Temperature calibration

```bash
python training/verify_inference.py \
    --onnx models/exports/spiro_effnetv2_s_best.onnx \
    --calibrate --calibration-dir datasets/processed/val
```

---

## ONNX Export

```bash
# Auto-export after training (default: on)
python training/train_effnet.py --variant s  # exports automatically

# Manual export
python training/verify_export.py \
    --config configs/verification/effnetv2_s.yaml \
    --weights models/verification_checkpoints/spiro_effnetv2_s/best_effnet.pt

# With benchmark
python training/verify_export.py \
    --config configs/verification/effnetv2_s.yaml \
    --weights path/to/best_effnet.pt \
    --benchmark --n-runs 200
```

Verification checks numerical parity between PyTorch and ORT (max diff < 0.001).

---

## Explainability

### Grad-CAM

```python
from lib.ml.verification import VerifierModel, VerifyConfig, GradCAM

cfg = VerifyConfig.load("configs/verification/effnetv2_s.yaml")
model = VerifierModel.load(cfg, "path/to/best_effnet.pt")

cam = GradCAM(model, method="gradcam")  # or "gradcam++"
heatmap, overlay, explanation = cam.explain("crop.jpg")

# Save all outputs
paths = cam.save_explanation("crop.jpg", output_dir="reports/gradcam")
# → original.jpg, heatmap.jpg, overlay.jpg, explanation.json

cam.remove_hooks()
```

### CLI

```bash
python training/verify_model.py explain \
    --weights models/verification_checkpoints/spiro_effnetv2_s/best_effnet.pt \
    --config configs/verification/effnetv2_s.yaml \
    --image datasets/crops/train/plastic_bottle/img_0001.jpg \
    --output-dir reports/gradcam \
    --method gradcam
```

### Explanation JSON structure

```json
{
  "class_id": 0,
  "class_name": "plastic_bottle",
  "confidence": 0.87,
  "top5_predictions": [
    {"rank": 1, "class_id": 0, "class_name": "plastic_bottle", "confidence": 0.87},
    ...
  ],
  "top_contributing_regions": [
    {"center_x": 145, "center_y": 82, "score": 0.94, "bbox": [129, 66, 161, 98]},
    ...
  ],
  "cam_method": "gradcam"
}
```

---

## Evaluation

```bash
python training/verify_metrics.py \
    --onnx models/exports/spiro_effnetv2_s_best.onnx \
    --config configs/verification/effnetv2_s.yaml \
    --split test --plots
```

Generates: accuracy_curve.png, loss_curve.png, confusion_matrix.png,
per_class_accuracy.png, roc_curves.png, classification_report.json

---

## Expected Outputs

```
models/
└── verification_checkpoints/
    └── spiro_effnetv2_s/
        ├── best_effnet.pt
        ├── last_effnet.pt
        └── training_summary.json

models/exports/
└── spiro_effnetv2_s_best_effnet.onnx

logs/verification/
└── training_history.csv

reports/verification/
├── test_classification_report.json
├── test_confusion_matrix.png
├── test_per_class_accuracy.png
├── test_roc_curves.png
├── accuracy_curve.png
└── loss_curve.png
```

---

## Deployment Integration

```python
# Production two-stage pipeline
from lib.ml.inference import ONNXInferenceEngine
from lib.ml.verification.inference import VerifyInferenceEngine
from lib.ml.verification.fusion import ConfidenceFusion

detector = ONNXInferenceEngine("models/exports/spiro_yolov11s_best.onnx")
verifier = VerifyInferenceEngine("models/exports/spiro_effnetv2_s_best.onnx")
fusion   = ConfidenceFusion(method="weighted_average", yolo_weight=0.4, effnet_weight=0.6)

# Per frame:
det_result = detector.infer(frame_bgr)
enriched   = verifier.verify_detections(frame_bgr, det_result["detections"])
for det in enriched:
    if det["verification"]:
        fr = fusion.fuse(det, det["verification"])
        print(fr.class_name, fr.fused_confidence, "agree:", fr.agreement)
```
