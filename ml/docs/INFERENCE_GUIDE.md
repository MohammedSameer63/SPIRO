# SPIRO ML — Inference Guide

## Pipeline Architecture

```
Input Image (file path | bytes | BGR ndarray)
    │
    ▼  Stage 1
┌───────────────────────────────────────────────────────┐
│  Image Validation + Preprocessing                      │
│  · EXIF auto-orientation (PIL)                        │
│  · Format & file-size validation                      │
│  · Blur detection (Laplacian variance)                │
│  · Brightness check (too dark / overexposed)          │
│  · Quality scoring (blur + brightness composite)      │
│  · Letterbox resize with aspect-ratio preservation    │
│  · Reject images below quality threshold              │
└───────────────────────────────────────────────────────┘
    │  PreprocessResult (blob, scale, pad_w, pad_h)
    ▼  Stage 2
┌───────────────────────────────────────────────────────┐
│  YOLOv11 Detection (ONNX Runtime)                     │
│  · Auto provider: TensorRT → CUDA → CPU              │
│  · Input: [1, 3, 640, 640] float32                    │
│  · Output: [1, 4+109, N] raw anchors                 │
└───────────────────────────────────────────────────────┘
    │  Stage 3: NMS + BBox decoding
    ▼
┌───────────────────────────────────────────────────────┐
│  Post-Processing                                       │
│  · Confidence threshold filter                        │
│  · Pure-NumPy NMS (IoU threshold)                     │
│  · BBox rescale to original image coordinates        │
│  · Minimum area filter                                │
│  → List[Detection]                                    │
└───────────────────────────────────────────────────────┘
    │  Stage 4
    ▼
┌───────────────────────────────────────────────────────┐
│  Object Cropping                                       │
│  · Padded bbox extraction (8% default)               │
│  · Aspect-ratio preservation                          │
│  · Minimum crop size validation                       │
│  · Pre-process for EfficientNetV2 (normalise, resize) │
└───────────────────────────────────────────────────────┘
    │  Stage 5
    ▼
┌───────────────────────────────────────────────────────┐
│  EfficientNetV2 Verification (ONNX Runtime)           │
│  · Batched inference (configurable batch size)        │
│  · Temperature scaling                                │
│  · Top-5 predictions per crop                        │
│  · Softmax probabilities                              │
└───────────────────────────────────────────────────────┘
    │  Stage 6
    ▼
┌───────────────────────────────────────────────────────┐
│  Confidence Fusion (5 methods)                        │
│  · weighted_average | geometric_mean                  │
│  · harmonic_mean | bayesian | temperature             │
│  · Agreement tracking (YOLO vs EffNet)               │
│  · Minimum confidence threshold                       │
└───────────────────────────────────────────────────────┘
    │  Stage 7
    ▼
┌───────────────────────────────────────────────────────┐
│  Waste Category Assignment                             │
│  · Maps class_id → SPIRO waste stream                │
│  · 7 streams: Organic | Recoverable | Non-Recoverable│
│    Sanitary | Hazardous | E-Waste | Reject Waste      │
└───────────────────────────────────────────────────────┘
    │  Stage 8
    ▼
┌───────────────────────────────────────────────────────┐
│  Contamination Analysis                               │
│  · Food residue on recyclables                       │
│  · Organic + hazardous mixing                        │
│  · Sanitary contamination                            │
│  · Wet recyclables                                   │
│  · E-waste with general waste                        │
│  · Contamination score (0–1) + explanation           │
└───────────────────────────────────────────────────────┘
    │  Stage 9
    ▼
┌───────────────────────────────────────────────────────┐
│  Guidance Generation                                  │
│  · Bin colour, collection point                      │
│  · Preparation steps                                 │
│  · Regulatory notes                                  │
│  · Per-class disposal instructions                   │
└───────────────────────────────────────────────────────┘
    │  Stage 10
    ▼
┌───────────────────────────────────────────────────────┐
│  Explainability                                       │
│  · Step-by-step reasoning chain                      │
│  · YOLO vs EffNet decision evidence                  │
│  · Annotated image with bbox overlay                 │
│  · Optional Grad-CAM heatmaps                        │
└───────────────────────────────────────────────────────┘
    │  Stage 11
    ▼
┌───────────────────────────────────────────────────────┐
│  Structured JSON Response                             │
│  · status, detections, contamination, guidance       │
│  · explainability, timings, model_versions, warnings │
└───────────────────────────────────────────────────────┘
```

---

## Quick Start

### Requirements
- Trained YOLO ONNX: `models/exports/spiro_yolov11s_best.onnx`
- Trained EfficientNetV2 ONNX: `models/exports/spiro_effnetv2_s_best_effnet.onnx`

### Build & run

```python
from lib.ml.pipeline import SPIROPipeline

pipeline = SPIROPipeline.from_config("configs/pipeline/inference_pipeline.yaml")
result   = pipeline.infer("photo.jpg")
print(result["detections"])
print(result["guidance"]["summary"])
print(result["contamination"]["contamination_score"])
```

### CLI

```bash
# Single image
python training/infer_pipeline.py --image photo.jpg

# Directory
python training/infer_pipeline.py --source images/ --output-dir results/

# Pretty JSON
python training/infer_pipeline.py --image photo.jpg --pretty

# Benchmark
python training/infer_pipeline.py --benchmark --n-runs 100

# Override thresholds
python training/infer_pipeline.py --image photo.jpg --conf 0.35 --fusion bayesian
```

---

## Model Loading

### Auto provider selection

The pipeline selects the fastest available runtime:

```
1. TensorrtExecutionProvider  (fastest — NVIDIA GPUs)
2. CUDAExecutionProvider      (fast    — any NVIDIA GPU)
3. CPUExecutionProvider       (fallback — always available)
```

Override in config:
```yaml
runtime:
  providers:
    - "CUDAExecutionProvider"
    - "CPUExecutionProvider"
```

### Config paths

```yaml
models:
  yolo_onnx:   "models/exports/spiro_yolov11s_best.onnx"
  effnet_onnx: "models/exports/spiro_effnetv2_s_best_effnet.onnx"
```

---

## Inference Flow

### Single image

```python
result = pipeline.infer("photo.jpg")
# or
result = pipeline.infer(bgr_array)
# or
result = pipeline.infer(open("photo.jpg", "rb").read())
```

### Batch inference

```python
results = pipeline.infer_batch(["a.jpg", "b.jpg", "c.jpg"])
```

### With request tracing

```python
result = pipeline.infer("photo.jpg", request_id="req_001")
```

---

## JSON Response Schema

```json
{
  "pipeline_version": "1.0.0",
  "status": "success",
  "timestamp": "2024-01-15T10:23:00Z",

  "image": {
    "original_size": {"width": 1920, "height": 1080},
    "quality": {
      "blur_score": 245.3,
      "brightness_mean": 127.5,
      "quality_score": 0.82,
      "is_blurry": false,
      "passed": true
    }
  },

  "detections": [
    {
      "detection_id": 0,
      "bbox_xyxy": [123.5, 45.2, 287.8, 312.1],
      "yolo_class_id": 0,
      "yolo_class_name": "plastic_bottle",
      "yolo_confidence": 0.8721,
      "effnet_class_id": 0,
      "effnet_class_name": "plastic_bottle",
      "effnet_confidence": 0.8134,
      "final_class_id": 0,
      "final_class_name": "plastic_bottle",
      "final_confidence": 0.8378,
      "fusion_method": "weighted_average",
      "models_agree": true,
      "waste_stream": "recoverable",
      "top5_predictions": [
        {"rank": 1, "class_id": 0, "class_name": "plastic_bottle", "confidence": 0.8134},
        {"rank": 2, "class_id": 1, "class_name": "plastic_bottle_cap", "confidence": 0.0821}
      ]
    }
  ],

  "detection_count": 1,

  "contamination": {
    "contamination_score": 0.0,
    "is_contaminated": false,
    "flags": [],
    "dominant_stream": "recoverable",
    "streams_detected": ["recoverable"],
    "explanation": "Dominant stream: recoverable. No contamination detected.",
    "action_required": "Dispose normally according to waste category."
  },

  "guidance": {
    "summary": "Dispose as Recoverable / Recyclable.",
    "items": [
      {
        "class_id": 0,
        "class_name": "plastic_bottle",
        "waste_stream": "Recoverable / Recyclable",
        "bin_colour": "Blue / Yellow",
        "collection_point": "Recycling bin or materials recovery facility",
        "preparation_steps": [
          "Rinse containers to remove food residue.",
          "Do not bag recyclables — place loose in bin."
        ],
        "warnings": ["Contaminated recyclables (with food) cannot be processed."],
        "regulatory_note": "Clean, dry, and sorted recyclables achieve highest recovery rates.",
        "extra_note": "Rinse plastic bottles and remove caps separately."
      }
    ]
  },

  "explainability": {
    "evidence": [
      {
        "detection_idx": 0,
        "yolo": {"class_id": 0, "class_name": "plastic_bottle", "confidence": 0.8721},
        "effnet": {"class_id": 0, "class_name": "plastic_bottle", "confidence": 0.8134,
                   "top5": [...]},
        "fusion": {"class_id": 0, "class_name": "plastic_bottle", "confidence": 0.8378,
                   "method": "weighted_average", "agreement": true},
        "waste_stream": "recoverable",
        "reasoning_chain": [
          "Step 1 — YOLOv11 Detection: identified as 'plastic_bottle' with confidence 0.872.",
          "Step 2 — EfficientNetV2 Verification: crop classified as 'plastic_bottle' with confidence 0.813.",
          "Step 3 — Agreement: both models agree on 'plastic_bottle'. High confidence decision.",
          "Step 4 — Confidence Fusion (weighted_average): final decision 'plastic_bottle' at 0.838.",
          "Step 5 — High-confidence decision. Result is reliable."
        ]
      }
    ]
  },

  "timings": {
    "preprocess_ms": 12.4,
    "detection_ms": 48.7,
    "cropping_ms": 0.8,
    "verification_ms": 22.3,
    "fusion_ms": 0.4,
    "contamination_ms": 0.2,
    "guidance_ms": 0.1,
    "explainability_ms": 1.2,
    "total_ms": 86.1
  },

  "model_versions": {
    "yolo": "spiro_yolov11s_best.onnx",
    "effnet": "spiro_effnetv2_s_best_effnet.onnx"
  },

  "warnings": []
}
```

---

## Configuration Reference

All pipeline options in `configs/pipeline/inference_pipeline.yaml`:

| Section | Key | Default | Description |
|---|---|---|---|
| models | yolo_onnx | — | Path to YOLO ONNX |
| models | effnet_onnx | — | Path to EfficientNetV2 ONNX |
| models | num_classes | 109 | SPIRO taxonomy size |
| runtime | providers | TRT→CUDA→CPU | ORT provider priority |
| runtime | effnet_batch | 8 | Crops per ORT batch |
| runtime | warmup_runs | 3 | Warmup forward passes |
| preprocessing | blur_threshold | 80.0 | Laplacian variance below this → reject |
| preprocessing | brightness_min | 20.0 | Mean pixel below this → too dark |
| preprocessing | brightness_max | 245.0 | Mean pixel above this → overexposed |
| preprocessing | quality_score_threshold | 0.3 | Composite score below this → reject |
| preprocessing | auto_orient | true | EXIF orientation correction |
| detection | conf_threshold | 0.25 | YOLO confidence cutoff |
| detection | iou_threshold | 0.45 | NMS IoU threshold |
| detection | max_detections | 100 | Maximum objects per image |
| detection | min_bbox_area | 0.001 | Minimum box area fraction |
| cropping | padding_fraction | 0.08 | Padding around each bbox |
| verification | temperature | 1.0 | Temperature scaling (1.0 = none) |
| fusion | method | weighted_average | Fusion strategy |
| fusion | yolo_weight | 0.45 | YOLO weight in fusion |
| fusion | effnet_weight | 0.55 | EfficientNetV2 weight |
| fusion | min_final_confidence | 0.25 | Below this → "uncertain" |
| contamination | enabled | true | Run contamination analysis |
| guidance | enabled | true | Generate disposal guidance |
| explainability | enabled | true | Build reasoning chains |
| explainability | gradcam | false | Grad-CAM (slower) |
| explainability | save_visualizations | false | Save annotated images |

---

## ONNX Optimisation

### Graph optimisation levels

```yaml
runtime:
  graph_optimization: "ORT_ENABLE_ALL"    # maximum (default)
  # ORT_DISABLE_ALL | ORT_ENABLE_BASIC | ORT_ENABLE_EXTENDED | ORT_ENABLE_ALL
```

### Thread tuning

```yaml
runtime:
  yolo_threads: 4     # CPU inference parallelism
  effnet_threads: 4
```

### FP16 (GPU only)

Export models with `--half`:
```bash
python training/verify_export.py \
    --config configs/verification/effnetv2_s.yaml \
    --weights path/to/best_effnet.pt --half
```

Then in pipeline config:
```yaml
runtime:
  fp16: true
```

---

## Performance

### Benchmark

```bash
python training/infer_pipeline.py --benchmark --n-runs 100
```

Expected latency (approximate):

| Hardware | YOLO | EffNet | Total pipeline |
|---|---|---|---|
| NVIDIA RTX 3080 | 6ms | 4ms | ~20ms (50 FPS) |
| NVIDIA T4 | 12ms | 8ms | ~35ms (28 FPS) |
| Intel Core i9 | 45ms | 25ms | ~120ms (8 FPS) |
| Apple M2 | 28ms | 15ms | ~70ms (14 FPS) |

### Memory usage

| Component | RAM |
|---|---|
| YOLO ORT session | ~80MB |
| EffNet ORT session | ~200MB |
| Image buffer | ~10MB |
| Total | ~350MB |

---

## Error Handling

| Error | Status | Description |
|---|---|---|
| Blurry image | `rejected` | Laplacian variance below threshold |
| Too dark | `rejected` | Mean brightness below min |
| File not found | `rejected` | Invalid path |
| Unsupported format | `rejected` | Non-image file |
| File too large | `rejected` | Exceeds max_file_size_mb |
| Model not found | `error` (startup) | ONNX file missing |
| ORT crash | `error` | Runtime error during inference |
| No detections | `partial` | Image valid, no objects found |

### Graceful degradation

- If EfficientNetV2 ONNX is unavailable → pipeline uses YOLO-only with `fusion_method: yolo_only`
- If a crop is too small → skipped (verification result = None, YOLO class used directly)
- If contamination engine errors → skipped, warning added to response

---

## Deployment

### Python API integration

```python
from lib.ml.pipeline import SPIROPipeline

# Initialise once at startup
pipeline = SPIROPipeline.from_config("configs/pipeline/inference_pipeline.yaml")

# Per-request
def process_image(image_bytes: bytes) -> dict:
    result = pipeline.infer(image_bytes, request_id="req_123")
    return result
```

### Output summary extraction

```python
result = pipeline.infer("photo.jpg")

# Top detection
if result["detections"]:
    top = result["detections"][0]
    print(f"Detected: {top['final_class_name']} ({top['final_confidence']:.2%})")
    print(f"Stream:   {top['waste_stream']}")

# Guidance
print(result["guidance"]["summary"])

# Contamination
cont = result["contamination"]
if cont["is_contaminated"]:
    print(f"⚠ Contamination score: {cont['contamination_score']:.2f}")
    print(cont["action_required"])

# Performance
print(f"Total: {result['timings']['total_ms']:.1f}ms")
```

---

## Troubleshooting

**`FileNotFoundError: YOLO ONNX not found`**
→ Train YOLOv11 and export: `make train-small && make export-best`

**`FileNotFoundError: EfficientNetV2 ONNX not found`**
→ Train EffNet and export: `make train-effnet && make export-effnet`

**All images rejected as blurry**
→ Lower `preprocessing.blur_threshold` in config (default 80.0)

**ORT crashes with CUDA**
→ Set `runtime.providers: [CPUExecutionProvider]` or update CUDA drivers

**Low detection confidence**
→ Lower `detection.conf_threshold` (default 0.25)

**Slow inference on CPU**
→ Increase `runtime.yolo_threads` and `runtime.effnet_threads`; try `graph_optimization: ORT_ENABLE_ALL`
