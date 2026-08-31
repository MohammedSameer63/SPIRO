# SPIRO ML — API Reference

## `lib.ml.core.config.ConfigManager`

### `ConfigManager.load(config_path, overrides=None)`
Load a YAML config with optional `_base_` inheritance.

```python
cfg = ConfigManager.load("configs/base_config.yaml")
cfg = ConfigManager.load("configs/experiments/yolov11_baseline.yaml",
                          overrides={"training.epochs": 200})
```

**Parameters**
- `config_path` — `str | Path` to YAML file
- `overrides` — `dict` of dot-notation keys to override, e.g. `{"training.epochs": 50}`

**Returns** `ConfigManager` instance (also stored as singleton).

### `ConfigManager.get()`
Return the current singleton instance. Raises `RuntimeError` if not initialised.

### `cfg.as_dict()` → `dict`
### `cfg.to_yaml()` → `str`
### `cfg.resolve_path(key, root=None)` → `Path`

---

## `lib.ml.data.DatasetManager`

```python
dm = DatasetManager(cfg)
dm.validate_raw()           # raises ValueError on bad data
dm.preprocess()             # letterbox → processed/
dm.create_splits()          # → dict[str, list[str]]
dm.analyze()                # → dict with per-split stats
dm.export_stats("out.json")
dm.update_dataset_yaml()
```

---

## `lib.ml.data.OfflineAugmentor`

```python
aug = OfflineAugmentor(cfg)
n_written = aug.augment_split("datasets/processed/train", factor=3)
```

---

## `lib.ml.models.SPIRODetector`

```python
detector = SPIRODetector(cfg)
detector = SPIRODetector(cfg, weights="models/checkpoints/best.pt")
detector = SPIRODetector.from_checkpoint(cfg, "models/checkpoints/v2.pt")

detections: List[Detection] = detector.predict("image.jpg")
detections = detector.predict(bgr_array)
batch: List[List[Detection]] = detector.predict_batch(["a.jpg", "b.jpg"])
canvas = detector.draw(bgr_array, detections)
detector.save("models/my_model.pt")
info = detector.info()   # → dict
```

### `Detection`
```python
det.bbox_xyxy   # Tuple[float, float, float, float]
det.confidence  # float
det.class_id    # int
det.class_name  # str
det.to_dict()   # → dict
```

---

## `lib.ml.models.SPIROClassifier`

```python
clf = SPIROClassifier(cfg)
clf = SPIROClassifier(cfg, freeze_backbone=True)

result: ClassificationResult = clf.predict("image.jpg")
result = clf.predict(bgr_array)
result = clf.predict(pil_image)
results: List[ClassificationResult] = clf.predict_batch([...])

clf.freeze_backbone()
clf.unfreeze_backbone(unfreeze_last_n_blocks=3)
clf.unfreeze_all()
clf.save("path/to/weights.pt")
clf2 = SPIROClassifier.load(cfg, "path/to/weights.pt")
counts = clf.parameter_count()   # → {"total": N, "trainable": M}
```

### `ClassificationResult`
```python
r.class_id       # int
r.class_name     # str
r.confidence     # float
r.probabilities  # List[float]  — sums to 1.0
r.to_dict()
```

---

## `lib.ml.models.ModelRegistry`

```python
registry = ModelRegistry("models/registry")

entry = registry.register(
    version="v1",
    architecture="yolov11",
    weights_path="models/checkpoints/best.pt",
    metrics={"mAP50": 0.87},
    onnx_path="models/exports/model.onnx",   # optional
    tags=["baseline"],
    notes="First production run",
    copy_weights=True,
)

registry.promote("v1")
prod = registry.get_production()    # → dict | None
entry = registry.get("v1")
versions = registry.list_versions() # → List[dict]
best = registry.best_version("mAP50")
registry.delete("v1", remove_files=False)
```

---

## `lib.ml.training.YOLOTrainer`

```python
trainer = YOLOTrainer(cfg)
results = trainer.train()
metrics = trainer.validate()        # → {"mAP50": ..., "precision": ..., ...}
path = trainer.save_best()          # → Path
trainer.best_weights                # → Path | None
```

---

## `lib.ml.training.EfficientNetTrainer`

```python
trainer = EfficientNetTrainer(cfg)
history = trainer.train(
    train_dir="datasets/processed/train",
    val_dir="datasets/processed/val",
    two_phase=True,
)
# history = {"train_loss": [...], "train_acc": [...], "val_loss": [...], "val_acc": [...]}
trainer.best_checkpoint             # → Path | None
trainer._best_val_acc               # → float
```

---

## `lib.ml.evaluation.DetectionEvaluator`

```python
ev = DetectionEvaluator(cfg)
report = ev.evaluate(
    weights_path="models/checkpoints/best.pt",
    split="test",
    save_report=True,
)
# report = {"mAP50": ..., "mAP50-95": ..., "precision": ..., "recall": ..., "per_class": {...}}
```

---

## `lib.ml.evaluation.ClassificationEvaluator`

```python
ev = ClassificationEvaluator(cfg)
result = ev.evaluate(
    y_true=[0, 1, 2, ...],
    y_pred=[0, 1, 1, ...],
    y_proba=[[0.9, 0.05, 0.05], ...],   # optional
    save_report=True,
)
# result = {"accuracy": ..., "roc_auc_macro": ..., "classification_report": {...}}
```

---

## `lib.ml.export.ONNXExporter`

```python
exp = ONNXExporter(cfg)

# YOLO
onnx_path = exp.export_yolo("models/checkpoints/best.pt", output_name="model.onnx")

# EfficientNet
onnx_path = exp.export_efficientnet(model, output_name="clf.onnx",
                                    input_shape=(1, 3, 224, 224))

# Verify
ok = exp.verify(onnx_path, input_shape=(1, 3, 640, 640), torch_model=model)
```

---

## `lib.ml.inference.ONNXInferenceEngine`

```python
engine = ONNXInferenceEngine(
    onnx_path="models/exports/model.onnx",
    providers=None,           # auto-detect
    input_size=(640, 640),
    conf_threshold=0.4,
    iou_threshold=0.45,
    warmup_runs=3,
    class_names=None,         # read from ONNX metadata
)

result = engine.infer("image.jpg")
result, elapsed_ms = engine.infer(bgr_array, return_timing=True)
results = engine.infer_batch(["a.jpg", "b.jpg"])
stats = engine.benchmark(n_runs=100)
```

---

## `lib.ml.continuous_learning.ContinuousLearningManager`

```python
clm = ContinuousLearningManager(
    cfg=cfg,
    registry=registry,
    retrain_callback=my_retrain_fn,   # Callable[[], Dict[str, float]]
)

clm.observe(image_bgr, detections, confidence=0.85, buffer_sample=True)
if clm.should_retrain():
    entry = clm.trigger_retrain(new_version="v2", auto_promote=True)

summary = clm.summary()
# → {"observations": N, "buffered_samples": M, "drift_detected": bool, ...}
```

---

## `lib.ml.continuous_learning.DriftDetector`

```python
det = DriftDetector(delta=0.005, lambda_=50.0, window=500)
drifted: bool = det.update(value)
det.reset()
det.n_observations   # int
```

---

## CLI Commands

```bash
# Dataset pipeline
spiro-dataset --config configs/base_config.yaml --action all

# Training
spiro-train --config configs/experiments/yolov11_baseline.yaml --version v1

# Evaluation
spiro-eval --config configs/base_config.yaml \
           --weights models/checkpoints/best.pt --split test

# ONNX export
spiro-export --config configs/base_config.yaml \
             --weights models/checkpoints/best.pt \
             --architecture yolov11 --verify

# Inference
spiro-infer --onnx models/exports/spiro_yolov11.onnx \
            --source images/ --output-dir results/

# Benchmark
spiro-infer --onnx models/exports/spiro_yolov11.onnx \
            --benchmark --n-runs 200
```
