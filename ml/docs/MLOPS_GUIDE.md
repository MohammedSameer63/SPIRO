# SPIRO ML — MLOps Guide

## Architecture Overview

```
Training Complete
     │
     ▼
┌─────────────────────────────────────────────────────────────┐
│  Experiment Tracker                                          │
│  · Record parameters, metrics, hardware                     │
│  · MLflow or JSON-backed store                              │
└─────────────────────────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────────────────────────┐
│  Dataset Registry                                           │
│  · Version every dataset snapshot                           │
│  · Track lineage, checksums, class distribution            │
└─────────────────────────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────────────────────────┐
│  Model Registry                                             │
│  · Register every trained version                           │
│  · Store metrics, paths, hyperparameters                    │
│  · Lifecycle: candidate → staging → production → archived   │
└─────────────────────────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────────────────────────┐
│  Model Validator                                            │
│  · ONNX graph check                                         │
│  · ORT inference test                                       │
│  · Output shape / class count (109)                        │
│  · Taxonomy consistency                                     │
│  · CPU latency check                                        │
│  · Memory check                                             │
│  · Regression test (deterministic)                         │
└─────────────────────────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────────────────────────┐
│  Model Comparator                                           │
│  · Head-to-head: candidate vs production                    │
│  · Metrics delta, latency delta, size delta                 │
│  · Recommendation: promote | reject | manual_review        │
└─────────────────────────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────────────────────────┐
│  Model Promoter                                             │
│  · Automated policy gate                                    │
│  · Calls validator + comparator                             │
│  · Promotes to staging → production                        │
└─────────────────────────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────────────────────────┐
│  Model Deployer                                             │
│  · Immediate / Blue-Green / Canary deployment              │
│  · Rollback to previous version                             │
│  · Deployment history (JSON)                               │
└─────────────────────────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────────────────────────┐
│  Monitoring + Drift Detection                               │
│  · Performance: CPU/GPU/RAM/latency/FPS                     │
│  · Input drift (PSI + KS test)                             │
│  · Prediction drift (class distribution)                   │
│  · Concept drift (Page-Hinkley + CUSUM + Entropy)          │
│  · Triggers retraining when drift detected                  │
└─────────────────────────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────────────────────────┐
│  Continuous Learning                                        │
│  · Collect new images                                       │
│  · Validate quality (blur, brightness)                      │
│  · Snapshot dataset version                                 │
│  · Queue retraining job                                     │
│  · Auto-promote if improved                                 │
└─────────────────────────────────────────────────────────────┘
```

---

## Storage Layout

```
mlops/
├── models/
│   ├── registry.json                    ← master model index
│   └── <model_id>/<version>/
│       └── metadata.json               ← full ModelMetadata record
│
├── datasets/
│   ├── registry.json
│   └── <dataset_id>/<version>/
│       └── dataset_version.json
│
├── experiments/
│   ├── index.json
│   ├── metrics/<model_id>/<metric>.jsonl  ← append-only time-series
│   └── <model_id>/<experiment_id>.json
│
├── deployments/
│   └── deployment_history.json
│
├── drift/
│   ├── drift_<model_id>_<ts>.json
│   └── concept_drift_<model_id>_<ts>.json
│
├── staging/                             ← new images awaiting review
├── approved/                            ← quality-approved images
├── retraining_queue.json
└── cl_records.json

mlops/reports/
├── model_registry.json
├── dataset_versions.json
├── training_history.json
├── deployment_history.json
├── performance_report.json
├── drift_report.json
├── validation/<model>_<version>_validation.json
├── promotions/<model>_<version>_promotion.json
└── comparison_<model>_<version>.json

models/serve/
├── yolo_active.onnx                     ← current production YOLO
└── effnet_active.onnx                   ← current production EffNet
```

---

## Quick Start

### 1. Register a trained model

```bash
# After training completes:
python training/mlops/model_metadata.py register \
    --model-id yolov11s --version v1.0.0 \
    --architecture yolo11s --task detection \
    --onnx models/exports/spiro_yolov11s_best.onnx \
    --pytorch models/checkpoints/spiro_yolov11s/weights/best.pt \
    --map50-95 0.72 --map50 0.85 \
    --dataset-version-id v1.0.0 \
    --experiment-id exp_001 \
    --num-classes 109 --input-size 640
```

### 2. Full training pipeline (register + validate + promote)

```bash
python training/mlops/training_pipeline.py \
    --model-id yolov11s --version v1.0.0 \
    --config configs/training/yolo11s.yaml \
    --onnx models/exports/spiro_yolov11s_best.onnx \
    --map50-95 0.72 \
    --auto-promote
```

### 3. Validate a model

```bash
python training/mlops/model_validator.py \
    --onnx models/exports/spiro_yolov11s_best.onnx \
    --model-id yolov11s --version v1.0.0 \
    --task detection --input-size 640
```

### 4. Deploy

```bash
# Blue-Green deployment
python training/mlops/model_deployer.py deploy \
    --model-id yolov11s --version v1.0.0 \
    --onnx models/exports/spiro_yolov11s_best.onnx \
    --strategy blue_green --task detection

# Immediate deployment
python training/mlops/model_deployer.py deploy \
    --model-id yolov11s --version v1.0.0 \
    --onnx models/exports/spiro_yolov11s_best.onnx \
    --strategy immediate

# Canary (10% traffic)
python training/mlops/model_deployer.py deploy \
    --model-id yolov11s --version v1.0.0 \
    --onnx models/exports/best.onnx \
    --strategy canary --canary-pct 10
```

### 5. Rollback

```bash
python training/mlops/rollback.py --task detection --reason "accuracy drop"
```

---

## Dataset Versioning

```bash
# Snapshot current dataset
python training/mlops/dataset_registry.py snapshot \
    --dataset-id spiro_v1 \
    --root datasets/processed \
    --description "Added 500 new litter images"

# List versions
python training/mlops/dataset_registry.py list

# Export registry
python training/mlops/dataset_registry.py export
```

```python
from lib.ml.mlops import DatasetRegistry, DatasetVersion

registry = DatasetRegistry()

# Snapshot from disk (auto-counts images + annotations)
dv = registry.snapshot_from_disk(
    dataset_id="spiro_v1",
    dataset_root=Path("datasets/processed"),
    description="CL cycle 3",
    parent_version="v1.0.0",
)
registry.register(dv)

# Get lineage
chain = dv.lineage_chain(registry)
for ancestor in chain:
    print(ancestor.version, ancestor.image_count)
```

---

## Model Registry

```bash
python training/mlops/model_registry.py list
python training/mlops/model_registry.py show --model-id yolov11s --version v1.0.0
python training/mlops/model_registry.py promote --model-id yolov11s --version v1.0.0 --stage production
python training/mlops/model_registry.py export
```

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

# Get production model
prod = registry.get_production_model("yolov11s")
# Compare two versions
v1 = registry.get("yolov11s", "v1.0.0")
v2 = registry.get("yolov11s", "v2.0.0")
print(v2.is_better_than(v1))  # True
```

---

## Model Validation

```python
from lib.ml.mlops import ModelValidator

validator = ModelValidator(
    min_accuracy=0.60,
    max_latency_ms=200.0,
    max_memory_mb=1024.0,
)
report = validator.validate(
    onnx_path="models/exports/best.onnx",
    model_id="yolov11s",
    version="v2.0.0",
    input_size=640,
    num_classes=109,
    task="detection",
)
print(report.overall_passed, report.summary)
# Checks: file_exists, onnx_graph, ort_loads, inference_executes,
#         output_shape, taxonomy_consistency, latency_cpu,
#         memory_usage, accuracy_threshold, regression_deterministic
```

---

## Model Comparison

```python
from lib.ml.mlops import ModelComparator, ModelRegistry

registry   = ModelRegistry()
comparator = ModelComparator(primary_metric="mAP50_95", min_improvement=0.002)

candidate  = registry.get("yolov11s", "v2.0.0")
production = registry.get_production_model("yolov11s")

report = comparator.compare(candidate, production)
print(report.recommendation)  # "promote" or "reject"
for r in report.reasons:
    print(f"  • {r}")
# Saves JSON + Markdown comparison report
```

---

## Automated Promotion

```python
from lib.ml.mlops import ModelPromoter

promoter = ModelPromoter(
    auto_deploy=True,
    primary_metric="mAP50_95",
    min_improvement=0.002,
    deployment_strategy="blue_green",
)
decision = promoter.evaluate_and_promote(
    model_id="yolov11s",
    candidate_version="v2.0.0",
    task="detection",
    deployed_by="ci_pipeline",
)
print(decision.approved, decision.deployed)
```

---

## Drift Detection

```python
from lib.ml.mlops import DriftDetector, ConceptDriftDetector
import numpy as np

# Input / prediction drift
detector = DriftDetector()
detector.set_baseline(
    confidences=baseline_confidences,    # [N] array
    class_ids=baseline_class_ids,        # [N] array
    blur_scores=baseline_blur_scores,    # [N] array
)
report = detector.detect(
    confidences=current_confidences,
    class_ids=current_class_ids,
    model_id="yolov11s",
    version="v2.0.0",
)
print(report.overall_drift_score, report.overall_severity)
# → 0.05 → "none"  OR  0.35 → "high"

# Concept drift (Page-Hinkley + CUSUM)
cd = ConceptDriftDetector()
cd.set_baseline(baseline_confidences)
events = cd.update_batch(current_confidences, model_id="yolov11s")
if events:
    # Trigger retraining
    scheduler.schedule_drift_triggered("yolov11s", drift_score=0.35)
```

---

## Performance Monitoring

```python
from lib.ml.mlops import PerformanceMonitor

monitor = PerformanceMonitor(
    window_size=500,
    slow_threshold_ms=300.0,
)
# Record from SPIROPipeline response
monitor.record_from_pipeline_result("req_001", pipeline_result)

# Or manually
monitor.record_request(
    request_id="req_002",
    preprocess_ms=12.0,
    detection_ms=48.0,
    verification_ms=22.0,
    total_ms=90.0,
    detections=3,
)

report = monitor.report()
print(report["fps"], report["latency"]["p95_ms"])
monitor.export_performance_report()
```

---

## Scheduled Retraining

```bash
# Schedule weekly retraining
python training/mlops/retraining_scheduler.py schedule \
    --model-id yolov11s --schedule weekly

# Run pending jobs now
python training/mlops/retraining_scheduler.py run-pending

# List pending
python training/mlops/retraining_scheduler.py list
```

```python
from lib.ml.mlops import RetrainingScheduler

scheduler = RetrainingScheduler()

# Weekly schedule
scheduler.schedule("yolov11s", schedule="weekly")

# Drift-triggered
scheduler.schedule_drift_triggered("yolov11s", drift_score=0.35)

# Start background worker (polls every 60s)
scheduler.start_background_worker(poll_interval_s=60.0)
```

---

## Continuous Learning

```bash
# Collect new images
python training/mlops/continuous_learning.py collect \
    --source incoming_images/ --model-id yolov11s

# Validate quality
python training/mlops/continuous_learning.py validate

# Run full CL cycle
python training/mlops/continuous_learning.py run-cycle \
    --model-id yolov11s --force

# Export report
python training/mlops/continuous_learning.py report
```

```python
from lib.ml.mlops import ContinuousLearningManager

clm = ContinuousLearningManager(
    model_id="yolov11s",
    min_new_samples=50,
    blur_threshold=30.0,
    auto_promote=True,
)
clm.collect_from_directory(Path("incoming_images/"))
clm.validate_staged_samples()
cycle = clm.run_cycle()
print(cycle.samples_approved, cycle.dataset_version, cycle.retraining_job_id)
```

---

## Reports Generated

| File | Contents |
|---|---|
| `mlops/reports/model_registry.json` | All registered model versions with metrics |
| `mlops/reports/dataset_versions.json` | Dataset version history with checksums |
| `mlops/reports/training_history.json` | Experiment records with parameters and metrics |
| `mlops/reports/deployment_history.json` | All deployments, strategies, rollbacks |
| `mlops/reports/performance_report.json` | Latency stats, FPS, error rates |
| `mlops/reports/drift_report.json` | Drift events per model |
| `mlops/reports/validation/*.json` | Per-model validation check results |
| `mlops/reports/promotions/*.json` | Promotion decision records |
| `mlops/reports/comparison_*.json` | Head-to-head model comparisons |
| `mlops/reports/comparison_*.md` | Human-readable Markdown comparison |

---

## Rollback

Automatic rollback is triggered when:
- Validation fails during deployment
- Inference crashes in production
- Drift score exceeds critical threshold
- Manual rollback requested

```python
from lib.ml.mlops import ModelDeployer

deployer = ModelDeployer()
record = deployer.rollback(task="detection", reason="accuracy below threshold")
```

```bash
python training/mlops/rollback.py --task detection --reason "latency exceeded 500ms"
```

---

## Experiment Tracking

```python
from lib.ml.mlops import ExperimentTracker

tracker = ExperimentTracker(
    experiments_dir=Path("mlops/experiments"),
    mlflow_tracking_uri="http://localhost:5000",  # optional
)
exp_id = tracker.start_run(
    "yolov11s_epoch300", "yolov11s", "v2.0.0", "dataset_v1.0.1",
    hyperparameters={"lr": 0.01, "epochs": 300, "batch": 16},
)
tracker.log_metrics(exp_id, {"mAP50": 0.85, "val_loss": 0.43}, step=150)
tracker.log_artefact(exp_id, onnx_path="models/exports/best.onnx")
tracker.end_run(exp_id, status="completed")
```

---

## Makefile Targets

```bash
make mlops-registry        # Export model registry report
make mlops-validate        # Validate production models
make mlops-drift           # Run drift detection
make mlops-performance     # Export performance report
make mlops-cl-cycle        # Run CL cycle
make mlops-retrain-weekly  # Schedule weekly retraining
make mlops-reports         # Generate all reports
```
