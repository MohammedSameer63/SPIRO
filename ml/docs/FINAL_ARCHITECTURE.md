# SPIRO ML — Final Architecture

## System Overview

SPIRO ML is a production-grade waste detection and classification subsystem comprising 385 files across 7 build increments. It integrates two neural networks in a sequential verification pipeline, a full MLOps lifecycle platform, and automated deployment infrastructure.

## Module Map

```
spiro_ml/
│
├── src/lib/ml/
│   ├── core/              Config, Logger, Device — shared infrastructure
│   ├── data/              DatasetManager, Augmentation
│   ├── models/            SPIRODetector (YOLOv11), SPIROClassifier (EfficientNetV2), Registry
│   ├── training/          YOLOTrainer, VerifyTrainer, CheckpointManager, Callbacks, Plotter
│   ├── evaluation/        DetectionEvaluator, ClassificationEvaluator
│   ├── export/            ONNXExporter
│   ├── inference/         ONNXInferenceEngine
│   ├── dataset_engineering/ Downloaders, Mappers, Cleaners, Splitters, Stats, Versioning
│   │
│   ├── verification/      EfficientNetV2 second-stage verifier
│   │   ├── model/         VerifierModel — timm EfficientNetV2
│   │   ├── training/      VerifyTrainer, VerifyDataset, Losses
│   │   ├── evaluation/    VerifyMetrics (Top-1/5, F1, ROC-AUC, Confusion)
│   │   ├── export/        VerifyExporter (ONNX + ORT verification)
│   │   ├── inference/     VerifyInferenceEngine (ORT batched)
│   │   ├── fusion/        ConfidenceFusion (5 methods)
│   │   └── explainability/ GradCAM, GradCAM++
│   │
│   ├── pipeline/          Full 11-stage production inference pipeline
│   │   ├── preprocessing/ ImagePreprocessor (EXIF, blur, letterbox)
│   │   ├── detection/     YOLODetectionEngine (NMS, bbox decode)
│   │   ├── cropping/      ObjectCropper
│   │   ├── contamination/ ContaminationEngine (7 scenarios)
│   │   ├── guidance/      GuidanceEngine (7 streams, 109 classes)
│   │   ├── explainability/ PipelineExplainer (reasoning chain)
│   │   ├── output/        OutputBuilder (structured JSON)
│   │   └── spiro_pipeline.py  SPIROPipeline orchestrator
│   │
│   ├── mlops/             Full MLOps lifecycle platform
│   │   ├── registry/      ModelRegistry, DatasetRegistry, ExperimentTracker
│   │   ├── validation/    ModelValidator (10 checks), ModelComparator
│   │   ├── deployment/    ModelDeployer (4 strategies), ModelPromoter
│   │   ├── drift/         DriftDetector (PSI+KS), ConceptDriftDetector (PH+CUSUM)
│   │   ├── monitoring/    MetricsStore (JSONL), PerformanceMonitor
│   │   └── continuous_learning/ RetrainingScheduler, ContinuousLearningManager
│   │
│   ├── optimization/      ONNXOptimizer (graph, FP16, INT8-dynamic, INT8-static)
│   ├── benchmarking/      PipelineBenchmarker, OnnxSessionBenchmarker
│   └── robustness/        RobustnessTester (12 categories), AccuracyValidator
│
├── training/              33 CLI entry points
│   ├── *.py               YOLO + EffNet training scripts
│   └── mlops/             17 MLOps management scripts
│
├── configs/
│   ├── taxonomy/          spiro_taxonomy.yaml — 109 classes, 22 groups
│   ├── mappings/          6 dataset→SPIRO class mappings
│   ├── training/          6 YOLOv11 variant configs
│   ├── verification/      4 EfficientNetV2 variant configs
│   ├── pipeline/          Production inference config
│   ├── production/        Production environment overrides
│   ├── staging/           Staging overrides
│   ├── development/       Development overrides (all gates open)
│   └── testing/           Test overrides (conf=0.001, no quality gates)
│
├── tests/                 35 test files, ~250 test methods
│   ├── unit/              28 unit test modules
│   └── integration/       7 integration test modules
│
├── docker/                Dockerfile.cpu, .gpu, .training, docker-compose.yml
├── .github/workflows/     ci.yml, benchmark.yml, deployment.yml
├── scripts/               healthcheck.py, optimize.py, benchmark.py
├── deployment/            deployment_checklist.md
├── docs/                  11 documentation files
└── mlops/                 Runtime: models/, datasets/, experiments/, drift/, deployments/
```

## Two-Stage Inference Pipeline

```
Input Image
    │ Stage 1 — ImagePreprocessor
    │  • EXIF auto-orient  • Blur detection (Laplacian)
    │  • Brightness check  • Quality score  • Letterbox
    ↓
YOLO ONNX (TensorRT → CUDA → CPU)
    │ Stage 2-3 — YOLODetectionEngine
    │  • Anchor decode  • NMS  • BBox rescale to original
    ↓
ObjectCropper — padded patches
    │ Stage 4
    │  • Padding 8%  • Min-size validation  • Normalise
    ↓
EffNet ONNX — batched crop verification
    │ Stage 5 — VerifyInferenceEngine
    │  • Top-5 predictions  • Temperature scaling
    ↓
ConfidenceFusion
    │ Stage 6 — 5 strategies
    │  • Weighted average (default)  • Geometric/Harmonic mean
    │  • Bayesian  • Temperature scaling
    ↓
WasteStream + ContaminationEngine + GuidanceEngine
    │ Stages 7-9 — 109 classes → 7 streams
    ↓
PipelineExplainer → OutputBuilder
    │ Stages 10-11 — JSON response
    ↓
Structured JSON (status, detections, contamination, guidance, timings)
```

## MLOps Lifecycle

```
Training Complete
    → ExperimentTracker.start_run()
    → ModelRegistry.register()
    → ModelValidator (10 checks)
    → ModelComparator (vs production)
    → ModelPromoter (policy gate)
    → ModelDeployer (blue_green/canary/immediate/rollback)
    → PerformanceMonitor + MetricsStore
    → DriftDetector + ConceptDriftDetector
    → RetrainingScheduler (daily/weekly/monthly/drift-triggered)
    → ContinuousLearningManager
    → [loop back to Training]
```

## Key Constants

| Constant | Value |
|---|---|
| SPIRO classes | **109** (IDs 0–108) |
| Waste groups | 22 |
| Waste streams | 7 |
| ONNX opset | 17 |
| YOLOv11 input | 640×640 |
| EffNetV2-S input | 300×300 |
| NMS IoU threshold | 0.45 (default) |
| Detection conf | 0.25–0.35 (env-dependent) |
| Fusion method | weighted_average (default) |
| Registry storage | mlops/models/registry.json |
