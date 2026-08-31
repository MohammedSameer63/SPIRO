# SPIRO ML — Changelog

## [1.0.0] — Production Release (Prompt 8 Audit)

### Fixed
- **CRITICAL**: Taxonomy YAML parsing error — missing spaces after `:` in `infrastructure` and `multi_material` group entries
- **CRITICAL**: Broken import `lib.ml.verification.verify_metrics` in `verify_trainer.py` — corrected to `lib.ml.verification.evaluation.verify_metrics`
- **SECURITY**: `torch.load()` in `efficientnet_model.py` missing `weights_only=False` explicit flag
- **CORRECTNESS**: Guidance engine `_NON_RECOVERABLE_IDS` missing 9 class IDs (71, 72, 75, 77, 78, 96–99) — now covers all 109 classes

### Added
- `docs/FINAL_ARCHITECTURE.md` — complete system diagram and module map
- `docs/TROUBLESHOOTING.md` — diagnostic guide for all common errors
- `docs/CONFIG_REFERENCE.md` — complete parameter reference for all configs
- `docs/KNOWN_LIMITATIONS.md` — documented model, fusion, and infrastructure constraints
- `docs/CHANGELOG.md` — this file
- `docs/RELEASE_NOTES.md` — production release summary
- `docs/ML_SYSTEM_GUIDE.md` — end-to-end system operation guide
- `docs/MODEL_REGISTRY_GUIDE.md` — model lifecycle management guide
- `docs/API_REFERENCE.md` (updated) — Python API documentation
- `docs/BENCHMARK_RESULTS.md` — expected performance benchmarks
- `reports/audit_report.md` — static analysis audit results

### Verified (audit pass)
- All 174 Python source files parse without syntax errors
- All 35 test files parse without syntax errors  
- All 26 YAML config files parse correctly (taxonomy and supplementary mapping fixed)
- No circular imports detected
- No duplicate top-level function/class names
- All required CLI scripts present (33/33)
- All required documentation files present (23/23)
- 109 classes verified in taxonomy (IDs 0–108, no duplicates)
- Guidance engine now covers all 109 classes (was 100/109)
- All contamination engine class IDs within valid range (0–108)
- All environment configs have `num_classes: 109`

---

## [0.7.0] — Production Hardening (Prompt 7)

- ONNX optimization pipeline (graph, FP16, INT8-dynamic, INT8-static)
- Full benchmarking suite (latency, FPS, batch throughput, load test, concurrent)
- Robustness testing (12 image degradation categories)
- Accuracy validation (FP32 vs FP16 vs INT8 vs PyTorch)
- Docker infrastructure (CPU, GPU, training, docker-compose)
- GitHub Actions CI/CD (ci.yml, benchmark.yml, deployment.yml)
- Environment-specific configs (production, staging, development, testing)
- Container healthcheck script
- DEPLOYMENT_GUIDE.md, PRODUCTION_GUIDE.md, BENCHMARK_GUIDE.md

## [0.6.0] — MLOps Platform (Prompt 6)

- ModelRegistry, DatasetRegistry, ExperimentTracker
- ModelValidator (10 automated pre-deployment checks)
- ModelComparator (JSON + Markdown comparison reports)
- ModelDeployer (immediate, blue-green, canary, rollback)
- ModelPromoter (automated policy gate)
- DriftDetector (PSI + KS test for input/prediction drift)
- ConceptDriftDetector (Page-Hinkley + CUSUM + Entropy)
- MetricsStore (append-only JSONL time-series)
- PerformanceMonitor (rolling window, background thread)
- RetrainingScheduler (daily/weekly/monthly/drift-triggered)
- ContinuousLearningManager (collect, validate, snapshot, queue, promote)
- 17 MLOps CLI scripts in training/mlops/
- MLOPS_GUIDE.md

## [0.5.0] — Production Inference Pipeline (Prompt 5)

- 11-stage SPIROPipeline orchestrator
- ImagePreprocessor (EXIF, blur detection, letterbox)
- YOLODetectionEngine (pure-NumPy NMS, provider auto-selection)
- ObjectCropper (padded, validated)
- ContaminationEngine (7 scenarios, 0–1 score)
- GuidanceEngine (7 streams, per-class instructions)
- PipelineExplainer (reasoning chain, annotated image)
- OutputBuilder (structured JSON schema)
- INFERENCE_GUIDE.md

## [0.4.0] — EfficientNetV2 Verification (Prompt 4)

- VerifierModel (7 timm variants: b0–b3, s, m, l)
- VerifyDataset (YOLO + crop modes, weighted sampler)
- VerifyTrainer (2-phase: frozen→full, AMP, grad accumulation)
- VerifyMetrics (Top-1/5, F1, ROC-AUC, confusion matrix)
- VerifyExporter (ONNX + ORT verification)
- VerifyInferenceEngine (ORT batched + temperature calibration)
- ConfidenceFusion (5 methods: WA, GM, HM, Bayesian, Temperature)
- GradCAM + GradCAM++ explainability
- EFFICIENTNET_GUIDE.md

## [0.3.0] — YOLOv11 Training Pipeline (Prompt 3)

- TrainingConfig with `_base_` YAML inheritance
- SPIROYOLOTrainer (full orchestrator)
- 5 model variant configs (yolo11n/s/m/l/x)
- TensorBoardCallback, CSVLoggerCallback, EarlyStoppingCallback
- CheckpointManager (save_best/last/periodic, auto-resume)
- ExperimentTracker (MLflow)
- ResultsPlotter (loss/mAP/PR/dashboard)
- HyperparameterSweep (grid + random)
- TRAINING_GUIDE.md

## [0.2.0] — Dataset Engineering (Prompt 2)

- DatasetOrchestrator (10-step pipeline)
- 6 dataset downloaders (TACO, TrashNet, ZeroWaste, OpenLitterMap, Kaggle GC, MJU-Waste)
- COCOMapper, ClassificationFolderMapper, OLMMapper, YOLOMapper
- DataCleaner (MD5 + dHash dedup, corrupt detection, bbox validation)
- DatasetMerger, DatasetSplitter (stratified)
- DatasetStatistics, DatasetVersioning, DatasetVisualizer
- 6 dataset→SPIRO class mapping YAMLs
- DATASET_GUIDE.md

## [0.1.0] — ML Foundation (Prompt 1)

- ConfigManager, Logger, Device auto-detection
- DatasetManager, OfflineAugmentor (Albumentations)
- SPIRODetector (YOLOv11 wrapper)
- SPIROClassifier (EfficientNetV2 wrapper)
- ModelRegistry (JSON versioning)
- YOLOTrainer, EfficientNetTrainer
- DetectionEvaluator, ClassificationEvaluator
- ONNXExporter (opset 17, metadata)
- ONNXInferenceEngine (pure-NumPy NMS)
- ContinuousLearningManager (Page-Hinkley drift)
- 109-class SPIRO taxonomy (22 groups)
- Full test suite
