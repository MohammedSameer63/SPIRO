# SPIRO ML v1.0.0 — Production Release Notes

**Release date:** 2026-08-09  
**Build:** Prompts 1–8 (complete audit and hardening)  
**Files:** 395 total | **Python source:** 174 files  
**Tests:** 35 test files | ~250 test methods  
**Coverage:** Core pipeline, MLOps, Optimization, Robustness

---

## What's Included

### Two-Stage Neural Network Pipeline
- **YOLOv11** (5 variants: n/s/m/l/x) — primary object detector
- **EfficientNetV2** (7 variants: b0–b3, s, m, l) — second-stage crop verifier
- **5-method confidence fusion** — weighted average, geometric mean, harmonic mean, Bayesian, temperature scaling
- **109-class SPIRO taxonomy** — 22 waste groups, full guidance for every class

### Production Inference
- 11-stage `SPIROPipeline` orchestrator
- EXIF-aware image preprocessing with quality gating
- Pure-NumPy NMS (no PyTorch dependency at inference)
- Contamination detection (7 scenarios, 0–1 score)
- Disposal guidance (7 streams, preparation steps, regulatory notes)
- Step-by-step reasoning chain explainability
- Structured JSON response with timings and model versions

### MLOps Platform
- **ModelRegistry** — versioned JSON-backed model lifecycle
- **DatasetRegistry** — dataset snapshots with checksums and lineage
- **ExperimentTracker** — MLflow + JSON-backed experiment records
- **ModelValidator** — 10 automated pre-deployment checks
- **ModelDeployer** — 4 strategies: immediate, blue-green, canary, rollback
- **DriftDetector** — PSI + KS test on input/prediction distributions
- **ConceptDriftDetector** — Page-Hinkley + CUSUM + Entropy monitoring
- **ContinuousLearningManager** — collect → validate → snapshot → retrain → promote
- **RetrainingScheduler** — daily/weekly/monthly/drift-triggered

### Production Optimization
- ONNX graph optimization (constant folding, dead node elimination, op fusion)
- FP16 conversion (requires `onnxconverter-common`)
- INT8 dynamic and static quantization (requires `onnxruntime.quantization`)
- Full latency benchmarking (cold start, warm, P50/P95/P99, FPS)
- 12-category robustness testing
- Cross-format accuracy validation (FP32 vs FP16 vs INT8)

### Deployment Infrastructure
- 3 Dockerfiles (CPU, GPU, Training)
- Docker Compose stack with MLflow + TensorBoard
- 3 GitHub Actions workflows (CI, Benchmark, Deployment)
- 4 environment configs (production, staging, development, testing)
- Container healthcheck script
- Blue-green and canary deployment support

---

## Breaking Changes from Pre-Release

None. This is the initial production release.

---

## Verified

| Requirement | Status |
|---|---|
| 109 SPIRO classes exactly | ✓ PASS |
| No taxonomy mismatches | ✓ PASS |
| All Python files syntax-valid | ✓ PASS (174/174) |
| All YAML configs parse | ✓ PASS (26/26, after fix) |
| No broken internal imports | ✓ PASS (1 fixed) |
| Guidance covers all 109 classes | ✓ PASS (9 missing fixed) |
| Contamination IDs all valid | ✓ PASS |
| Security: no unsafe torch.load | ✓ PASS (1 fixed) |
| All required CLI scripts present | ✓ PASS (33/33) |
| All required docs present | ✓ PASS |

---

## Known Limitations

See `docs/KNOWN_LIMITATIONS.md` for the complete list of 17 documented constraints.

---

## Upgrade Path

This subsystem integrates with the SPIRO backend via:
```python
from lib.ml.pipeline import SPIROPipeline
pipeline = SPIROPipeline.from_config("configs/production/pipeline.yaml")
result = pipeline.infer(image_source)
```

No architectural changes to the frontend or backend are required.
