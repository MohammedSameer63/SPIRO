# SPIRO ML — Production Deployment Checklist

**Date:** ___________  
**Version:** ___________  
**Deployed by:** ___________  
**Environment:** [ ] Development  [ ] Staging  [ ] Production

---

## 1. Model Artefacts

- [ ] `models/exports/spiro_yolov11s_best.onnx` exists
- [ ] `models/exports/spiro_effnetv2_s_best_effnet.onnx` exists
- [ ] Both ONNX files have opset version 17
- [ ] YOLOv11 input shape: `[1, 3, 640, 640]`
- [ ] EfficientNetV2 input shape: `[1, 3, 300, 300]`

## 2. Model Validation

- [ ] `python training/mlops/model_validator.py --onnx ... --model-id yolov11s` → **PASSED**
- [ ] `python training/mlops/model_validator.py --onnx ... --model-id effnetv2_s --input-size 300 --task verification` → **PASSED**
- [ ] `file_exists` ✓
- [ ] `onnx_graph_valid` ✓
- [ ] `ort_loads` ✓
- [ ] `inference_executes` ✓
- [ ] `output_shape` ✓
- [ ] `taxonomy_consistency` ✓ (109 classes)
- [ ] `latency_cpu` ✓ (< 500ms)
- [ ] `memory_usage` ✓ (< 2048MB)
- [ ] `regression_deterministic` ✓

## 3. Taxonomy

- [ ] `configs/taxonomy/spiro_taxonomy.yaml` has exactly 109 classes
- [ ] All 22 waste groups present
- [ ] `python -c "from lib.ml.dataset_engineering.taxonomy import TaxonomyLoader; assert TaxonomyLoader().num_classes == 109"` → OK

## 4. Registry

- [ ] Model registered: `python training/mlops/model_registry.py show --model-id yolov11s --version vX.Y.Z`
- [ ] Model registered: `python training/mlops/model_registry.py show --model-id effnetv2_s --version vX.Y.Z`
- [ ] Dataset version linked in metadata
- [ ] Experiment ID linked in metadata

## 5. Comparison

- [ ] `python training/mlops/model_compare.py ...` completed
- [ ] Primary metric improved (mAP50-95 delta ≥ 0.001)
- [ ] Latency regression < 50ms
- [ ] No secondary metric regression > 1%
- [ ] Comparison Markdown saved to `mlops/reports/comparison_*.md`

## 6. Deployment

- [ ] `python training/mlops/model_deployer.py deploy --strategy blue_green --task detection` → OK
- [ ] `python training/mlops/model_deployer.py deploy --strategy blue_green --task verification` → OK
- [ ] `models/serve/yolo_active.onnx` exists and loads
- [ ] `models/serve/effnet_active.onnx` exists and loads
- [ ] Deployment recorded in `mlops/deployments/deployment_history.json`

## 7. Health Check

- [ ] `python scripts/healthcheck.py` → **PASS** (all checks green)
- [ ] Docker healthcheck passes (if containerised)

## 8. Smoke Test

- [ ] `python training/infer_pipeline.py --image test_image.jpg --pretty` returns valid JSON
- [ ] `status` is `"success"` or `"partial"`
- [ ] `detections` list present
- [ ] `guidance.summary` non-empty
- [ ] `timings.total_ms` > 0

## 9. Rollback Plan

- [ ] Previous model version confirmed in `mlops/deployments/deployment_history.json`
- [ ] `python training/mlops/rollback.py --task detection` tested on staging
- [ ] Rollback command bookmarked: `make mlops-rollback`

## 10. Monitoring

- [ ] Drift baseline saved: `mlops/baselines/baseline_confs.npy`
- [ ] Retraining schedule configured
- [ ] Performance monitor active
- [ ] Alert thresholds configured

---

**Sign-off:** _______________________

**Notes:** 

___________________________________________

___________________________________________
