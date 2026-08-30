# SPIRO ML — Known Limitations

## Model Limitations

1. **Small object detection**: YOLOv11s may miss objects smaller than ~32×32px at 640px input. Mitigation: use `model.input_size: 1280` for high-resolution images.

2. **Overlapping objects**: NMS may suppress valid detections when objects heavily overlap (IoU > 0.45). Mitigation: lower `detection.iou_threshold` to 0.6 or use `detection.agnostic_nms: true`.

3. **Occluded waste**: Heavily occluded objects (>70% covered) may be undetected. This is a fundamental limitation of single-view detection.

4. **Night/infrared images**: Models trained on daytime RGB. Performance degrades significantly on infrared or very-low-light images (brightness < 20).

5. **109-class ceiling**: The SPIRO taxonomy is fixed at 109 classes. Adding new classes requires full retraining pipeline.

## Fusion Limitations

6. **Bayesian fusion**: Requires accurate class priors. Uniform priors (default) may not reflect real-world waste distribution.

7. **Temperature calibration**: Optimal temperature T is dataset-specific. Default T=1.0 may not be calibrated for all deployment domains.

## MLOps Limitations

8. **Drift detection baseline**: PSI-based drift detection requires a stable baseline of ≥200 samples. Results unreliable with fewer samples.

9. **Concept drift**: Page-Hinkley test has a burn-in period of 30 samples before activation. Acute concept drift in the first 30 inferences is not detected.

10. **Rollback**: Can only roll back one step (to the immediately previous deployment). Multi-step rollback requires manual registry queries.

## Infrastructure Limitations

11. **TensorRT first-run**: TRT engine compilation takes 2–10 minutes on first load. Subsequent loads use the cached engine. Set `warmup_runs: 0` to skip during development.

12. **INT8 static quantization**: Requires `onnxruntime.quantization` package (not included in base `onnxruntime`). Install separately: `pip install onnxruntime-extensions`.

13. **FP16 conversion**: Requires `onnxconverter-common`. Not available by default: `pip install onnxconverter-common`.

14. **Multi-GPU training**: Requires Ultralytics DDP setup. EfficientNetV2 trainer uses single-GPU with gradient accumulation instead of DDP.

## Dataset Limitations

15. **OpenLitterMap license**: CC BY 4.0 — attribution required in any published results.

16. **ZeroWaste-f and MJU-Waste**: CC BY-NC 4.0 — commercial use requires separate licensing.

17. **Class imbalance**: Some rare SPIRO classes (e.g., ID 102–105) may have fewer than 10 training samples in the combined dataset. Use `loss.name: focal` for better handling.
