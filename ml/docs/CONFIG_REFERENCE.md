# SPIRO ML — Configuration Reference

## Pipeline Config (`configs/pipeline/inference_pipeline.yaml`)

### models
| Key | Type | Default | Description |
|---|---|---|---|
| yolo_onnx | str | `models/serve/yolo_active.onnx` | Active YOLOv11 ONNX |
| effnet_onnx | str | `models/serve/effnet_active.onnx` | Active EffNetV2 ONNX |
| yolo_input_size | int | 640 | YOLO square input size |
| effnet_input_size | int | 300 | EffNetV2 square input size |
| num_classes | int | **109** | SPIRO taxonomy size |

### runtime
| Key | Type | Default | Description |
|---|---|---|---|
| providers | list | `[TRT, CUDA, CPU]` | ORT provider priority |
| yolo_threads | int | 4 | ORT intra-op threads (YOLO) |
| effnet_threads | int | 4 | ORT intra-op threads (EffNet) |
| graph_optimization | str | `ORT_ENABLE_ALL` | ORT graph opt level |
| fp16 | bool | false | FP16 inference (GPU only) |
| effnet_batch | int | 8 | Crops per EffNet ORT batch |
| warmup_runs | int | 3 | Warmup forward passes |

### preprocessing
| Key | Type | Default | Description |
|---|---|---|---|
| blur_threshold | float | 80.0 | Laplacian variance; below → reject |
| brightness_min | float | 20.0 | Mean brightness lower bound |
| brightness_max | float | 245.0 | Mean brightness upper bound |
| quality_score_threshold | float | 0.3 | Composite quality 0–1 |
| auto_orient | bool | true | EXIF orientation correction |
| target_size | int | 640 | Letterbox output size |

### detection
| Key | Type | Default | Description |
|---|---|---|---|
| conf_threshold | float | 0.25 | YOLO confidence cutoff |
| iou_threshold | float | 0.45 | NMS IoU threshold |
| max_detections | int | 100 | Max objects per image |
| min_bbox_area | float | 0.001 | Min box area (fraction of image) |

### fusion
| Key | Type | Default | Description |
|---|---|---|---|
| method | str | `weighted_average` | weighted_average \| geometric_mean \| harmonic_mean \| bayesian \| temperature |
| yolo_weight | float | 0.45 | YOLO weight |
| effnet_weight | float | 0.55 | EffNet weight |
| min_final_confidence | float | 0.25 | Below → "uncertain" |

## Training Config (`configs/training/yolov11_training.yaml`)

| Key | Default | Description |
|---|---|---|
| training.epochs | 300 | Total epochs |
| training.batch_size | 16 | Images per batch |
| training.device | auto | auto \| cpu \| 0 \| 0,1 |
| training.amp | true | Automatic Mixed Precision |
| training.patience | 50 | Early stopping patience |
| optimizer.name | SGD | SGD \| Adam \| AdamW |
| optimizer.lr0 | 0.01 | Initial learning rate |
| model.weights | yolo11s.pt | Starting weights |
| model.num_classes | 109 | SPIRO class count |
| model.input_size | 640 | Training image size |
| export.auto_export_onnx | true | Export after training |
| export.opset | 17 | ONNX opset version |

## Verification Config (`configs/verification/effnetv2_verify.yaml`)

| Key | Default | Description |
|---|---|---|
| model.variant | efficientnetv2_s | b0/b1/b2/b3/s/m/l |
| model.num_classes | 109 | SPIRO class count |
| training.epochs | 50 | Total epochs |
| training.phase1_epochs | 10 | Frozen backbone epochs |
| training.phase2_epochs | 40 | Full fine-tune epochs |
| loss.name | label_smoothing | cross_entropy \| label_smoothing \| focal \| weighted |
| loss.label_smoothing | 0.1 | Smoothing factor |
| export.auto_export_onnx | true | Export after training |

## Environment Overrides

| Env | blur_threshold | conf | effnet_batch | warmup |
|---|---|---|---|---|
| production | 80.0 | 0.30 | 16 | 10 |
| staging | 80.0 | 0.30 | 8 | 5 |
| development | 0.0 | 0.15 | 4 | 1 |
| testing | 0.0 | 0.001 | 2 | 0 |
