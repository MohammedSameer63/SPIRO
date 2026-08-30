# SPIRO ML — Troubleshooting Guide

## Import Errors

**`No module named 'lib.ml'`**
```bash
export PYTHONPATH=/path/to/spiro_ml/src
# or
pip install -e .
```

**`No module named 'onnxruntime'`**
```bash
pip install onnxruntime          # CPU
pip install onnxruntime-gpu      # GPU
```

**`No module named 'timm'`**
```bash
pip install timm
```

## Model Loading Errors

**`FileNotFoundError: yolo_active.onnx`**
- Models not yet deployed. Run:
```bash
python training/mlops/model_deployer.py deploy \
    --model-id yolov11s --version v1.0.0 \
    --onnx models/exports/spiro_yolov11s_best.onnx \
    --strategy immediate --task detection
```

**`ONNX graph invalid`**
```bash
python training/mlops/model_validator.py \
    --onnx path/to/model.onnx --model-id test --version v1
```

**`ort.InferenceSession: No such file`**
- Check ONNX path in `configs/production/pipeline.yaml`

## Inference Errors

**`All images rejected as blurry`**
- Lower `preprocessing.blur_threshold` in pipeline config:
```yaml
preprocessing:
  blur_threshold: 0.0   # accept all
```

**`No detections on clear images`**
- Lower detection threshold:
```yaml
detection:
  conf_threshold: 0.15
```
- Use development config: `configs/development/pipeline.yaml`

**`109 class mismatch`**
```bash
python -c "from lib.ml.dataset_engineering.taxonomy import TaxonomyLoader; print(TaxonomyLoader().num_classes)"
```
Should print `109`. If not, check `configs/taxonomy/spiro_taxonomy.yaml`.

**`CUDA out of memory`**
```yaml
runtime:
  effnet_batch: 4   # reduce from default 16
  providers: ["CPUExecutionProvider"]
```

## Training Errors

**`Dataset not found`**
```bash
make pipeline   # run full dataset download + processing pipeline
```

**`YOLO training crashes`**
- Check GPU memory with `nvidia-smi`
- Reduce batch size: `--batch 8`
- Reduce input size: `--set model.input_size=320`

**`EfficientNetV2 out of memory`**
```yaml
training:
  batch_size: 8
  gradient_accumulation: 4
```

## MLOps Errors

**`Model not found in registry`**
```bash
python training/mlops/model_registry.py list
python training/mlops/model_metadata.py register --model-id yolov11s --version v1.0.0 ...
```

**`Rollback: no previous deployment`**
- No previous deployment in history. Deploy a baseline first.

**`Drift detection: no baseline`**
```python
import numpy as np
from lib.ml.mlops import DriftDetector
detector = DriftDetector()
baseline = np.load("path/to/baseline_confidences.npy")
detector.set_baseline(confidences=baseline)
detector.save_baseline(Path("mlops/baselines/baseline.json"))
```

## Docker Errors

**`docker: Cannot connect to Docker daemon`**
```bash
sudo systemctl start docker
```

**`GPU not available in container`**
```bash
# Install nvidia-container-toolkit
sudo apt install nvidia-container-toolkit
sudo systemctl restart docker
docker run --gpus all ...
```

**`Healthcheck failing`**
```bash
python scripts/healthcheck.py   # run outside Docker to diagnose
```

## Performance Issues

**Slow CPU inference (>500ms)**
1. Enable graph optimization: `runtime.graph_optimization: ORT_ENABLE_ALL`
2. Increase threads: `runtime.yolo_threads: 8`
3. Use FP16 on GPU or INT8 dynamic on CPU

**Memory leak symptoms**
- PerformanceMonitor tracks RAM — export and review:
```bash
python training/mlops/performance_monitor.py export
```

**Low FPS on GPU**
1. Check TensorRT is active: should appear in `active_providers`
2. Enable FP16: `runtime.fp16: true`
3. Reduce crop batch size or increase `effnet_batch`
