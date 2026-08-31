# SPIRO ML — Training Guide

## Overview

SPIRO ML trains YOLOv11 object detectors on the 109-class SPIRO waste taxonomy.
The pipeline supports all five model variants, multi-GPU training, automatic
resume, custom callbacks, TensorBoard/MLflow logging, and automatic ONNX export.

---

## Hardware Requirements

| Model | Min GPU RAM | Recommended | Batch Size | Training Time (est.) |
|---|---|---|---|---|
| yolo11n | 4 GB | 8 GB | 64 | 6–12 h on RTX 3080 |
| yolo11s | 6 GB | 12 GB | 32 | 12–20 h on RTX 3080 |
| yolo11m | 10 GB | 16 GB | 16 | 24–40 h on RTX 3080 |
| yolo11l | 14 GB | 24 GB | 8 | 40–60 h on A100 |
| yolo11x | 20 GB | 40 GB | 4 | 60–100 h on A100 |

CPU training is supported but ~50–100× slower — only recommended for debugging.

**Disk**: ~10 GB for the full SPIRO dataset + augmentation + checkpoints.

---

## Quick Start

### 1. Prepare the dataset
```bash
# Run the full dataset engineering pipeline first (Prompt 2)
make pipeline

# Or run just the split step if data is already merged
python training/dataset_splitter.py
```

### 2. Train
```bash
# Nano (fastest)
python training/train.py --config configs/training/yolo11n.yaml

# Small (recommended starting point)
python training/trainer.py --variant s

# With common overrides
python training/trainer.py --variant s --epochs 150 --batch 16 --device 0
```

### 3. Monitor
```bash
# TensorBoard
tensorboard --logdir logs/tensorboard

# MLflow
mlflow ui --backend-store-uri logs/mlflow --port 5000
```

### 4. Evaluate
```bash
python training/train.py --config configs/training/yolo11s.yaml \
    --validate-only \
    --weights models/checkpoints/spiro_yolo11s/weights/best.pt \
    --split test
```

### 5. Export to ONNX
```bash
python training/experiment.py export \
    --name spiro_yolo11s \
    --config configs/training/yolo11s.yaml
```

---

## Training Commands

### Basic training
```bash
python training/train.py --config configs/training/yolo11s.yaml
```

### Variant shortcut
```bash
python training/trainer.py --variant s
python training/trainer.py --variant m --device 0
```

### Override any config value
```bash
python training/train.py \
    --config configs/training/yolo11s.yaml \
    --set training.epochs=100 \
           training.batch_size=32 \
           optimizer.lr0=0.005 \
           augmentation.mosaic=0.5
```

### Multi-GPU (DDP)
```bash
# Two GPUs
python training/train.py \
    --config configs/training/yolo11m.yaml \
    --set training.device=0,1 training.batch_size=32

# Or let Ultralytics handle DDP automatically:
python -m torch.distributed.run --nproc_per_node=2 training/train.py \
    --config configs/training/yolo11m.yaml
```

### Disable augmentation (debug mode)
```bash
python training/trainer.py --variant n --no-aug --epochs 5
```

---

## Resume Commands

### Auto-resume (restarts from last.pt)
```bash
python training/train.py --config configs/training/yolo11s.yaml --resume
```

### Resume from specific checkpoint
```bash
python training/train.py \
    --config configs/training/yolo11s.yaml \
    --resume \
    --checkpoint models/checkpoints/spiro_yolo11s/weights/epoch_00050.pt
```

### Resume with variant shortcut
```bash
python training/trainer.py --variant s --resume
```

---

## Config Options

All options are in `configs/training/yolov11_training.yaml`. Variant configs
inherit from this base via `_base_:`.

### Key parameters

| Section | Key | Default | Description |
|---|---|---|---|
| model | weights | yolo11n.pt | Model variant |
| model | num_classes | 109 | Must match dataset |
| model | input_size | 640 | Image size for training |
| training | epochs | 300 | Total training epochs |
| training | batch_size | 16 | Images per batch |
| training | device | auto | auto/cpu/0/0,1/mps |
| training | amp | true | Automatic Mixed Precision |
| training | patience | 50 | Early stopping patience (0=off) |
| training | save_period | 10 | Save checkpoint every N epochs |
| training | gradient_accumulation | 1 | Grad accumulation steps |
| optimizer | name | SGD | SGD/Adam/AdamW/auto |
| optimizer | lr0 | 0.01 | Initial learning rate |
| optimizer | lrf | 0.01 | Final LR factor (cosine) |
| optimizer | momentum | 0.937 | SGD momentum / Adam β₁ |
| optimizer | weight_decay | 0.0005 | L2 regularization |
| optimizer | warmup_epochs | 3.0 | LR warmup duration |
| scheduler | name | cosine | cosine/linear/onecycle |
| ema | enabled | true | Exponential Moving Average |
| ema | decay | 0.9999 | EMA decay rate |
| loss | box | 7.5 | Box regression loss weight |
| loss | cls | 0.5 | Classification loss weight |
| loss | dfl | 1.5 | DFL loss weight |
| augmentation | mosaic | 1.0 | Mosaic probability |
| augmentation | mixup | 0.0 | MixUp probability |
| augmentation | copy_paste | 0.0 | Copy-paste probability |
| augmentation | fliplr | 0.5 | Horizontal flip probability |
| augmentation | hsv_h/s/v | 0.015/0.7/0.4 | HSV augmentation |
| export | auto_export_onnx | true | Export ONNX after training |
| export | opset | 17 | ONNX opset version |
| export | verify | true | Verify ORT vs PyTorch |

### Optimizer options

| Optimizer | Use case |
|---|---|
| SGD | Default, best for most cases with cosine LR |
| Adam | Faster convergence on small datasets |
| AdamW | Adam with decoupled weight decay |
| auto | Ultralytics selects based on model |

### LR Scheduler options

| Scheduler | Config |
|---|---|
| cosine | `scheduler.name: cosine` (default) |
| linear | `scheduler.name: linear` |
| onecycle | `scheduler.name: onecycle` + `pct_start`, `div_factor` |

---

## Expected Outputs

After training, outputs are organised as:

```
models/
├── checkpoints/
│   └── spiro_yolo11s/
│       ├── weights/
│       │   ├── best.pt          ← best model (tracked metric)
│       │   ├── last.pt          ← latest epoch
│       │   └── epoch_00050.pt   ← periodic saves
│       ├── checkpoint_meta.json
│       ├── training_summary.json
│       ├── confusion_matrix.png
│       ├── PR_curve.png
│       ├── P_curve.png
│       ├── R_curve.png
│       └── results.png
│
├── exports/
│   └── spiro_yolo11s_best.onnx  ← production ONNX model
│
logs/
├── tensorboard/
│   └── spiro_yolo11s/           ← TensorBoard events
├── mlflow/
│   └── <experiment>/            ← MLflow runs
└── training_history.csv         ← per-epoch metrics CSV

reports/
└── training/
    └── spiro_yolo11s/
        ├── loss_curves.png
        ├── map_curve.png
        ├── precision_recall_curves.png
        └── results.png
```

---

## Metrics Logged Per Epoch

| Metric | Description |
|---|---|
| mAP50 | Mean Average Precision @ IoU 0.5 |
| mAP50-95 | mAP @ IoU 0.5:0.95 (primary metric) |
| precision | Mean precision across all classes |
| recall | Mean recall across all classes |
| train/box_loss | Training box regression loss |
| train/cls_loss | Training classification loss |
| train/dfl_loss | Training DFL loss |
| val/box_loss | Validation box loss |
| val/cls_loss | Validation classification loss |
| val/dfl_loss | Validation DFL loss |
| lr | Current learning rate |
| time_s | Epoch duration in seconds |
| gpu_mem_GB | GPU memory reserved (GB) |

---

## Experiment Management

```bash
# List all experiments
python training/experiment.py list

# Show details
python training/experiment.py show --name spiro_yolo11s

# Compare two experiments
python training/experiment.py compare --names spiro_yolo11n spiro_yolo11s

# Generate plots
python training/experiment.py plot --name spiro_yolo11s

# Export best model
python training/experiment.py export --name spiro_yolo11s
```

---

## Callback System

Callbacks are registered via `build_callbacks()` and hooked into Ultralytics events:

| Callback | Event | Purpose |
|---|---|---|
| TensorBoardCallback | on_fit_epoch_end | Writes metrics to TensorBoard |
| CSVLoggerCallback | on_fit_epoch_end | Appends row to training_history.csv |
| EarlyStoppingCallback | on_val_end | Stops on metric plateau |
| EpochSummaryCallback | on_fit_epoch_end | Console epoch summary |

### Testing callbacks
```bash
python training/callbacks.py --list
python training/callbacks.py --test-csv
python training/callbacks.py --test-tb --tb-dir /tmp/spiro_tb_test
python training/callbacks.py --replay logs/training_history.csv
```

---

## ONNX Export

ONNX is automatically exported after training if `export.auto_export_onnx: true`.

Manual export:
```bash
# Via experiment CLI
python training/experiment.py export --name spiro_yolo11s

# Via script (all variants)
python -m lib.ml.scripts.export_onnx \
    --config configs/training/yolo11s.yaml \
    --weights models/checkpoints/spiro_yolo11s/weights/best.pt \
    --architecture yolov11 \
    --verify
```

Verification checks numerical parity between PyTorch and ONNXRuntime (max diff < 0.01).

---

## Tips

**Stuck loss?** Reduce `optimizer.lr0` by 5×, enable `optimizer.warmup_epochs: 5.0`.

**GPU OOM?** Reduce `training.batch_size`, increase `training.gradient_accumulation`.

**Slow convergence?** Try `optimizer.name: AdamW` with `optimizer.lr0: 0.001`.

**Poor small object detection?** Increase `model.input_size` to 1280.

**Class imbalance?** Set `dataset.class_weights` in the config (per-class float list).
