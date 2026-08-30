# SPIRO ML — Google Colab Training Package

## Overview

This package trains SPIRO waste detection models on 5 real datasets using
Google Colab's free T4 GPU (~15 GB VRAM).

**Training time: ~3–4 hours total on free T4**

---

## What Gets Trained

| Model | Variant | Parameters | Purpose |
|---|---|---|---|
| YOLOv11n | nano | 2.6M | Waste object detection |
| EfficientNetV2-B0 | b0 | 7.4M | Crop verification / re-classification |

Both export to ONNX opset 17 for production deployment.

---

## Datasets Used

| Dataset | Classes | Format | License |
|---|---|---|---|
| TACO | 60 → mapped to SPIRO | COCO JSON | CC BY 4.0 |
| TrashNet | 6 → mapped to SPIRO | Image folder | MIT |
| ZeroWaste-f | 8 → mapped to SPIRO | YOLO | CC BY-NC 4.0 |
| Garbage Classification (Kaggle) | 12 → mapped to SPIRO | Image folder | Public |
| Recyclable & HH Waste | 30 → mapped to SPIRO | Image folder/YOLO | Public |

All datasets are automatically downloaded, mapped to 109 SPIRO classes, and merged.

---

## Quick Start

### Step 1: Open in Colab
1. Upload `SPIRO_ML_Training.ipynb` to Google Drive
2. Open with Google Colab
3. Runtime → Change runtime type → **T4 GPU** → Save

### Step 2: Run cells in order
- **Cell 1**: Check GPU (verify T4 is active)
- **Cell 2**: Mount Google Drive (checkpoints save here)
- **Cell 3**: Install dependencies
- **Cell 4**: Upload `spiro_colab.zip`
- **Cell 5**: Extract package
- **Cell 6**: Download & prepare all datasets (~30–60 min)
- **Cell 7**: View dataset statistics
- **Cell 8**: Train YOLOv11n (~2–3 hours)
- **Cell 9**: Export YOLOv11 to ONNX
- **Cell 10**: Train EfficientNetV2-B0 (~45 min)
- **Cell 11**: Export EfficientNetV2 to ONNX
- **Cell 12**: Test inference (smoke test)
- **Cell 13**: Package models → `spiro_models.zip`

---

## Individual Script Usage

```bash
# Download and prepare all datasets
python scripts/setup_datasets.py --all

# Download specific datasets only
python scripts/setup_datasets.py --taco --trashnet

# Show dataset statistics
python scripts/dataset_stats.py

# Train YOLOv11 (nano, 100 epochs)
python scripts/train_yolo.py --variant nano --epochs 100

# Train with fewer epochs (for testing)
python scripts/train_yolo.py --variant nano --epochs 10

# Resume interrupted training
python scripts/train_yolo.py --variant nano --resume

# Export YOLOv11 to ONNX
python scripts/export_yolo.py \
    --weights runs/detect/spiro_yolov11n/weights/best.pt

# Train EfficientNetV2-B0
python scripts/train_effnet.py --variant b0 --epochs 30

# Export EfficientNetV2 to ONNX
python scripts/export_effnet.py --variant b0 \
    --weights runs/verify/spiro_effnetv2_b0/best_effnet.pt

# Run inference smoke test
python scripts/test_inference.py \
    --yolo /content/drive/MyDrive/SPIRO_ML/yolo_best.onnx \
    --effnet /content/drive/MyDrive/SPIRO_ML/effnet_best.onnx

# Package models for download
python scripts/package_models.py
```

---

## Directory Structure

```
spiro_colab/
├── SPIRO_ML_Training.ipynb   ← Open this in Colab
├── README_COLAB.md           ← This file
├── scripts/
│   ├── setup_datasets.py     ← Download + map + merge datasets
│   ├── dataset_stats.py      ← Show class distribution stats
│   ├── train_yolo.py         ← YOLOv11 training
│   ├── export_yolo.py        ← Export YOLOv11 → ONNX
│   ├── train_effnet.py       ← EfficientNetV2 training
│   ├── export_effnet.py      ← Export EfficientNetV2 → ONNX
│   ├── test_inference.py     ← Smoke test both models
│   └── package_models.py     ← Zip models for download
└── configs/
    ├── taxonomy/
    │   └── spiro_taxonomy.yaml  ← 109-class SPIRO taxonomy
    └── training/
        └── colab_yolo.yaml      ← YOLO training config
```

---

## After Training

Your Google Drive will contain:
```
SPIRO_ML/
├── yolo_nano_best.pt        ← PyTorch YOLOv11 weights
├── yolo_best.onnx           ← Production ONNX (deploy this)
├── yolo_metadata.json
├── effnet_b0_best.pt        ← PyTorch EfficientNetV2 weights
├── effnet_best.onnx         ← Production ONNX (deploy this)
├── effnet_metadata.json
├── effnet_b0_history.csv    ← Training loss/accuracy log
└── spiro_models.zip         ← Everything bundled for download
```

### Deploy to SPIRO backend

1. Download `spiro_models.zip`
2. Extract models:
   ```bash
   unzip spiro_models.zip
   cp models/yolo_best.onnx  spiro_ml/models/serve/yolo_active.onnx
   cp models/effnet_best.onnx spiro_ml/models/serve/effnet_active.onnx
   ```
3. Verify:
   ```bash
   python scripts/healthcheck.py
   python training/infer_pipeline.py --image test.jpg --pretty
   ```

---

## Troubleshooting

**Colab disconnects mid-training**
- Training auto-saves every 10 epochs to `runs/detect/.../weights/last.pt`
- Resume with `python scripts/train_yolo.py --resume`

**Out of memory (CUDA OOM)**
- Reduce batch: `python scripts/train_yolo.py --batch 8`
- Disable mosaic: `python scripts/train_yolo.py --no-mosaic`

**Dataset download fails**
- Some Kaggle datasets require `kaggle.json` credentials
- Run individual downloaders: `python scripts/setup_datasets.py --taco --trashnet`
- Package still trains on whatever datasets downloaded successfully

**Low mAP after training**
- Increase epochs: `--epochs 200`
- Try small variant: `--variant small` (needs more VRAM)
- Check dataset stats: `python scripts/dataset_stats.py`

---

## Expected Results (free T4, 100 epochs)

| Model | mAP50 | mAP50-95 | Training Time |
|---|---|---|---|
| YOLOv11n | ~0.45–0.60 | ~0.30–0.42 | ~2–3h |
| EfficientNetV2-B0 | Top-1: ~0.60–0.75 | — | ~45min |

Results vary by dataset availability and class distribution.
More data = better performance.
