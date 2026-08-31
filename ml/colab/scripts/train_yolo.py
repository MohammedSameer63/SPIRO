#!/usr/bin/env python3
"""
SPIRO — train_yolo.py
Trains YOLOv11 on the merged SPIRO dataset.

Free T4 GPU defaults:
  - nano variant (2.6M params)
  - batch 16
  - 640px input
  - 100 epochs (resume supported)
  - Saves best.pt to Google Drive
"""
import argparse
import os
import shutil
import sys
import time
from pathlib import Path

# ── Colab paths ───────────────────────────────────────────────────────────────
BASE      = Path("/content/spiro")
DATA_YAML = BASE / "datasets" / "spiro_merged" / "data.yaml"
RUNS_DIR  = BASE / "runs" / "detect"

# Model variant → pretrained weights + recommended batch
VARIANT_CFG = {
    "nano":   {"model": "yolo11n.pt", "batch": 16,  "img": 640},
    "small":  {"model": "yolo11s.pt", "batch": 8,   "img": 640},
    "medium": {"model": "yolo11m.pt", "batch": 4,   "img": 640},
    "large":  {"model": "yolo11l.pt", "batch": 2,   "img": 640},
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train YOLOv11 for SPIRO")
    p.add_argument("--variant", default="nano",
                   choices=list(VARIANT_CFG.keys()),
                   help="YOLOv11 variant (default: nano for free T4)")
    p.add_argument("--epochs",    type=int, default=100,
                   help="Training epochs (default: 100)")
    p.add_argument("--batch",     type=int, default=None,
                   help="Override batch size")
    p.add_argument("--img-size",  type=int, default=None,
                   help="Input image size (default: 640)")
    p.add_argument("--save-dir",  default="/content/drive/MyDrive/SPIRO_ML",
                   help="Directory to copy best.pt after training")
    p.add_argument("--resume",    action="store_true",
                   help="Resume from last checkpoint")
    p.add_argument("--patience",  type=int, default=30,
                   help="Early stopping patience")
    p.add_argument("--no-mosaic", action="store_true",
                   help="Disable mosaic augmentation (saves VRAM)")
    p.add_argument("--workers",   type=int, default=2)
    return p.parse_args()


def check_dataset() -> bool:
    if not DATA_YAML.exists():
        print(f"❌ Dataset not found: {DATA_YAML}")
        print("   Run: python scripts/setup_datasets.py --all")
        return False
    import yaml
    with open(DATA_YAML) as f:
        d = yaml.safe_load(f)
    nc = d.get("nc", 0)
    if nc != 109:
        print(f"❌ Expected 109 classes, got {nc}")
        return False
    # Count images
    train_imgs = list((Path(d["path"]) / "train" / "images").glob("*.*"))
    print(f"✅ Dataset: {nc} classes | {len(train_imgs)} training images")
    return True


def check_gpu() -> str:
    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            mem = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"✅ GPU: {name} ({mem:.1f} GB)")
            return "0"
        print("⚠️  No GPU detected — training on CPU (very slow!)")
        return "cpu"
    except ImportError:
        return "cpu"


def main() -> None:
    args = parse_args()
    cfg  = VARIANT_CFG[args.variant]

    print(f"\n{'='*55}")
    print(f"  SPIRO YOLOv11 Training")
    print(f"{'='*55}")
    print(f"  Variant:  {args.variant} ({cfg['model']})")
    print(f"  Epochs:   {args.epochs}")
    print(f"  Classes:  109")

    if not check_dataset():
        sys.exit(1)

    device = check_gpu()
    batch  = args.batch or cfg["batch"]
    imgsz  = args.img_size or cfg["img"]
    exp_name = f"spiro_yolov11{args.variant[0]}"

    # Reduce batch if very limited VRAM
    try:
        import torch
        if torch.cuda.is_available():
            vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            if vram_gb < 8 and batch > 8:
                batch = 8
                print(f"  ⚠️  Low VRAM ({vram_gb:.1f}GB) — reducing batch to {batch}")
    except Exception:
        pass

    print(f"  Batch:    {batch}")
    print(f"  ImgSize:  {imgsz}")
    print(f"  Device:   {device}")
    print(f"  Output:   {RUNS_DIR / exp_name}")
    print(f"{'='*55}\n")

    # ── Train ─────────────────────────────────────────────────────────────────
    try:
        from ultralytics import YOLO
    except ImportError:
        print("❌ ultralytics not installed. Run: pip install ultralytics")
        sys.exit(1)

    model = YOLO(cfg["model"])  # downloads pretrained weights

    train_kwargs = dict(
        data=str(DATA_YAML),
        epochs=args.epochs,
        batch=batch,
        imgsz=imgsz,
        device=device,
        project=str(RUNS_DIR),
        name=exp_name,
        exist_ok=True,
        patience=args.patience,
        workers=args.workers,
        save=True,
        save_period=10,
        val=True,
        plots=True,
        verbose=True,
        # Augmentation — reduced for stability on free tier
        mosaic=0.0 if args.no_mosaic else 0.8,
        mixup=0.0,
        copy_paste=0.0,
        flipud=0.0,
        fliplr=0.5,
        degrees=10.0,
        translate=0.1,
        scale=0.4,
        hsv_h=0.015,
        hsv_s=0.5,
        hsv_v=0.3,
        # Optimiser
        optimizer="SGD",
        lr0=0.01,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3.0,
        # Loss weights
        box=7.5,
        cls=0.5,
        dfl=1.5,
        # Resume
        resume=args.resume,
    )

    t0 = time.perf_counter()
    results = model.train(**train_kwargs)
    elapsed = (time.perf_counter() - t0) / 3600

    # ── Post-training ─────────────────────────────────────────────────────────
    run_dir  = RUNS_DIR / exp_name
    best_pt  = run_dir / "weights" / "best.pt"
    last_pt  = run_dir / "weights" / "last.pt"

    print(f"\n{'='*55}")
    print(f"  Training complete in {elapsed:.2f}h")
    if best_pt.exists():
        print(f"  Best weights: {best_pt}")
        metrics = results.results_dict if hasattr(results, "results_dict") else {}
        print(f"  mAP50:        {metrics.get('metrics/mAP50(B)', 0):.4f}")
        print(f"  mAP50-95:     {metrics.get('metrics/mAP50-95(B)', 0):.4f}")

    # Copy to Google Drive
    save_dir = Path(args.save_dir)
    if save_dir.exists():
        save_dir.mkdir(parents=True, exist_ok=True)
        if best_pt.exists():
            dst = save_dir / f"yolo_{args.variant}_best.pt"
            shutil.copy2(best_pt, dst)
            print(f"  ✅ Saved to Drive: {dst}")
        if last_pt.exists():
            shutil.copy2(last_pt, save_dir / f"yolo_{args.variant}_last.pt")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    main()
