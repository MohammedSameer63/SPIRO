#!/usr/bin/env python3
"""
SPIRO — train_effnet.py
Trains EfficientNetV2-B0 crop verifier on YOLO-labelled patches.

Workflow:
  1. Extract crop patches from YOLO-labelled dataset
  2. Two-phase training (frozen backbone → full fine-tune)
  3. Saves best_effnet.pt to Google Drive
"""
import argparse
import csv
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

BASE      = Path("/content/spiro")
DATA_DIR  = BASE / "datasets" / "spiro_merged"
CROPS_DIR = BASE / "datasets" / "spiro_crops"
RUNS_DIR  = BASE / "runs" / "verify"
NUM_CLASSES = 109

VARIANT_CFG = {
    "b0": {"timm": "tf_efficientnetv2_b0", "size": 192, "batch": 64},
    "b1": {"timm": "tf_efficientnetv2_b1", "size": 240, "batch": 32},
    "s":  {"timm": "tf_efficientnetv2_s",  "size": 300, "batch": 16},
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train EfficientNetV2 for SPIRO")
    p.add_argument("--variant", default="b0", choices=list(VARIANT_CFG.keys()))
    p.add_argument("--epochs",   type=int, default=30)
    p.add_argument("--batch",    type=int, default=None)
    p.add_argument("--phase1",   type=int, default=5,
                   help="Frozen-backbone epochs")
    p.add_argument("--phase2",   type=int, default=25,
                   help="Full fine-tune epochs")
    p.add_argument("--save-dir", default="/content/drive/MyDrive/SPIRO_ML")
    p.add_argument("--skip-crops", action="store_true",
                   help="Skip crop extraction if already done")
    p.add_argument("--workers", type=int, default=2)
    return p.parse_args()


# =============================================================================
# Step 1: Extract crops from YOLO-labelled images
# =============================================================================

def extract_crops(
    data_dir: Path,
    crops_dir: Path,
    min_size: int = 32,
    padding: float = 0.10,
) -> Dict[str, int]:
    """Extract padded object crops → crops_dir/<split>/<class_id>/img.jpg"""
    import cv2
    import yaml

    data_yaml = data_dir / "data.yaml"
    with open(data_yaml) as f:
        data = yaml.safe_load(f)
    class_names = data.get("names", [f"c{i}" for i in range(NUM_CLASSES)])
    if isinstance(class_names, dict):
        class_names = [class_names.get(i, f"c{i}") for i in range(NUM_CLASSES)]

    counts: Dict[str, int] = {}
    total = 0

    for split in ["train", "val", "test"]:
        imgs_dir = data_dir / split / "images"
        lbls_dir = data_dir / split / "labels"
        if not imgs_dir.exists():
            continue

        img_paths = [p for p in imgs_dir.iterdir() if p.suffix.lower() in IMG_EXTS]
        print(f"  Extracting {split} crops from {len(img_paths)} images...")

        for img_path in img_paths:
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            ih, iw = img.shape[:2]

            lbl_path = lbls_dir / (img_path.stem + ".txt")
            if not lbl_path.exists():
                continue

            for i, line in enumerate(lbl_path.read_text().splitlines()):
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                cls_id = int(parts[0])
                cx, cy, w, h = map(float, parts[1:])

                # Padded pixel coordinates
                pw = w * padding
                ph = h * padding
                x1 = max(0, int((cx - w/2 - pw) * iw))
                y1 = max(0, int((cy - h/2 - ph) * ih))
                x2 = min(iw, int((cx + w/2 + pw) * iw))
                y2 = min(ih, int((cy + h/2 + ph) * ih))

                crop = img[y1:y2, x1:x2]
                if crop.shape[0] < min_size or crop.shape[1] < min_size:
                    continue

                cls_name = class_names[cls_id] if cls_id < len(class_names) else f"cls_{cls_id}"
                out_dir = crops_dir / split / str(cls_id)
                out_dir.mkdir(parents=True, exist_ok=True)
                out_path = out_dir / f"{img_path.stem}_b{i:04d}.jpg"
                cv2.imwrite(str(out_path), crop, [cv2.IMWRITE_JPEG_QUALITY, 90])
                counts[cls_name] = counts.get(cls_name, 0) + 1
                total += 1

    print(f"  Total crops extracted: {total}")
    print(f"  Classes with crops: {len(counts)}")
    return counts


# =============================================================================
# Step 2: Build PyTorch DataLoaders from crops
# =============================================================================

def build_loaders(
    crops_dir: Path,
    split: str,
    input_size: int,
    batch_size: int,
    is_train: bool,
    workers: int = 2,
):
    import torch
    from torch.utils.data import DataLoader, Dataset
    from torchvision import transforms
    import cv2
    import numpy as np

    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]

    if is_train:
        tf = transforms.Compose([
            transforms.RandomResizedCrop(input_size, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.3, 0.3, 0.3, 0.1),
            transforms.RandomGrayscale(0.05),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
            transforms.RandomErasing(p=0.2),
        ])
    else:
        tf = transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

    class CropDataset(Dataset):
        def __init__(self, root: Path, transform):
            from PIL import Image
            self.samples = []
            self.transform = transform
            split_dir = root / split
            if not split_dir.exists():
                return
            for cls_dir in sorted(split_dir.iterdir()):
                if not cls_dir.is_dir():
                    continue
                try:
                    cls_id = int(cls_dir.name)
                except ValueError:
                    continue
                for img_p in cls_dir.iterdir():
                    if img_p.suffix.lower() in IMG_EXTS:
                        self.samples.append((img_p, cls_id))

        def __len__(self): return len(self.samples)

        def __getitem__(self, idx):
            from PIL import Image
            img_path, label = self.samples[idx]
            try:
                img = Image.open(img_path).convert("RGB")
                return self.transform(img), label
            except Exception:
                return torch.zeros(3, input_size, input_size), label

    dataset = CropDataset(crops_dir, tf)
    if len(dataset) == 0:
        return None

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=is_train,
        num_workers=workers,
        pin_memory=True,
        drop_last=is_train,
    )


# =============================================================================
# Step 3: Training loop
# =============================================================================

def train_phase(
    model,
    loader,
    val_loader,
    optimizer,
    scheduler,
    criterion,
    device,
    n_epochs: int,
    phase_name: str,
    best_acc: float,
    run_dir: Path,
    csv_writer,
) -> Tuple[float, float]:
    import torch
    import torch.nn.functional as F
    from torch.cuda.amp import GradScaler, autocast

    scaler = GradScaler(enabled=device.type == "cuda")
    patience = 10
    no_improve = 0

    for epoch in range(1, n_epochs + 1):
        # Train
        model.train()
        tr_loss = tr_correct = tr_total = 0
        for imgs, labels in loader:
            imgs   = imgs.to(device)
            labels = labels.to(device)
            with autocast(enabled=device.type == "cuda"):
                out  = model(imgs)
                loss = criterion(out, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            tr_loss    += loss.item() * imgs.size(0)
            tr_correct += (out.argmax(1) == labels).sum().item()
            tr_total   += imgs.size(0)

        tr_acc  = tr_correct / max(tr_total, 1)
        tr_loss /= max(tr_total, 1)

        # Validate
        model.eval()
        val_loss = val_correct = val_total = 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                with autocast(enabled=device.type == "cuda"):
                    out  = model(imgs)
                    loss = criterion(out, labels)
                val_loss    += loss.item() * imgs.size(0)
                val_correct += (out.argmax(1) == labels).sum().item()
                val_total   += imgs.size(0)
        val_acc  = val_correct / max(val_total, 1)
        val_loss /= max(val_total, 1)

        lr = optimizer.param_groups[0]["lr"]
        if scheduler:
            scheduler.step()

        print(
            f"  [{phase_name} {epoch:>3}/{n_epochs}] "
            f"loss={tr_loss:.4f}/{val_loss:.4f} "
            f"acc={tr_acc:.4f}/{val_acc:.4f} "
            f"lr={lr:.2e}"
        )

        if csv_writer:
            csv_writer.writerow([
                phase_name, epoch, round(tr_loss, 6), round(tr_acc, 6),
                round(val_loss, 6), round(val_acc, 6), round(lr, 8),
            ])

        # Save best
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(
                {"state_dict": model.state_dict(), "val_acc": val_acc,
                 "epoch": epoch, "phase": phase_name},
                str(run_dir / "best_effnet.pt"),
            )
            print(f"    ✅ New best val_acc={val_acc:.4f}")
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"    Early stopping at epoch {epoch}")
                break

        # Always save last
        torch.save(
            {"state_dict": model.state_dict(), "val_acc": val_acc},
            str(run_dir / "last_effnet.pt"),
        )

    return best_acc, val_acc


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    args   = parse_args()
    cfg    = VARIANT_CFG[args.variant]
    batch  = args.batch or cfg["batch"]
    sz     = cfg["size"]
    timm_n = cfg["timm"]

    print(f"\n{'='*55}")
    print(f"  SPIRO EfficientNetV2-{args.variant.upper()} Training")
    print(f"{'='*55}")
    print(f"  timm model: {timm_n}")
    print(f"  Input size: {sz}×{sz}")
    print(f"  Batch:      {batch}")
    print(f"  Phase 1:    {args.phase1} epochs (frozen)")
    print(f"  Phase 2:    {args.phase2} epochs (full)")
    print(f"{'='*55}\n")

    import torch
    import torch.nn as nn
    try:
        import timm
    except ImportError:
        print("❌ timm not installed: pip install timm")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Extract crops ─────────────────────────────────────────────────────────
    if not args.skip_crops:
        print("\n📦 Extracting crop patches...")
        extract_crops(DATA_DIR, CROPS_DIR)
    else:
        print("⏭️  Skipping crop extraction")

    # ── DataLoaders ───────────────────────────────────────────────────────────
    print("\n📊 Building DataLoaders...")
    train_loader = build_loaders(CROPS_DIR, "train", sz, batch, True,  args.workers)
    val_loader   = build_loaders(CROPS_DIR, "val",   sz, batch, False, args.workers)

    if train_loader is None:
        print("❌ No training crops found. Run setup_datasets.py first.")
        sys.exit(1)
    print(f"  Train batches: {len(train_loader)}")
    if val_loader:
        print(f"  Val batches:   {len(val_loader)}")

    # ── Build model ───────────────────────────────────────────────────────────
    print(f"\n🏗️  Building {timm_n}...")
    model = timm.create_model(
        timm_n,
        pretrained=True,
        num_classes=NUM_CLASSES,
        drop_rate=0.2,
        drop_path_rate=0.2,
    )
    model = model.to(device)

    run_dir = RUNS_DIR / f"spiro_effnetv2_{args.variant}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Loss with label smoothing
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1).to(device)

    # CSV logger
    csv_path = run_dir / "training_history.csv"
    csv_file = open(csv_path, "w", newline="")
    csv_w    = csv.writer(csv_file)
    csv_w.writerow(["phase", "epoch", "train_loss", "train_acc",
                     "val_loss", "val_acc", "lr"])

    best_acc = 0.0
    t0 = time.perf_counter()

    # ── Phase 1: Frozen backbone ──────────────────────────────────────────────
    if args.phase1 > 0:
        print(f"\n🔒 Phase 1: Training head only ({args.phase1} epochs)")
        # Freeze all except classifier
        for name, p in model.named_parameters():
            if "classifier" not in name and "head" not in name:
                p.requires_grad = False
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  Trainable params: {trainable:,}")

        opt1  = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=5e-4, weight_decay=1e-4,
        )
        sched1 = torch.optim.lr_scheduler.CosineAnnealingLR(opt1, T_max=args.phase1, eta_min=1e-6)
        val_l  = val_loader or train_loader
        best_acc, _ = train_phase(
            model, train_loader, val_l, opt1, sched1,
            criterion, device, args.phase1, "phase1",
            best_acc, run_dir, csv_w,
        )

    # ── Phase 2: Full fine-tune ───────────────────────────────────────────────
    print(f"\n🔓 Phase 2: Full fine-tune ({args.phase2} epochs)")
    for p in model.parameters():
        p.requires_grad = True
    trainable = sum(p.numel() for p in model.parameters())
    print(f"  Trainable params: {trainable:,}")

    opt2   = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    sched2 = torch.optim.lr_scheduler.CosineAnnealingLR(opt2, T_max=args.phase2, eta_min=1e-6)
    val_l  = val_loader or train_loader
    best_acc, final_acc = train_phase(
        model, train_loader, val_l, opt2, sched2,
        criterion, device, args.phase2, "phase2",
        best_acc, run_dir, csv_w,
    )

    csv_file.close()
    elapsed = (time.perf_counter() - t0) / 60

    # ── Save to Drive ─────────────────────────────────────────────────────────
    save_dir = Path(args.save_dir)
    if save_dir.exists():
        best_pt = run_dir / "best_effnet.pt"
        if best_pt.exists():
            dst = save_dir / f"effnet_{args.variant}_best.pt"
            shutil.copy2(best_pt, dst)
            print(f"\n  ✅ Saved to Drive: {dst}")
        shutil.copy2(csv_path, save_dir / f"effnet_{args.variant}_history.csv")

    print(f"\n{'='*55}")
    print(f"  EfficientNetV2 Training Complete")
    print(f"{'='*55}")
    print(f"  Best val accuracy: {best_acc:.4f}")
    print(f"  Time:              {elapsed:.1f}min")
    print(f"  Best weights:      {run_dir / 'best_effnet.pt'}")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    main()
