#!/usr/bin/env python3
"""
SPIRO — package_models.py
Packages trained ONNX models + configs into spiro_models.zip for download.

Contents of spiro_models.zip:
  models/
    yolo_best.onnx
    effnet_best.onnx
    yolo_metadata.json
    effnet_metadata.json
  configs/
    taxonomy/spiro_taxonomy.yaml
    production/pipeline.yaml
  MODELS_README.md
"""
import hashlib
import json
import shutil
import sys
import zipfile
from pathlib import Path

BASE      = Path("/content/spiro")
DRIVE_DIR = Path("/content/drive/MyDrive/SPIRO_ML")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def find_file(name_patterns: list, search_dirs: list) -> Path:
    for d in search_dirs:
        d = Path(d)
        if not d.exists():
            continue
        for pattern in name_patterns:
            matches = list(d.rglob(pattern))
            if matches:
                return sorted(matches)[-1]
    return None


def write_readme(tmp: Path, yolo_meta: dict, eff_meta: dict) -> None:
    md = f"""# SPIRO ML — Trained Models

## Contents

| File | Size | Description |
|---|---|---|
| `models/yolo_best.onnx` | — | YOLOv11 detection model (ONNX opset 17) |
| `models/effnet_best.onnx` | — | EfficientNetV2 verifier (ONNX opset 17) |
| `configs/taxonomy/spiro_taxonomy.yaml` | — | 109-class SPIRO taxonomy |
| `configs/production/pipeline.yaml` | — | Production inference config |

## Model Details

### YOLOv11 (Detection)
- Architecture: {yolo_meta.get('architecture', 'yolov11')}
- Input size: {yolo_meta.get('input_size', 640)}×{yolo_meta.get('input_size', 640)}
- Classes: {yolo_meta.get('num_classes', 109)}
- ONNX opset: {yolo_meta.get('opset', 17)}

### EfficientNetV2 (Verification)
- Architecture: {eff_meta.get('architecture', 'efficientnetv2')}
- Variant: {eff_meta.get('variant', 'b0')}
- Input size: {eff_meta.get('input_size', 192)}×{eff_meta.get('input_size', 192)}
- Classes: {eff_meta.get('num_classes', 109)}
- Val accuracy: {eff_meta.get('val_accuracy', 'N/A')}

## Integration

```python
from lib.ml.pipeline import SPIROPipeline

pipeline = SPIROPipeline.from_config("configs/production/pipeline.yaml")
# Copy yolo_best.onnx → models/serve/yolo_active.onnx
# Copy effnet_best.onnx → models/serve/effnet_active.onnx
result = pipeline.infer("image.jpg")
```

## Trained on

- TACO (litter detection dataset)
- TrashNet (6-class waste classification)
- ZeroWaste-f (fine-grained recyclables)
- Garbage Classification (Kaggle)
- Recyclable & Household Waste Classification

109 SPIRO waste classes — 22 waste groups — 7 disposal streams
"""
    (tmp / "MODELS_README.md").write_text(md)


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--save-dir", default=str(DRIVE_DIR))
    p.add_argument("--output",   default="spiro_models.zip")
    args = p.parse_args()

    save_dir = Path(args.save_dir)
    search_dirs = [save_dir, BASE / "runs"]

    # Find ONNX models
    yolo_onnx = find_file(["yolo_best.onnx", "yolo*.onnx", "best.onnx"], search_dirs)
    eff_onnx  = find_file(["effnet_best.onnx", "effnet*.onnx", "best_effnet.onnx"], search_dirs)

    print(f"\n{'='*60}")
    print(f"  SPIRO Model Packager")
    print(f"{'='*60}")

    if not yolo_onnx:
        print("  ⚠️  yolo_best.onnx not found — export with export_yolo.py")
    else:
        print(f"  YOLO:   {yolo_onnx}")

    if not eff_onnx:
        print("  ⚠️  effnet_best.onnx not found — export with export_effnet.py")
    else:
        print(f"  EffNet: {eff_onnx}")

    if not yolo_onnx and not eff_onnx:
        print("  ❌ No models found. Run export_yolo.py and export_effnet.py first.")
        sys.exit(1)

    # Load metadata
    yolo_meta = {}
    eff_meta  = {}
    for meta_f in search_dirs:
        m = Path(meta_f) / "yolo_metadata.json"
        if m.exists():
            yolo_meta = json.loads(m.read_text())
        m = Path(meta_f) / "effnet_metadata.json"
        if m.exists():
            eff_meta = json.loads(m.read_text())

    # Build temp staging dir
    tmp = Path("/tmp/spiro_package")
    if tmp.exists():
        shutil.rmtree(tmp)
    (tmp / "models").mkdir(parents=True)

    # Copy configs
    cfg_src = BASE / "configs"
    (tmp / "configs" / "taxonomy").mkdir(parents=True)
    (tmp / "configs" / "production").mkdir(parents=True)

    tax_src = cfg_src / "taxonomy" / "spiro_taxonomy.yaml"
    if tax_src.exists():
        shutil.copy2(tax_src, tmp / "configs" / "taxonomy" / "spiro_taxonomy.yaml")

    prod_src = cfg_src / "production" / "pipeline.yaml"
    if prod_src.exists():
        shutil.copy2(prod_src, tmp / "configs" / "production" / "pipeline.yaml")

    # Copy models and compute checksums
    checksums = {}
    if yolo_onnx:
        dst = tmp / "models" / "yolo_best.onnx"
        shutil.copy2(yolo_onnx, dst)
        checksums["yolo_best.onnx"] = sha256(dst)
        if yolo_meta:
            (tmp / "models" / "yolo_metadata.json").write_text(json.dumps(yolo_meta, indent=2))

    if eff_onnx:
        dst = tmp / "models" / "effnet_best.onnx"
        shutil.copy2(eff_onnx, dst)
        checksums["effnet_best.onnx"] = sha256(dst)
        if eff_meta:
            (tmp / "models" / "effnet_metadata.json").write_text(json.dumps(eff_meta, indent=2))

    # Write checksums
    (tmp / "models" / "checksums.json").write_text(json.dumps(checksums, indent=2))

    # Write README
    write_readme(tmp, yolo_meta, eff_meta)

    # Create ZIP
    zip_path = save_dir / args.output
    save_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in sorted(tmp.rglob("*")):
            if f.is_file():
                zf.write(f, f.relative_to(tmp))

    size_mb = zip_path.stat().st_size / 1e6
    print(f"\n  ✅ Package created: {zip_path} ({size_mb:.1f} MB)")
    print(f"\n  Contents:")
    with zipfile.ZipFile(zip_path) as zf:
        for name in sorted(zf.namelist()):
            info = zf.getinfo(name)
            print(f"    {name}  ({info.file_size/1e6:.1f}MB)")

    # Also make available for direct Colab download
    local_zip = Path(f"/content/{args.output}")
    shutil.copy2(zip_path, local_zip)
    print(f"\n  📥 Download directly from Colab Files panel: {local_zip}")

    try:
        from google.colab import files
        print("\n  Triggering browser download...")
        files.download(str(local_zip))
    except Exception:
        print(f"  (Manual download: click {local_zip} in the Files panel)")

    print(f"\n{'='*60}")
    print(f"  ✅ spiro_models.zip ready!")
    print(f"  Place yolo_best.onnx → models/serve/yolo_active.onnx")
    print(f"  Place effnet_best.onnx → models/serve/effnet_active.onnx")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
