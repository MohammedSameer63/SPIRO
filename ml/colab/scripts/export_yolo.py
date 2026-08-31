#!/usr/bin/env python3
"""
SPIRO — export_yolo.py
Exports YOLOv11 best.pt to ONNX (opset 17) and verifies the export.
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

BASE = Path("/content/spiro")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--weights",  required=True, help="Path to best.pt")
    p.add_argument("--img-size", type=int, default=640)
    p.add_argument("--opset",    type=int, default=17)
    p.add_argument("--save-dir", default="/content/drive/MyDrive/SPIRO_ML")
    p.add_argument("--verify",   action="store_true", default=True)
    p.add_argument("--simplify", action="store_true", default=True)
    return p.parse_args()


def main():
    args = parse_args()
    pt_path = Path(args.weights)
    if not pt_path.exists():
        # Try to find best.pt in runs
        candidates = list((BASE / "runs" / "detect").rglob("best.pt"))
        if candidates:
            pt_path = candidates[-1]
            print(f"  Found: {pt_path}")
        else:
            print(f"❌ Weights not found: {args.weights}")
            sys.exit(1)

    print(f"\n📤 Exporting YOLOv11 ONNX")
    print(f"   Source:   {pt_path}")
    print(f"   ImgSize:  {args.img_size}")
    print(f"   Opset:    {args.opset}")

    from ultralytics import YOLO
    model = YOLO(str(pt_path))

    # Export
    onnx_result = model.export(
        format="onnx",
        imgsz=args.img_size,
        opset=args.opset,
        dynamic=True,
        simplify=args.simplify,
        half=False,
    )

    onnx_path = Path(str(onnx_result))
    if not onnx_path.exists():
        # Fallback: look next to pt
        onnx_path = pt_path.with_suffix(".onnx")

    if not onnx_path.exists():
        print("❌ ONNX export failed")
        sys.exit(1)

    size_mb = onnx_path.stat().st_size / 1e6
    print(f"   ✅ ONNX: {onnx_path} ({size_mb:.1f} MB)")

    # Validate ONNX graph
    try:
        import onnx
        model_onnx = onnx.load(str(onnx_path))
        onnx.checker.check_model(model_onnx)
        print(f"   ✅ ONNX graph valid (opset {args.opset})")
    except Exception as e:
        print(f"   ⚠️  ONNX validation: {e}")

    # Numerical verification
    if args.verify:
        try:
            import onnxruntime as ort
            import numpy as np
            sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
            inp_name = sess.get_inputs()[0].name
            dummy = np.zeros((1, 3, args.img_size, args.img_size), dtype=np.float32)
            out = sess.run(None, {inp_name: dummy})
            print(f"   ✅ ORT inference OK | output shape: {out[0].shape}")
        except Exception as e:
            print(f"   ⚠️  ORT verification: {e}")

    # Copy to Drive
    save_dir = Path(args.save_dir)
    if save_dir.exists() or str(save_dir).startswith("/content/drive"):
        save_dir.mkdir(parents=True, exist_ok=True)
        dst = save_dir / "yolo_best.onnx"
        shutil.copy2(onnx_path, dst)
        print(f"   ✅ Saved to Drive: {dst}")

        # Write metadata
        meta = {
            "architecture": "yolov11",
            "num_classes": 109,
            "input_size": args.img_size,
            "opset": args.opset,
            "source_pt": str(pt_path),
        }
        (save_dir / "yolo_metadata.json").write_text(json.dumps(meta, indent=2))

    print(f"\n✅ YOLOv11 ONNX export complete: {onnx_path}")


if __name__ == "__main__":
    main()
