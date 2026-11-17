#!/usr/bin/env python3
"""
Export YOLO .pt model to ONNX for SPIRO pipeline.
Run this ONCE after getting best.pt from teammate.
"""
import sys
from pathlib import Path

# Add backend to path
backend_dir = Path(__file__).parent
sys.path.insert(0, str(backend_dir))

def export_yolo_to_onnx(pt_path: str, output_path: str, imgsz: int = 640):
    """Export YOLO model to ONNX without NMS (for ONNX Runtime compatibility)."""
    try:
        from ultralytics import YOLO
    except ImportError:
        print("Installing ultralytics...")
        import subprocess
        subprocess.run([sys.executable, "-m", "pip", "install", "ultralytics"], check=True)
        from ultralytics import YOLO

    print(f"Loading model from: {pt_path}")
    model = YOLO(pt_path)

    print(f"Exporting to ONNX: {output_path}")
    print(f"Image size: {imgsz}")
    print("Settings: NMS disabled, opset 17, simplify=True")

    # Export with NMS disabled for ONNX Runtime compatibility
    model.export(
        format="onnx",
        imgsz=imgsz,
        opset=17,
        simplify=True,
        nms=False,  # Critical: disable NMS in export
        half=False,  # Keep FP32 for compatibility
        dynamic=False,
    )

    # Find the exported file (ultralytics creates it next to .pt)
    pt_file = Path(pt_path)
    exported = pt_file.with_suffix(".onnx")

    if exported.exists():
        import shutil
        shutil.move(str(exported), output_path)
        print(f"✅ Exported successfully to: {output_path}")
    else:
        # Try default location
        default_export = Path("runs/detect/export") / exported.name
        if default_export.exists():
            import shutil
            shutil.move(str(default_export), output_path)
            print(f"✅ Exported successfully to: {output_path}")
        else:
            print(f"❌ Export file not found at expected locations")
            print(f"   Checked: {exported}")
            print(f"   Checked: {default_export}")
            return False

    # Verify the ONNX model
    try:
        import onnx
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        print(f"✅ ONNX model validation passed")
        print(f"   Input: {onnx_model.graph.input[0].name} {onnx_model.graph.input[0].type.tensor_type.shape.dim}")
        print(f"   Output: {onnx_model.graph.output[0].name} {onnx_model.graph.output[0].type.tensor_type.shape.dim}")
    except Exception as e:
        print(f"⚠️  ONNX validation warning: {e}")

    return True


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Export YOLO .pt to ONNX for SPIRO")
    parser.add_argument("--weights", required=True, help="Path to best.pt file")
    parser.add_argument("--output", default="models/exports/spiro_yolov11n_best.onnx", help="Output ONNX path")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    args = parser.parse_args()

    success = export_yolo_to_onnx(args.weights, args.output, args.imgsz)
    sys.exit(0 if success else 1)