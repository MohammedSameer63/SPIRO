#!/usr/bin/env python3
"""
SPIRO — export_effnet.py
Exports EfficientNetV2 best_effnet.pt to ONNX (opset 17).
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

BASE = Path("/content/spiro")

VARIANT_CFG = {
    "b0": {"timm": "tf_efficientnetv2_b0", "size": 192},
    "b1": {"timm": "tf_efficientnetv2_b1", "size": 240},
    "s":  {"timm": "tf_efficientnetv2_s",  "size": 300},
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--weights",  default=None, help="Path to best_effnet.pt")
    p.add_argument("--variant",  default="b0", choices=list(VARIANT_CFG.keys()))
    p.add_argument("--opset",    type=int, default=17)
    p.add_argument("--save-dir", default="/content/drive/MyDrive/SPIRO_ML")
    p.add_argument("--verify",   action="store_true", default=True)
    return p.parse_args()


def main():
    args = parse_args()
    cfg  = VARIANT_CFG[args.variant]
    timm_name = cfg["timm"]
    input_size = cfg["size"]
    NUM_CLASSES = 109

    # Find weights
    pt_path = None
    if args.weights:
        pt_path = Path(args.weights)
    if pt_path is None or not pt_path.exists():
        candidates = list((BASE / "runs" / "verify").rglob("best_effnet.pt"))
        if candidates:
            pt_path = sorted(candidates)[-1]
            print(f"  Auto-found: {pt_path}")
        else:
            print("❌ best_effnet.pt not found. Train first: python scripts/train_effnet.py")
            sys.exit(1)

    print(f"\n📤 Exporting EfficientNetV2-{args.variant.upper()} to ONNX")
    print(f"   Source:     {pt_path}")
    print(f"   timm model: {timm_name}")
    print(f"   Input size: {input_size}×{input_size}")
    print(f"   Classes:    {NUM_CLASSES}")

    import torch
    import timm

    # Rebuild model
    model = timm.create_model(timm_name, pretrained=False, num_classes=NUM_CLASSES)

    # Load weights
    ckpt = torch.load(str(pt_path), map_location="cpu", weights_only=False)
    state = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"   ⚠️  Missing keys: {len(missing)}")
    if unexpected:
        print(f"   ⚠️  Unexpected keys: {len(unexpected)}")

    model.eval()
    val_acc = ckpt.get("val_acc", 0.0) if isinstance(ckpt, dict) else 0.0
    print(f"   Val accuracy: {val_acc:.4f}")

    # Export
    dummy = torch.zeros(1, 3, input_size, input_size)
    onnx_path = pt_path.parent / f"effnet_{args.variant}_best.onnx"

    torch.onnx.export(
        model,
        dummy,
        str(onnx_path),
        input_names=["images"],
        output_names=["logits"],
        dynamic_axes={"images": {0: "batch_size"}, "logits": {0: "batch_size"}},
        opset_version=args.opset,
        do_constant_folding=True,
        export_params=True,
    )

    # Validate
    try:
        import onnx
        m = onnx.load(str(onnx_path))
        onnx.checker.check_model(m)
        print(f"   ✅ ONNX graph valid")
    except Exception as e:
        print(f"   ⚠️  ONNX validation: {e}")

    # Embed metadata
    try:
        import onnx
        m = onnx.load(str(onnx_path))
        meta_map = {
            "architecture": "efficientnetv2",
            "variant": args.variant,
            "timm_name": timm_name,
            "num_classes": str(NUM_CLASSES),
            "input_size": str(input_size),
            "val_accuracy": str(round(val_acc, 4)),
        }
        for k, v in meta_map.items():
            prop = m.metadata_props.add()
            prop.key = k
            prop.value = v
        onnx.save(m, str(onnx_path))
        print(f"   ✅ Metadata embedded")
    except Exception as e:
        print(f"   ⚠️  Metadata embedding: {e}")

    # ORT verification
    if args.verify:
        try:
            import onnxruntime as ort
            import numpy as np
            sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
            inp  = sess.get_inputs()[0].name
            out  = sess.run(None, {inp: np.zeros((1, 3, input_size, input_size), dtype=np.float32)})
            print(f"   ✅ ORT inference OK | output: {out[0].shape} (expected [1,{NUM_CLASSES}])")
            assert out[0].shape == (1, NUM_CLASSES), f"Shape mismatch: {out[0].shape}"
        except Exception as e:
            print(f"   ⚠️  ORT verification: {e}")

    size_mb = onnx_path.stat().st_size / 1e6
    print(f"   ✅ ONNX: {onnx_path} ({size_mb:.1f} MB)")

    # Copy to Drive
    save_dir = Path(args.save_dir)
    if save_dir.exists() or str(save_dir).startswith("/content/drive"):
        save_dir.mkdir(parents=True, exist_ok=True)
        dst = save_dir / "effnet_best.onnx"
        shutil.copy2(onnx_path, dst)
        meta = {
            "architecture": "efficientnetv2",
            "variant": args.variant,
            "timm_name": timm_name,
            "num_classes": NUM_CLASSES,
            "input_size": input_size,
            "val_accuracy": val_acc,
            "opset": args.opset,
        }
        (save_dir / "effnet_metadata.json").write_text(json.dumps(meta, indent=2))
        print(f"   ✅ Saved to Drive: {dst}")

    print(f"\n✅ EfficientNetV2 export complete: {onnx_path}\n")


if __name__ == "__main__":
    main()
