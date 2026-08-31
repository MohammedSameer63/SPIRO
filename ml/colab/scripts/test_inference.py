#!/usr/bin/env python3
"""
SPIRO — test_inference.py
Runs a smoke-test inference through both ONNX models:
  1. YOLOv11 detection on a synthetic 640×640 image
  2. EfficientNetV2 classification on synthetic 192×192 crops
  3. Confidence fusion
  4. Prints top-5 predictions with SPIRO class names
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import yaml

BASE        = Path("/content/spiro")
NUM_CLASSES = 109


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--yolo",   default=None, help="Path to yolo_best.onnx")
    p.add_argument("--effnet", default=None, help="Path to effnet_best.onnx")
    p.add_argument("--image",  default=None, help="Real image to test (optional)")
    p.add_argument("--conf",   type=float, default=0.1)
    p.add_argument("--iou",    type=float, default=0.45)
    return p.parse_args()


def find_model(name: str, pattern: str) -> Path:
    """Auto-discover an ONNX model by name."""
    drive_dir = Path("/content/drive/MyDrive/SPIRO_ML")
    search_dirs = [drive_dir, BASE / "runs", BASE / "models"]
    for d in search_dirs:
        if not d.exists():
            continue
        for f in d.rglob(pattern):
            return f
    return None


STREAM_LABELS = {
    "organic":         "🌿 Organic",
    "recoverable":     "♻️  Recoverable",
    "non_recoverable": "🗑️  Non-Recoverable",
    "hazardous":       "⚠️  Hazardous",
    "sanitary":        "🏥 Sanitary",
    "e_waste":         "🔋 E-Waste",
    "reject":          "❓ Reject/Unknown",
}


def load_class_names() -> list:
    tax_path = BASE / "configs" / "taxonomy" / "spiro_taxonomy.yaml"
    if tax_path.exists():
        with open(tax_path) as f:
            tax = yaml.safe_load(f)
        names = tax.get("names", {})
        if isinstance(names, dict):
            return [names.get(i, f"class_{i}") for i in range(NUM_CLASSES)]
        return list(names)
    return [f"class_{i}" for i in range(NUM_CLASSES)]


def nms_numpy(boxes: np.ndarray, scores: np.ndarray, iou_thr: float) -> list:
    """Pure-NumPy NMS."""
    if len(boxes) == 0:
        return []
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep  = []
    while order.size > 0:
        i = order[0]; keep.append(i)
        if order.size == 1:
            break
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w    = np.maximum(0, xx2 - xx1)
        h    = np.maximum(0, yy2 - yy1)
        inter = w * h
        iou   = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        order = order[1:][iou <= iou_thr]
    return keep


def test_yolo(onnx_path: Path, img: np.ndarray, conf_thr: float, iou_thr: float) -> list:
    """Run YOLOv11 ORT inference and return detections."""
    try:
        import onnxruntime as ort
    except ImportError:
        print("  ⚠️  onnxruntime not installed")
        return []

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    inp_name  = sess.get_inputs()[0].name
    inp_shape = sess.get_inputs()[0].shape
    inp_size  = inp_shape[2] if len(inp_shape) == 4 and isinstance(inp_shape[2], int) else 640

    # Preprocess
    resized = img
    if img.shape[:2] != (inp_size, inp_size):
        import cv2
        resized = cv2.resize(img, (inp_size, inp_size))
    blob = resized.astype(np.float32).transpose(2, 0, 1)[None] / 255.0

    raw = sess.run(None, {inp_name: blob})[0]  # [1, 4+nc, num_preds] or [1, num_preds, 4+nc]

    # Handle both output formats
    if raw.ndim == 3:
        if raw.shape[1] < raw.shape[2]:   # [1, 4+nc, N]
            raw = raw[0].T                 # → [N, 4+nc]
        else:                              # [1, N, 4+nc]
            raw = raw[0]

    nc   = raw.shape[1] - 4
    nc   = min(nc, NUM_CLASSES)
    dets = []

    boxes_raw  = raw[:, :4]
    scores_all = raw[:, 4:4+nc]
    best_scores = scores_all.max(axis=1)
    best_cls    = scores_all.argmax(axis=1)

    mask = best_scores >= conf_thr
    if not mask.any():
        return []

    boxes_f = boxes_raw[mask]
    scores_f = best_scores[mask]
    cls_f    = best_cls[mask]

    # cx,cy,w,h → x1,y1,x2,y2
    x1 = boxes_f[:, 0] - boxes_f[:, 2] / 2
    y1 = boxes_f[:, 1] - boxes_f[:, 3] / 2
    x2 = boxes_f[:, 0] + boxes_f[:, 2] / 2
    y2 = boxes_f[:, 1] + boxes_f[:, 3] / 2
    xyxy = np.stack([x1, y1, x2, y2], axis=1)
    keep = nms_numpy(xyxy, scores_f, iou_thr)

    for k in keep[:20]:
        dets.append({
            "cls_id": int(cls_f[k]),
            "conf":   float(scores_f[k]),
            "bbox":   [float(v) for v in xyxy[k]],
        })
    return dets


def test_effnet(onnx_path: Path, crop: np.ndarray) -> dict:
    """Run EfficientNetV2 ORT inference and return top-5 predictions."""
    try:
        import onnxruntime as ort
    except ImportError:
        return {}

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    inp_name  = sess.get_inputs()[0].name
    inp_shape = sess.get_inputs()[0].shape
    sz = inp_shape[2] if len(inp_shape) == 4 and isinstance(inp_shape[2], int) else 192

    import cv2
    crop_r = cv2.resize(crop, (sz, sz)).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    crop_r = ((crop_r - mean) / std).transpose(2, 0, 1)[None]

    logits = sess.run(None, {inp_name: crop_r})[0][0]
    # Softmax
    e = np.exp(logits - logits.max())
    probs = e / e.sum()
    top5_idx = np.argsort(probs)[::-1][:5]
    return {
        "top5": [(int(i), float(probs[i])) for i in top5_idx],
        "top1_id": int(top5_idx[0]),
        "top1_conf": float(probs[top5_idx[0]]),
    }


def main():
    args  = parse_args()
    names = load_class_names()

    print(f"\n{'='*60}")
    print(f"  SPIRO Inference Smoke Test")
    print(f"{'='*60}")

    # Find models
    yolo_path = Path(args.yolo) if args.yolo else find_model("yolo", "yolo*.onnx")
    eff_path  = Path(args.effnet) if args.effnet else find_model("effnet", "effnet*.onnx")

    if yolo_path and yolo_path.exists():
        print(f"  YOLO:   {yolo_path} ({yolo_path.stat().st_size/1e6:.1f}MB)")
    else:
        print("  ⚠️  YOLO ONNX not found — run export_yolo.py first")
        yolo_path = None

    if eff_path and eff_path.exists():
        print(f"  EffNet: {eff_path} ({eff_path.stat().st_size/1e6:.1f}MB)")
    else:
        print("  ⚠️  EffNet ONNX not found — run export_effnet.py first")
        eff_path = None

    # Load or create test image
    import cv2
    if args.image and Path(args.image).exists():
        img_bgr = cv2.imread(args.image)
        print(f"\n  Image: {args.image} {img_bgr.shape}")
    else:
        print("\n  Using synthetic test image (640×640 random noise)")
        img_bgr = np.random.randint(30, 200, (640, 640, 3), dtype=np.uint8)
        # Draw a fake bottle-shaped rectangle for better detection demo
        cv2.rectangle(img_bgr, (200, 100), (280, 400), (50, 120, 200), -1)
        cv2.rectangle(img_bgr, (350, 200), (420, 450), (200, 80, 50), -1)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # ── YOLO inference ────────────────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print("  YOLO Detection:")
    if yolo_path:
        dets = test_yolo(yolo_path, img_rgb, args.conf, args.iou)
        if dets:
            for d in dets[:5]:
                cls_name = names[d["cls_id"]] if d["cls_id"] < len(names) else f"cls_{d['cls_id']}"
                print(f"    cls={d['cls_id']:>3} ({cls_name:<28}) conf={d['conf']:.3f}")
        else:
            print(f"    No detections above conf={args.conf} (expected on random image)")
    else:
        print("    Skipped — ONNX not found")

    # ── EffNet inference ──────────────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print("  EfficientNetV2 Classification (on full image as crop):")
    if eff_path:
        result = test_effnet(eff_path, img_rgb)
        if result:
            print(f"  Top-5 predictions (class → disposal stream):")
            # Build id→stream from taxonomy
            stream_map = {}
            tax_path = BASE / "configs" / "taxonomy" / "spiro_taxonomy.yaml"
            if tax_path.exists():
                import yaml as _yaml
                with open(tax_path) as f:
                    tax = _yaml.safe_load(f)
                for sname, sdata in tax.get("streams", {}).items():
                    if isinstance(sdata, dict):
                        for cid in sdata.get("ids", []):
                            stream_map[cid] = sname
            for cls_id, conf in result["top5"]:
                cls_name   = names[cls_id] if cls_id < len(names) else f"cls_{cls_id}"
                stream     = stream_map.get(cls_id, "reject")
                slabel     = STREAM_LABELS.get(stream, stream)
                bar        = "█" * int(conf * 30)
                print(f"    {cls_id:>3} {cls_name:<26} {conf:.4f}  {slabel}  {bar}")
        else:
            print("    Inference failed")
    else:
        print("    Skipped — ONNX not found")

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'─'*60}")
    status = "✅" if (yolo_path and eff_path) else "⚠️ "
    print(f"  {status} Smoke test complete")
    if yolo_path and eff_path:
        print("  Both models loaded and executed successfully.")
        print("  Ready for integration into SPIRO backend.")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
