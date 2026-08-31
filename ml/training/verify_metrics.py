#!/usr/bin/env python3
"""
SPIRO ML — training/verify_metrics.py
Evaluate a trained verifier on a dataset split.

Usage
-----
    python training/verify_metrics.py \\
        --onnx models/exports/spiro_effnetv2_s_best.onnx \\
        --config configs/verification/effnetv2_s.yaml \\
        --split test

    python training/verify_metrics.py \\
        --onnx models/exports/spiro_effnetv2_s_best.onnx \\
        --config configs/verification/effnetv2_s.yaml \\
        --split val \\
        --output-dir reports/verification \\
        --plots
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate EfficientNetV2 verifier")
    p.add_argument("--onnx",   required=True, help="Path to .onnx model")
    p.add_argument("--config", required=True, help="Verification config YAML")
    p.add_argument("--split",  default="test", choices=["train", "val", "test"])
    p.add_argument("--output-dir", default="reports/verification")
    p.add_argument("--plots",  action="store_true", help="Generate visualisation charts")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    import numpy as np
    import torch
    import onnxruntime as ort
    from torch.utils.data import DataLoader

    from lib.ml.core.logger import get_logger
    from lib.ml.verification.verify_config import VerifyConfig
    from lib.ml.verification.training.verify_dataset import VerifyDataset, build_val_transforms
    from lib.ml.verification.evaluation.verify_metrics import VerifyMetrics

    log = get_logger("verify_metrics")
    cfg = VerifyConfig.load(args.config)

    # Build dataset
    dataset = VerifyDataset(
        root=Path(cfg.dataset.root),
        split=args.split,
        transform=build_val_transforms(cfg),
    )
    loader = DataLoader(
        dataset,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=0,
    )

    if len(dataset) == 0:
        log.error(f"No samples found for split '{args.split}'")
        sys.exit(1)

    log.info(f"Evaluating on {args.split}: {len(dataset)} samples")

    # ORT session
    sess = ort.InferenceSession(
        args.onnx, providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
    )
    input_name = sess.get_inputs()[0].name

    all_logits, all_labels = [], []
    for images, labels in loader:
        batch_np = images.numpy()
        logits = sess.run(None, {input_name: batch_np})[0]
        all_logits.append(torch.from_numpy(logits))
        all_labels.append(labels)

    logits_t = torch.cat(all_logits)
    labels_t = torch.cat(all_labels)

    metrics = VerifyMetrics(
        num_classes=cfg.model.num_classes,
        output_dir=Path(args.output_dir),
    )
    report = metrics.evaluate(
        logits_t, labels_t,
        split=args.split,
        save_reports=True,
    )

    if args.plots:
        log.info("Generating additional plots from training history...")
        hist_csv = Path(cfg.logging.csv_path)
        if hist_csv.exists():
            import pandas as pd
            hist = pd.read_csv(hist_csv)
            history = {col: hist[col].tolist() for col in hist.columns if col != "epoch"}
            metrics.plot_training_curves(history)

    log.info(f"\n  Top-1:    {report['top1_accuracy']:.4f}")
    log.info(f"  Top-5:    {report['top5_accuracy']:.4f}")
    log.info(f"  F1 macro: {report['f1_macro']:.4f}")
    log.info(f"  ROC-AUC:  {report.get('roc_auc_macro', 'N/A')}")
    log.info(f"\nReport → {args.output_dir}")


if __name__ == "__main__":
    main()
