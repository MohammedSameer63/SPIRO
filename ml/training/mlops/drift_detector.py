#!/usr/bin/env python3
"""SPIRO ML — training/mlops/drift_detector.py
Run input/prediction drift detection.

Usage
-----
    python training/mlops/drift_detector.py \
        --model-id yolov11s \
        --baseline-confidences path/to/baseline_confs.npy \
        --current-confidences path/to/current_confs.npy
"""
import argparse, sys
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

def main():
    from lib.ml.mlops.drift.drift_detector import DriftDetector
    from lib.ml.core.logger import get_logger
    log = get_logger("drift_detector")

    p = argparse.ArgumentParser()
    p.add_argument("--model-id", required=True)
    p.add_argument("--version", default="")
    p.add_argument("--baseline-confidences", required=True)
    p.add_argument("--current-confidences", required=True)
    p.add_argument("--baseline-path", default=None)
    args = p.parse_args()

    baseline = np.load(args.baseline_confidences)
    current  = np.load(args.current_confidences)

    detector = DriftDetector()
    detector.set_baseline(confidences=baseline)
    report = detector.detect(
        confidences=current,
        model_id=args.model_id,
        version=args.version,
    )
    print(f"\nDrift score:   {report.overall_drift_score:.4f}")
    print(f"Severity:      {report.overall_severity}")
    print(f"Any drift:     {report.any_drifted}")
    print(f"Recommendation: {report.recommendation}")
    out = detector.export_report(args.model_id)
    print(f"Report: {out}")

if __name__ == "__main__":
    main()
